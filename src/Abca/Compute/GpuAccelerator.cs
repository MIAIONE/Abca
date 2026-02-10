// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// 模块: Compute / GpuAccelerator
// 目的: 稀疏脉冲网络的 GPU 加速（CUDA）/ CPU 多核并行回退。
//
// 加速目标: 隐藏层散射-聚集加权和（计算瓶颈）。
//   每时间步，HiddenSize 个神经元各有 FanIn 个稀疏连接，
//   需从输入中聚集活跃脉冲并加权求和。
//   GPU: 每个 CUDA 线程处理一个神经元，天然数据并行。
//   CPU: Parallel.For 跨核并行（无 CUDA 时自动回退）。
//
// 生物学类比:
//   GPU 大规模并行 ↔ 大脑皮层数千神经元同步放电计算。
//   加速的是"在哪里"计算，而非"计算什么"。
// ============================================================================

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace Abca.Compute;

/// <summary>
/// 稀疏脉冲网络的 GPU/CPU 并行加速器。
/// 主要加速隐藏层的散射-聚集加权和运算（训练/推理的计算瓶颈）。
/// </summary>
public sealed class GpuAccelerator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly bool _isGpu;
    private bool _disposed;

    // 编译后的 CUDA 核函数
    private readonly Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _hiddenKernel;

    // GPU 显存缓冲区（持久化，避免逐次分配）
    private MemoryBuffer1D<int, Stride1D.Dense>? _gpuIndices;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuWeights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuSpikes;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuResult;

    // CPU 侧引用（CPU 并行路径直接使用 AbcaNetwork 的原始平坦数组）
    private int[]? _cpuIndices;
    private float[]? _cpuWeights;

    // 输入脉冲 float 缓冲区（GPU 核函数使用 float 乘法而非 bool 分支）
    private float[]? _spikeBuffer;

    private int _hiddenSize;
    private int _fanIn;
    private int _inputSize;
    private bool _initialized;

    /// <summary>是否使用 CUDA GPU（否则 CPU 并行回退）。</summary>
    public bool IsGpu => _isGpu;

    /// <summary>计算设备名称。</summary>
    public string DeviceName => _accelerator.Name;

    /// <summary>加速器类型。</summary>
    public AcceleratorType AccelType => _accelerator.AcceleratorType;

    public GpuAccelerator()
    {
        _context = Context.Create(builder =>
        {
            builder.Cuda().CPU().Optimize(OptimizationLevel.O2);
        });

        // 优先 CUDA（NVIDIA GPU），无则回退 CPU
        Device? cuda = null;
        foreach (var device in _context.Devices)
        {
            if (device.AcceleratorType == AcceleratorType.Cuda)
            {
                cuda = device;
                break;
            }
        }
        cuda ??= _context.GetPreferredDevice(preferCPU: true);

        _accelerator = cuda.CreateAccelerator(_context);
        _isGpu = _accelerator.AcceleratorType == AcceleratorType.Cuda;

        // 编译核函数
        _hiddenKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(HiddenWeightedSumKernel);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  GPU 核函数
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU 核函数：计算一个隐藏神经元的加权输入脉冲和。
    /// result[h] = Σ_k weights[h×fanIn+k] × spikes[indices[h×fanIn+k]]
    /// 每个 GPU 线程处理一个神经元（h），天然无数据竞争。
    /// </summary>
    private static void HiddenWeightedSumKernel(
        Index1D h,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> spikes,
        ArrayView1D<float, Stride1D.Dense> result,
        int fanIn)
    {
        float sum = 0f;
        int baseIdx = h * fanIn;
        for (int k = 0; k < fanIn; k++)
        {
            sum += weights[baseIdx + k] * spikes[indices[baseIdx + k]];
        }
        result[h] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  公共 API
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// 用隐藏层的稀疏连接拓扑和权重初始化加速器。
    /// GPU: 上传连接索引和权重到显存（一次性）。
    /// CPU: 保存数组引用用于并行计算（零拷贝）。
    /// </summary>
    /// <param name="flatConnections">平坦连接索引 [HiddenSize × FanIn]，行主序。</param>
    /// <param name="flatWeights">平坦权重 [HiddenSize × FanIn]，行主序。</param>
    /// <param name="hiddenSize">隐藏层神经元数。</param>
    /// <param name="fanIn">每个神经元的固定入度。</param>
    /// <param name="inputSize">输入维度。</param>
    public void Initialize(int[] flatConnections, float[] flatWeights,
        int hiddenSize, int fanIn, int inputSize)
    {
        _hiddenSize = hiddenSize;
        _fanIn = fanIn;
        _inputSize = inputSize;
        _cpuIndices = flatConnections;
        _cpuWeights = flatWeights;
        _spikeBuffer = new float[inputSize];

        if (_isGpu && fanIn > 0 && hiddenSize > 0)
        {
            _gpuIndices?.Dispose();
            _gpuWeights?.Dispose();
            _gpuSpikes?.Dispose();
            _gpuResult?.Dispose();

            _gpuIndices = _accelerator.Allocate1D(flatConnections);
            _gpuWeights = _accelerator.Allocate1D(flatWeights);
            _gpuSpikes = _accelerator.Allocate1D<float>(inputSize);
            _gpuResult = _accelerator.Allocate1D<float>(hiddenSize);
        }
        _initialized = true;
    }

    /// <summary>
    /// 将 CPU 侧修改后的权重同步到 GPU 显存。
    /// STDP 或突触缩放修改权重后调用。
    /// CPU 路径无需操作（共享同一数组引用，零拷贝）。
    /// </summary>
    public void SyncWeights(float[] flatWeights)
    {
        if (_isGpu && _gpuWeights is not null)
            _gpuWeights.CopyFromCPU(flatWeights);
    }

    /// <summary>
    /// 计算所有隐藏神经元的加权输入脉冲和（每时间步调用一次）。
    /// GPU: CUDA 核函数，一线程一神经元。
    /// CPU: Parallel.For 多核并行（小网络退化为单线程）。
    /// </summary>
    /// <param name="inputSpikes">当步输入脉冲 [InputSize]。</param>
    /// <param name="result">输出缓冲区 [HiddenSize]：每个神经元的加权和。</param>
    public void ComputeHiddenWeightedSum(bool[] inputSpikes, float[] result)
    {
        if (!_initialized || _fanIn == 0) return;

        if (_isGpu)
            ComputeGpu(inputSpikes, result);
        else if (_hiddenSize >= 64)
            ComputeCpuParallel(inputSpikes, result);
        else
            ComputeCpuSingleThread(inputSpikes, result);
    }

    private void ComputeGpu(bool[] inputSpikes, float[] result)
    {
        // bool → float 转换（GPU 使用 float 乘法代替分支）
        for (int i = 0; i < _inputSize; i++)
            _spikeBuffer![i] = inputSpikes[i] ? 1f : 0f;

        _gpuSpikes!.CopyFromCPU(_spikeBuffer!);

        _hiddenKernel(_hiddenSize, _gpuIndices!.View, _gpuWeights!.View,
            _gpuSpikes!.View, _gpuResult!.View, _fanIn);
        // CopyToCPU 隐式等待同一 stream 上的核函数完成，无需显式 Synchronize

        _gpuResult.CopyToCPU(result);
    }

    private void ComputeCpuParallel(bool[] inputSpikes, float[] result)
    {
        int fanIn = _fanIn;
        int[] idx = _cpuIndices!;
        float[] w = _cpuWeights!;

        Parallel.For(0, _hiddenSize, h =>
        {
            float sum = 0f;
            int baseIdx = h * fanIn;
            for (int k = 0; k < fanIn; k++)
            {
                if (inputSpikes[idx[baseIdx + k]])
                    sum += w[baseIdx + k];
            }
            result[h] = sum;
        });
    }

    private void ComputeCpuSingleThread(bool[] inputSpikes, float[] result)
    {
        int fanIn = _fanIn;
        int[] idx = _cpuIndices!;
        float[] w = _cpuWeights!;

        for (int h = 0; h < _hiddenSize; h++)
        {
            float sum = 0f;
            int baseIdx = h * fanIn;
            for (int k = 0; k < fanIn; k++)
            {
                if (inputSpikes[idx[baseIdx + k]])
                    sum += w[baseIdx + k];
            }
            result[h] = sum;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gpuIndices?.Dispose();
            _gpuWeights?.Dispose();
            _gpuSpikes?.Dispose();
            _gpuResult?.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
            _disposed = true;
        }
    }
}
