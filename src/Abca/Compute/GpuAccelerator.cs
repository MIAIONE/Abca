// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Compute / GpuAccelerator
// Purpose: GPU-accelerated compute backend using ILGPU (CUDA/OpenCL).
//          Provides batch matrix-vector multiplication on GPU for the
//          hidden layer forward pass — the primary computational bottleneck.
//
// Bio-inspired note:
//   GPU parallelism mirrors biological massive parallelism:
//   thousands of neurons computing simultaneously.
//   The GPU doesn't change WHAT is computed, only WHERE.
// ============================================================================

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using Abca.Memory;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace Abca.Compute;

/// <summary>
/// GPU-accelerated compute for ABCA's most expensive operations.
/// Falls back to CPU SIMD if no GPU is available.
/// 
/// Primary acceleration target: hidden layer forward pass.
/// With 800 hidden cells × 784 input dims × 60K samples, this is
/// ~37 billion FMAs per epoch — ideal for GPU parallelism.
///
/// Bio-inspired note:
///   GPU parallelism mirrors biological massive parallelism —
///   thousands of neurons computing simultaneously.
///   The GPU doesn't change WHAT is computed, only WHERE.
/// </summary>
public sealed class GpuAccelerator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly bool _isGpu;
    private bool _disposed;

    // ── Persistent GPU buffers (avoid per-call allocation) ───────────────
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuWeights;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuInput;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuBias;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gpuResult;
    private int _cachedRows;
    private int _cachedCols;
    private bool _cachedWeightsDirty = true;

    // ── Compiled kernels ─────────────────────────────────────────────────
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int> _matVecBiasKernel;

    /// <summary>Whether the accelerator is a GPU (vs CPU fallback).</summary>
    public bool IsGpu => _isGpu;

    /// <summary>Device name.</summary>
    public string DeviceName => _accelerator.Name;

    /// <summary>Accelerator type (Cuda, OpenCL, or CPU).</summary>
    public AcceleratorType AccelType => _accelerator.AcceleratorType;

    /// <summary>
    /// Creates a GPU accelerator. Forces NVIDIA CUDA; falls back to CPU SIMD
    /// if no CUDA device is found. OpenCL (Intel iGPU) is explicitly skipped.
    /// </summary>
    public GpuAccelerator()
    {
        _context = Context.Create(builder =>
        {
            builder.Cuda().CPU().Optimize(OptimizationLevel.O2);
        });

        // Force CUDA (NVIDIA) — skip OpenCL/Intel iGPU entirely
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

        // Compile kernels
        _matVecBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int, int>(MatVecBiasKernelImpl);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  GPU Kernels
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU kernel: result[row] = dot(weights[row,:], input) + bias[row].
    /// Each GPU thread computes one output row (one neuron's response).
    /// </summary>
    private static void MatVecBiasKernelImpl(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> result,
        int cols,
        int stride)
    {
        float sum = bias[row];
        int offset = row * stride;
        for (int j = 0; j < cols; j++)
        {
            sum += weights[offset + j] * input[j];
        }
        result[row] = sum;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// GPU-accelerated matrix-vector multiply with bias.
    /// result[i] = dot(matrix.Row[i], vector) + bias[i]
    /// 
    /// If GPU is available, executes on GPU with persistent buffers.
    /// Otherwise falls back to CPU SIMD (TensorPrimitives.Dot).
    /// </summary>
    /// <summary>
    /// Uploads the weight matrix to GPU. Call once or when weights change.
    /// Avoids re-uploading every forward pass during evaluation.
    /// </summary>
    public void UploadWeights(NativeMatrix matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Cols;
        EnsureGpuBuffers(rows, cols, matrix.Stride);

        float[] temp = new float[rows * matrix.Stride];
        for (int r = 0; r < rows; r++)
        {
            matrix.GetRowReadOnly(r).CopyTo(temp.AsSpan(r * matrix.Stride, matrix.Stride));
        }
        _gpuWeights!.CopyFromCPU(temp);
    }

    /// <summary>
    /// GPU-accelerated matrix-vector multiply with bias.
    /// result[i] = dot(matrix.Row[i], vector) + bias[i]
    /// 
    /// If GPU is available, executes on GPU with persistent buffers.
    /// Otherwise falls back to CPU SIMD (TensorPrimitives.Dot).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void MatVecMulBias(
        NativeMatrix matrix, ReadOnlySpan<float> vector,
        ReadOnlySpan<float> bias, Span<float> result)
    {
        int rows = matrix.Rows;
        int cols = matrix.Cols;

        if (!_isGpu)
        {
            // CPU SIMD fallback
            SimdMath.MatVecMulBias(matrix, vector, bias, result);
            return;
        }

        // ── Ensure GPU buffers are allocated / correct size ──────────────
        EnsureGpuBuffers(rows, cols, matrix.Stride);

        // ── Upload input + bias (weights are uploaded via UploadWeights) ─
        _gpuInput!.CopyFromCPU(vector.ToArray());
        _gpuBias!.CopyFromCPU(bias.ToArray());

        // Upload weights if not pre-uploaded
        if (_cachedWeightsDirty)
        {
            UploadWeights(matrix);
            _cachedWeightsDirty = false;
        }

        // ── Execute kernel ──────────────────────────────────────────────
        _matVecBiasKernel(rows, _gpuWeights!.View, _gpuInput!.View,
            _gpuBias!.View, _gpuResult!.View, cols, matrix.Stride);
        _accelerator.Synchronize();

        // ── Download result ─────────────────────────────────────────────
        float[] cpuResult = new float[rows];
        _gpuResult.CopyToCPU(cpuResult);
        cpuResult.AsSpan(0, rows).CopyTo(result);
    }

    /// <summary>
    /// Marks GPU weight buffers as stale (call after CPU-side weight updates).
    /// Next MatVecMulBias call will re-upload weights.
    /// </summary>
    public void InvalidateWeights() => _cachedWeightsDirty = true;

    /// <summary>
    /// Ensures GPU memory buffers are allocated with the right dimensions.
    /// Buffers are cached and reused across calls.
    /// </summary>
    private void EnsureGpuBuffers(int rows, int cols, int stride)
    {
        if (_cachedRows == rows && _cachedCols == cols) return;

        _gpuWeights?.Dispose();
        _gpuInput?.Dispose();
        _gpuBias?.Dispose();
        _gpuResult?.Dispose();

        _gpuWeights = _accelerator.Allocate1D<float>(rows * stride);
        _gpuInput = _accelerator.Allocate1D<float>(cols);
        _gpuBias = _accelerator.Allocate1D<float>(rows);
        _gpuResult = _accelerator.Allocate1D<float>(rows);

        _cachedRows = rows;
        _cachedCols = cols;
        _cachedWeightsDirty = true;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gpuWeights?.Dispose();
            _gpuInput?.Dispose();
            _gpuBias?.Dispose();
            _gpuResult?.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
            _disposed = true;
        }
    }
}
