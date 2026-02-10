// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// 模块: Network / AbcaNetwork
// 目的: 稀疏、事件驱动、离散时间的脉冲网络实现，完全局部学习，无反向传播。
//       - 固定 fan-in 稀疏连接（输入→隐藏，隐藏→输出）
//       - 平坦行主序数组（GPU 友好，零拷贝权重同步）
//       - 膜电位衰减 + 阈值发放 + 折返期（LIF 模型）
//       - 每步 Top-K 侧抑制（抑制性中间神经元网络）
//       - 输入泊松脉冲编码或二值阈值脉冲
//       - 隐藏层 STDP（脉冲时序依赖可塑性）
//       - 输出层奖励调制三因子规则（资格迹 × 多巴胺信号）
//       - 自稳态内在可塑性（自适应发放阈值）
//       - 突触缩放与权值裁剪保持生物学有界性
//       - GPU (CUDA) / CPU 多核并行加速隐藏层计算
// ============================================================================

using System.Runtime.CompilerServices;
using Abca.Compute;

namespace Abca.Network;

public sealed class AbcaNetwork : IDisposable
{
    public NetworkConfig Config { get; }

    private readonly Random _rng;

    // 隐藏层稀疏连接（行主序平坦数组，GPU 友好布局）
    // _hiddenIn[h * _hFanIn + k] = 第 h 个隐藏细胞的第 k 个输入索引
    // _hiddenW [h * _hFanIn + k] = 对应的突触权重
    private readonly int[] _hiddenIn;
    private readonly float[] _hiddenW;
    private readonly int _hFanIn;

    // 输出层稀疏连接（同上）
    private readonly int[] _outputIn;
    private readonly float[] _outputW;
    private readonly float[] _outputElig; // 资格迹
    private readonly int _oFanIn;

    // 膜电位 & 折返期
    private readonly float[] _vHidden;
    private readonly float[] _vOutput;
    private readonly int[] _refHidden;
    private readonly int[] _refOutput;

    // 当步脉冲标记
    private readonly bool[] _inputSpikes;
    private readonly bool[] _hiddenSpikes;
    private readonly bool[] _outputSpikes;

    // 迹（短时记忆）
    private readonly float[] _inputTrace;
    private readonly float[] _hiddenTrace;

    // 发放计数（用于读出）
    private readonly int[] _outputSpikeCount;

    // 自稳态内在可塑性（自由能原理：最小化长期惊讶）
    private readonly float[] _thrHidden;        // 每个隐藏细胞的自适应发放阈值
    private readonly float[] _avgRateHidden;    // 运行平均发放率
    private readonly int[] _hiddenSpikeCount;   // 当前样本内发放计数

    // GPU / CPU 并行加速器
    private readonly GpuAccelerator _gpu;
    private readonly float[] _hiddenInput; // GPU/并行计算结果缓冲区
    private bool _weightsDirty;

    // 侧抑制排序缓存
    private readonly List<(float v, int idx)> _candidates = new(256);

    private int _sampleCounter;

    /// <summary>GPU 加速器实例（诊断/状态查询）。</summary>
    public GpuAccelerator Gpu => _gpu;

    public AbcaNetwork(NetworkConfig config)
    {
        Config = config;
        _rng = new Random(config.Seed);

        // 自动限制 fan-in（生物学中若输入少于目标入度，则连接全部可用输入）
        int hFanIn = Math.Min(config.HiddenFanIn, config.InputSize);
        int oFanIn = Math.Min(config.OutputFanIn, config.HiddenSize);
        _hFanIn = hFanIn;
        _oFanIn = oFanIn;

        // 隐藏层：构建平坦稀疏连接
        _hiddenIn = new int[config.HiddenSize * hFanIn];
        _hiddenW = new float[config.HiddenSize * hFanIn];
        for (int h = 0; h < config.HiddenSize; h++)
        {
            int[] indices = SampleUnique(config.InputSize, hFanIn, _rng);
            float[] weights = InitWeights(hFanIn, _rng);
            int baseIdx = h * hFanIn;
            Array.Copy(indices, 0, _hiddenIn, baseIdx, hFanIn);
            Array.Copy(weights, 0, _hiddenW, baseIdx, hFanIn);
        }

        // 输出层：构建平坦稀疏连接
        _outputIn = new int[config.OutputSize * oFanIn];
        _outputW = new float[config.OutputSize * oFanIn];
        _outputElig = new float[config.OutputSize * oFanIn];
        for (int o = 0; o < config.OutputSize; o++)
        {
            int[] indices = SampleUnique(config.HiddenSize, oFanIn, _rng);
            float[] weights = InitWeights(oFanIn, _rng);
            int baseIdx = o * oFanIn;
            Array.Copy(indices, 0, _outputIn, baseIdx, oFanIn);
            Array.Copy(weights, 0, _outputW, baseIdx, oFanIn);
        }

        // 神经元状态
        _vHidden = new float[config.HiddenSize];
        _vOutput = new float[config.OutputSize];
        _refHidden = new int[config.HiddenSize];
        _refOutput = new int[config.OutputSize];

        _inputSpikes = new bool[config.InputSize];
        _hiddenSpikes = new bool[config.HiddenSize];
        _outputSpikes = new bool[config.OutputSize];

        _inputTrace = new float[config.InputSize];
        _hiddenTrace = new float[config.HiddenSize];

        _outputSpikeCount = new int[config.OutputSize];

        // 自稳态内在可塑性初始化
        _thrHidden = new float[config.HiddenSize];
        Array.Fill(_thrHidden, config.HiddenThreshold);
        _avgRateHidden = new float[config.HiddenSize];
        Array.Fill(_avgRateHidden, config.TargetFiringRate);
        _hiddenSpikeCount = new int[config.HiddenSize];

        // GPU 加速初始化
        _gpu = new GpuAccelerator();
        _hiddenInput = new float[config.HiddenSize];
        _gpu.Initialize(_hiddenIn, _hiddenW, config.HiddenSize, hFanIn, config.InputSize);
    }

    // 用于反序列化的内部构造
    internal AbcaNetwork(NetworkConfig config,
        int[] hiddenIn, float[] hiddenW, int hFanIn,
        int[] outputIn, float[] outputW, float[] outputElig, int oFanIn,
        float[]? thrHidden = null)
    {
        Config = config;
        _rng = new Random(config.Seed);
        _hFanIn = hFanIn;
        _oFanIn = oFanIn;

        _hiddenIn = hiddenIn;
        _hiddenW = hiddenW;
        _outputIn = outputIn;
        _outputW = outputW;
        _outputElig = outputElig;

        _vHidden = new float[config.HiddenSize];
        _vOutput = new float[config.OutputSize];
        _refHidden = new int[config.HiddenSize];
        _refOutput = new int[config.OutputSize];

        _inputSpikes = new bool[config.InputSize];
        _hiddenSpikes = new bool[config.HiddenSize];
        _outputSpikes = new bool[config.OutputSize];

        _inputTrace = new float[config.InputSize];
        _hiddenTrace = new float[config.HiddenSize];

        _outputSpikeCount = new int[config.OutputSize];

        _thrHidden = thrHidden ?? new float[config.HiddenSize];
        if (thrHidden is null) Array.Fill(_thrHidden, config.HiddenThreshold);
        _avgRateHidden = new float[config.HiddenSize];
        Array.Fill(_avgRateHidden, config.TargetFiringRate);
        _hiddenSpikeCount = new int[config.HiddenSize];

        // GPU 加速初始化
        _gpu = new GpuAccelerator();
        _hiddenInput = new float[config.HiddenSize];
        _gpu.Initialize(_hiddenIn, _hiddenW, config.HiddenSize, hFanIn, config.InputSize);
    }

    // 序列化访问器（仅序列化器使用）
    internal int[] GetHiddenConnections() => _hiddenIn;
    internal float[] GetHiddenWeights() => _hiddenW;
    internal int GetHiddenFanIn() => _hFanIn;
    internal int[] GetOutputConnections() => _outputIn;
    internal float[] GetOutputWeights() => _outputW;
    internal float[] GetOutputEligibility() => _outputElig;
    internal int GetOutputFanIn() => _oFanIn;
    internal float[] GetHiddenThresholds() => _thrHidden;

    // ─────────────────────────────────────────────────────────────────────
    //  训练 / 推理（事件驱动）
    // ─────────────────────────────────────────────────────────────────────

    public bool TrainSample(ReadOnlySpan<float> input, int label)
    {
        int pred = SimulateSample(input, learn: true);
        bool correct = pred == label;

        // 奖励调制：三因子规则 + 小脑攀爬纤维教学信号
        ApplyReward(pred, label, correct);

        // 施加奖励后清除资格迹（避免跨样本污染）
        ClearEligibility();

        // 自稳态内在可塑性：调节隐藏层每个细胞的发放阈值
        // 生物学：离子通道表达调节兴奋性（Desai 等 1999）
        // 自由能原理：维持内部模型的稳定预测能力
        UpdateHomeostasis();

        _sampleCounter++;
        if (Config.SynapticScaleTarget > 0 && (_sampleCounter % Config.SynapticScalePeriod) == 0)
        {
            ScaleAll();
        }

        return correct;
    }

    public int Predict(ReadOnlySpan<float> input)
    {
        return SimulateSample(input, learn: false);
    }

    public float Evaluate(ReadOnlySpan<float> images, ReadOnlySpan<byte> labels, int count)
    {
        int correct = 0;
        int stride = Config.InputSize;
        for (int i = 0; i < count; i++)
        {
            ReadOnlySpan<float> img = images.Slice(i * stride, stride);
            if (Predict(img) == labels[i]) correct++;
        }
        return (float)correct / count;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  核心仿真
    // ─────────────────────────────────────────────────────────────────────

    private int SimulateSample(ReadOnlySpan<float> input, bool learn)
    {
        // 权重已被 STDP/突触缩放修改，同步到 GPU 显存
        if (_weightsDirty)
        {
            _gpu.SyncWeights(_hiddenW);
            _weightsDirty = false;
        }

        // 每个样本开始前完全重置神经元状态（样本间无时间连续性）
        ResetState();

        for (int t = 0; t < Config.TimeSteps; t++)
        {
            EncodeInput(input);
            UpdateHidden(learn);
            UpdateOutput(learn);
        }

        // 读出：输出层发放次数 ArgMax；若全零则用膜电位最大者
        int winner = 0;
        int bestCount = _outputSpikeCount[0];
        for (int i = 1; i < _outputSpikeCount.Length; i++)
        {
            if (_outputSpikeCount[i] > bestCount)
            {
                bestCount = _outputSpikeCount[i];
                winner = i;
            }
        }
        if (bestCount == 0)
        {
            float bestV = _vOutput[0];
            winner = 0;
            for (int i = 1; i < _vOutput.Length; i++)
            {
                if (_vOutput[i] > bestV)
                {
                    bestV = _vOutput[i];
                    winner = i;
                }
            }
        }
        return winner;
    }

    private void EncodeInput(ReadOnlySpan<float> input)
    {
        if (Config.UseRateCoding)
        {
            for (int i = 0; i < input.Length; i++)
            {
                float p = input[i] * Config.InputRateScale;
                _inputSpikes[i] = _rng.NextDouble() < p;
            }
        }
        else
        {
            float thr = Config.SpikeThreshold;
            for (int i = 0; i < input.Length; i++)
            {
                _inputSpikes[i] = input[i] > thr;
            }
        }

        // 输入迹衰减并写入当前脉冲
        float traceDecay = Config.TraceDecay;
        for (int i = 0; i < _inputTrace.Length; i++)
        {
            _inputTrace[i] = _inputTrace[i] * traceDecay + (_inputSpikes[i] ? 1f : 0f);
        }
    }

    private void UpdateHidden(bool learn)
    {
        Array.Clear(_hiddenSpikes, 0, _hiddenSpikes.Length);
        _candidates.Clear();

        float decay = Config.HiddenDecay;

        // GPU / CPU 并行：计算所有隐藏神经元的加权输入脉冲和
        _gpu.ComputeHiddenWeightedSum(_inputSpikes, _hiddenInput);

        for (int h = 0; h < Config.HiddenSize; h++)
        {
            if (_refHidden[h] > 0)
            {
                _refHidden[h]--;
                _vHidden[h] = Config.ResetPotential;
                continue;
            }

            // 膜电位 = 衰减后的旧电位 + GPU/并行计算的突触输入
            float v = _vHidden[h] * decay + _hiddenInput[h];
            _vHidden[h] = v;

            // 使用自适应阈值（自稳态内在可塑性）
            if (v >= _thrHidden[h])
            {
                _candidates.Add((v, h));
            }
        }

        // 侧抑制：仅保留 top-K（生物学：抑制性中间神经元网络）
        if (_candidates.Count > 0)
        {
            int keep = Math.Min(Config.HiddenMaxSpikesPerStep, _candidates.Count);
            _candidates.Sort((a, b) => b.v.CompareTo(a.v));
            for (int i = 0; i < keep; i++)
            {
                int h = _candidates[i].idx;
                _hiddenSpikes[h] = true;
                _hiddenSpikeCount[h]++;
                _vHidden[h] = Config.ResetPotential;
                _refHidden[h] = Config.HiddenRefractory;

                if (learn)
                {
                    ApplyHiddenStdp(h);
                }
            }
            // STDP 修改了权重，标记 GPU 需要同步（下个样本开始时）
            if (learn) _weightsDirty = true;
        }

        // 更新隐藏迹
        float traceDecay = Config.TraceDecay;
        for (int h = 0; h < _hiddenTrace.Length; h++)
        {
            _hiddenTrace[h] = _hiddenTrace[h] * traceDecay + (_hiddenSpikes[h] ? 1f : 0f);
        }
    }

    private void ApplyHiddenStdp(int h)
    {
        int baseIdx = h * _hFanIn;
        float lrLtp = Config.HiddenLtpLearningRate;
        float lrLtd = Config.HiddenLtdLearningRate;
        float clamp = Config.WeightClamp;

        for (int k = 0; k < _hFanIn; k++)
        {
            float pre = _inputTrace[_hiddenIn[baseIdx + k]]; // 近似 STDP：前迹越大，LTP 越强
            float dw = lrLtp * pre - lrLtd * (1f - pre);
            float w = _hiddenW[baseIdx + k] + dw;
            if (w > clamp) w = clamp;
            if (w < 0f) w = 0f; // Dale 定律：兴奋性突触非负
            _hiddenW[baseIdx + k] = w;
        }
    }

    private void UpdateOutput(bool learn)
    {
        Array.Clear(_outputSpikes, 0, _outputSpikes.Length);
        _candidates.Clear();

        float decay = Config.OutputDecay;
        float thr = Config.OutputThreshold;

        for (int o = 0; o < Config.OutputSize; o++)
        {
            if (_refOutput[o] > 0)
            {
                _refOutput[o]--;
                _vOutput[o] = Config.ResetPotential;
                continue;
            }

            float v = _vOutput[o] * decay;
            int baseIdx = o * _oFanIn;
            for (int k = 0; k < _oFanIn; k++)
            {
                if (_hiddenSpikes[_outputIn[baseIdx + k]]) v += _outputW[baseIdx + k];
            }
            _vOutput[o] = v;
            if (v >= thr)
            {
                _candidates.Add((v, o));
            }
        }

        if (_candidates.Count > 0)
        {
            int keep = Math.Min(Config.OutputMaxSpikesPerStep, _candidates.Count);
            _candidates.Sort((a, b) => b.v.CompareTo(a.v));
            for (int i = 0; i < keep; i++)
            {
                int o = _candidates[i].idx;
                _outputSpikes[o] = true;
                _outputSpikeCount[o]++;
                _vOutput[o] = Config.ResetPotential;
                _refOutput[o] = Config.OutputRefractory;

                if (learn)
                {
                    UpdateEligibility(o);
                }
            }
        }

        // 资格迹衰减（整个平坦数组）
        if (learn)
        {
            float decayElig = Config.EligibilityDecay;
            for (int i = 0; i < _outputElig.Length; i++)
            {
                _outputElig[i] *= decayElig;
            }
        }
    }

    private void UpdateEligibility(int o)
    {
        int baseIdx = o * _oFanIn;
        for (int k = 0; k < _oFanIn; k++)
        {
            if (_hiddenSpikes[_outputIn[baseIdx + k]])
            {
                _outputElig[baseIdx + k] += 1f;
            }
        }
    }

    /// <summary>
    /// 奖励调制学习：结合多巴胺信号（全局奖励）和小脑教学信号。
    /// 正确时：多巴胺爆发 → LTP（强化获胜神经元的资格迹加权连接）
    /// 错误时：多巴胺低谷 → 轻度 LTD（弱化错误获胜者）
    ///          + 教学信号 → LTP（直接强化正确类别与活跃隐藏神经元的连接）
    /// 生物学基础：纹状体 D1/D2 受体通路 + 小脑攀爬纤维误差修正。
    /// </summary>
    private void ApplyReward(int prediction, int correctLabel, bool correct)
    {
        float lr = Config.OutputLearningRate;
        float clamp = Config.WeightClamp;

        if (correct)
        {
            // 多巴胺爆发：强化获胜神经元的资格迹加权连接（D1 受体 LTP）
            int baseIdx = prediction * _oFanIn;
            for (int k = 0; k < _oFanIn; k++)
            {
                float w = _outputW[baseIdx + k] + lr * _outputElig[baseIdx + k];
                _outputW[baseIdx + k] = Math.Clamp(w, 0f, clamp);
            }
        }
        else
        {
            // 多巴胺低谷：轻度弱化错误获胜者（D2 受体 LTD，强度为 LTP 的0.3×）
            float ltdLr = lr * 0.3f;
            int wrongBase = prediction * _oFanIn;
            for (int k = 0; k < _oFanIn; k++)
            {
                float w = _outputW[wrongBase + k] - ltdLr * _outputElig[wrongBase + k];
                _outputW[wrongBase + k] = Math.Clamp(w, 0f, clamp);
            }

            // 小脑教学信号：直接强化正确类别与活跃隐藏神经元的连接
            // 生物学：攀爬纤维提供精确的误差修正信号（Marr-Albus 理论）
            int correctBase = correctLabel * _oFanIn;
            int ts = Config.TimeSteps;
            for (int k = 0; k < _oFanIn; k++)
            {
                int hiddenIdx = _outputIn[correctBase + k];
                if (_hiddenSpikeCount[hiddenIdx] > 0)
                {
                    float activity = (float)_hiddenSpikeCount[hiddenIdx] / ts;
                    float w = _outputW[correctBase + k] + lr * activity;
                    _outputW[correctBase + k] = Math.Clamp(w, 0f, clamp);
                }
            }
        }
    }

    private void ScaleAll()
    {
        float target = Config.SynapticScaleTarget;
        if (target <= 0f) return;

        for (int h = 0; h < Config.HiddenSize; h++)
        {
            ScaleRow(_hiddenW, h * _hFanIn, _hFanIn, target);
        }
        for (int o = 0; o < Config.OutputSize; o++)
        {
            ScaleRow(_outputW, o * _oFanIn, _oFanIn, target);
        }
        _weightsDirty = true;
    }

    /// <summary>重置所有神经元运行时状态（样本间调用）。</summary>
    private void ResetState()
    {
        Array.Clear(_vHidden);
        Array.Clear(_vOutput);
        Array.Clear(_refHidden);
        Array.Clear(_refOutput);
        Array.Clear(_inputSpikes);
        Array.Clear(_hiddenSpikes);
        Array.Clear(_outputSpikes);
        Array.Clear(_inputTrace);
        Array.Clear(_hiddenTrace);
        Array.Clear(_hiddenSpikeCount);
        Array.Clear(_outputSpikeCount);
    }

    /// <summary>清除输出层资格迹（奖励施加后调用，避免跨样本污染）。</summary>
    private void ClearEligibility()
    {
        Array.Clear(_outputElig);
    }

    /// <summary>
    /// 自稳态内在可塑性：根据运行平均发放率调节隐藏层每个细胞的阈值。
    /// 过于活跃的细胞升高阈值（抑制），过于沉默的细胞降低阈值（兴奋）。
    /// 生物学基础：离子通道密度通过基因表达动态调节（Desai 等 1999）。
    /// 自由能原理：维持内部表征的多样性以最小化长期预测惊讶。
    /// </summary>
    private void UpdateHomeostasis()
    {
        float decay = Config.HomeostasisDecay;
        float oneMinusDecay = 1f - decay;
        float lr = Config.HomeostasisRate;
        float target = Config.TargetFiringRate;
        int ts = Config.TimeSteps;
        float thrMin = Config.ThresholdMin;
        float thrMax = Config.ThresholdMax;

        for (int h = 0; h < Config.HiddenSize; h++)
        {
            float rate = (float)_hiddenSpikeCount[h] / ts;
            _avgRateHidden[h] = decay * _avgRateHidden[h] + oneMinusDecay * rate;
            // 偏差驱动阈值调整：发放率高于目标 → 阈值升高，低于目标 → 阈值降低
            _thrHidden[h] += lr * (_avgRateHidden[h] - target);
            _thrHidden[h] = Math.Clamp(_thrHidden[h], thrMin, thrMax);
        }
    }

    private static void ScaleRow(float[] arr, int offset, int length, float target)
    {
        ReadOnlySpan<float> roSpan = arr.AsSpan(offset, length);
        float norm = SimdMath.L2Norm(roSpan);
        if (norm > target && norm > 1e-6f)
        {
            float scale = target / norm;
            Span<float> span = arr.AsSpan(offset, length);
            SimdMath.ScaleInPlace(span, scale);
        }
    }

    private static float[] InitWeights(int fanIn, Random rng)
    {
        float[] w = new float[fanIn];
        // 非负初始化（Dale 定律：兴奋性突触权重 ≥ 0）
        float s = 1f / MathF.Sqrt(fanIn);
        for (int i = 0; i < fanIn; i++)
        {
            w[i] = (float)rng.NextDouble() * s;
        }
        return w;
    }

    private static int[] SampleUnique(int universe, int count, Random rng)
    {
        // 自动限制 count 不超过 universe
        count = Math.Min(count, universe);
        if (count <= 0) return [];

        // 小规模用栈分配，大规模用堆分配
        if (universe <= 2048)
        {
            Span<int> pool = stackalloc int[universe];
            for (int i = 0; i < universe; i++) pool[i] = i;
            for (int i = 0; i < count; i++)
            {
                int j = rng.Next(i, universe);
                (pool[i], pool[j]) = (pool[j], pool[i]);
            }
            int[] result = new int[count];
            for (int i = 0; i < count; i++) result[i] = pool[i];
            return result;
        }
        else
        {
            int[] pool = new int[universe];
            for (int i = 0; i < universe; i++) pool[i] = i;
            for (int i = 0; i < count; i++)
            {
                int j = rng.Next(i, universe);
                (pool[i], pool[j]) = (pool[j], pool[i]);
            }
            int[] result = new int[count];
            Array.Copy(pool, result, count);
            return result;
        }
    }

    public void Dispose()
    {
        _gpu.Dispose();
    }
}
