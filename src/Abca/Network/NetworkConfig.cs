// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Network / NetworkConfig
// Purpose: Immutable configuration for the ABCA network.
//
// Bio-inspired design principles:
//   ✓ Rate coding / spike-based input encoding
//   ✓ Competitive lateral inhibition (top-K winners)
//   ✓ Hebbian learning (fire together → wire together)
//   ✓ Reward-modulated plasticity (global reward, local synapses)
//   ✓ Homeostatic excitability regulation
//
// Explicitly excluded (traditional ML artifacts):
//   ✗ Weight decay / L2 regularization
//   ✗ Input L2-normalization
//   ✗ Any gradient-derived learning signal
// ============================================================================

namespace Abca.Network;

/// <summary>
/// Configuration for an ABCA network instance.
/// All hyperparameters are set at construction time.
/// </summary>
public sealed record NetworkConfig
{
    /// <summary>输入维度（如 MNIST 28×28=784）。</summary>
    public int InputSize { get; init; } = 784;

    /// <summary>隐藏层细胞数（稀疏连接，每个细胞固定 fan-in）。</summary>
    public int HiddenSize { get; init; } = 1200;

    /// <summary>输出类别数。</summary>
    public int OutputSize { get; init; } = 10;

    // ── 稀疏连接参数 ────────────────────────────────────────────────

    /// <summary>隐藏层每个细胞的固定入度（从输入层）。</summary>
    public int HiddenFanIn { get; init; } = 96;

    /// <summary>输出层每个细胞的固定入度（从隐藏层）。</summary>
    public int OutputFanIn { get; init; } = 64;

    /// <summary>隐藏层每步允许发放的最大细胞数（侧抑制上限）。</summary>
    public int HiddenMaxSpikesPerStep { get; init; } = 32;

    /// <summary>输出层每步允许发放的最大细胞数（全局抑制上限）。</summary>
    public int OutputMaxSpikesPerStep { get; init; } = 4;

    /// <summary>权重范数上限（突触缩放目标，0 关闭）。</summary>
    public float SynapticScaleTarget { get; init; } = 3f;

    /// <summary>权重软裁剪上限（避免少数突触爆炸）。</summary>
    public float WeightClamp { get; init; } = 2.5f;

    // ── 时间与膜电位动力学 ──────────────────────────────────────

    /// <summary>每个样本的离散时间步数。</summary>
    public int TimeSteps { get; init; } = 20;

    /// <summary>隐藏层膜电位衰减系数（0~1，越小衰减越快）。</summary>
    public float HiddenDecay { get; init; } = 0.9f;

    /// <summary>输出层膜电位衰减系数。</summary>
    public float OutputDecay { get; init; } = 0.9f;

    /// <summary>隐藏层发放阈值。</summary>
    public float HiddenThreshold { get; init; } = 1.0f;

    /// <summary>输出层发放阈值。</summary>
    public float OutputThreshold { get; init; } = 1.0f;

    /// <summary>隐藏层折返期（时间步）。</summary>
    public int HiddenRefractory { get; init; } = 2;

    /// <summary>输出层折返期（时间步）。</summary>
    public int OutputRefractory { get; init; } = 2;

    /// <summary>膜电位重置值。</summary>
    public float ResetPotential { get; init; } = 0f;

    // ── 学习率与可塑性 ──────────────────────────────────────────

    /// <summary>隐藏层 STDP 学习率（LTP 部分）。</summary>
    public float HiddenLtpLearningRate { get; init; } = 0.012f;

    /// <summary>隐藏层 STDP 学习率（LTD 部分）。</summary>
    public float HiddenLtdLearningRate { get; init; } = 0.006f;

    /// <summary>输出层奖励调制 Hebb 学习率。</summary>
    public float OutputLearningRate { get; init; } = 0.02f;

    /// <summary>资格迹衰减系数（输出层）。</summary>
    public float EligibilityDecay { get; init; } = 0.95f;

    /// <summary>输入/隐藏脉冲迹衰减系数（STDP 窗口）。</summary>
    public float TraceDecay { get; init; } = 0.92f;

    /// <summary>突触缩放触发步频（样本粒度）。</summary>
    public int SynapticScalePeriod { get; init; } = 32;

    // ── 自稳态内在可塑性（自由能原理：最小化预测惊讶）─────────

    /// <summary>隐藏层自稳态学习率（调节发放阈值）。</summary>
    public float HomeostasisRate { get; init; } = 0.002f;

    /// <summary>隐藏层发放率 EMA 衰减系数。</summary>
    public float HomeostasisDecay { get; init; } = 0.99f;

    /// <summary>隐藏层目标发放率（每步发放概率）。</summary>
    public float TargetFiringRate { get; init; } = 0.03f;

    /// <summary>自适应阈值下限。</summary>
    public float ThresholdMin { get; init; } = 0.1f;

    /// <summary>自适应阈值上限。</summary>
    public float ThresholdMax { get; init; } = 5f;

    /// <summary>奖励正例值（正确）。</summary>
    public float RewardPositive { get; init; } = 1.0f;

    /// <summary>奖励负例值（错误）。</summary>
    public float RewardNegative { get; init; } = -1.0f;

    // ── 输入编码 ──────────────────────────────────────────────────

    /// <summary>使用速率编码（泊松脉冲）；false 时用二值阈值脉冲。</summary>
    public bool UseRateCoding { get; init; } = true;

    /// <summary>二值脉冲阈值（UseRateCoding=false 时生效）。</summary>
    public float SpikeThreshold { get; init; } = 0.1f;

    /// <summary>输入泊松脉冲比例系数（像素强度 × 此系数 = 期望发放概率）。</summary>
    public float InputRateScale { get; init; } = 0.25f;

    // ── 训练控制 ──────────────────────────────────────────────────

    /// <summary>训练 epoch 数。</summary>
    public int Epochs { get; init; } = 20;

    /// <summary>随机种子。</summary>
    public int Seed { get; init; } = 42;
}
