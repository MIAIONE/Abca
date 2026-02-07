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
    /// <summary>Input dimension (e.g. 784 for 28×28 MNIST images).</summary>
    public int InputSize { get; init; } = 784;

    /// <summary>Number of hidden cells (bio-analog: cortical column population).</summary>
    public int HiddenSize { get; init; } = 800;

    /// <summary>Number of output classes.</summary>
    public int OutputSize { get; init; } = 10;

    /// <summary>Fraction of hidden cells that survive lateral inhibition (top-K).</summary>
    public float TopKFraction { get; init; } = 0.1f;

    /// <summary>Number of top-K winners (computed from HiddenSize × TopKFraction).</summary>
    public int TopK => Math.Max(1, (int)(HiddenSize * TopKFraction));

    // ── Learning Rates ──────────────────────────────────────────────────

    /// <summary>Hidden layer competitive Hebbian learning rate.</summary>
    public float HiddenLearningRate { get; init; } = 0.01f;

    /// <summary>Output layer reward-modulated Hebbian learning rate.</summary>
    public float OutputLearningRate { get; init; } = 0.01f;

    /// <summary>Homeostatic plasticity learning rate.</summary>
    public float HomeostasisRate { get; init; } = 0.002f;

    /// <summary>Homeostatic running-average decay (EMA factor).</summary>
    public float HomeostasisDecay { get; init; } = 0.999f;

    // ── Bio-inspired Parameters ─────────────────────────────────────────

    /// <summary>
    /// Use rate coding: firing rate is proportional to stimulus intensity [0,1].
    /// This is the dominant neural coding scheme in biological visual cortex.
    /// Rate coding preserves intensity information (grays ≠ blacks).
    /// When false, uses binary spike encoding (threshold binarization).
    /// </summary>
    public bool UseRateCoding { get; init; } = true;

    /// <summary>
    /// Input spike threshold (only used when UseRateCoding = false):
    /// pixels > this value generate a spike (1.0), otherwise silent (0.0).
    /// Bio-analog: retinal ganglion cell threshold.
    /// </summary>
    public float SpikeThreshold { get; init; } = 0f;

    /// <summary>
    /// Target L2 norm for synaptic scaling (bio-plausible weight homeostasis).
    /// After weight updates, each neuron's weight vector is scaled to this norm
    /// if it exceeds it. Bio-analog: limited synaptic receptor density.
    /// Set to 0 to disable.
    /// </summary>
    public float SynapticScaleTarget { get; init; } = 3f;

    // ── Training ─────────────────────────────────────────────────────────

    /// <summary>Number of training epochs.</summary>
    public int Epochs { get; init; } = 30;

    /// <summary>Random seed for reproducibility.</summary>
    public int Seed { get; init; } = 42;
}
