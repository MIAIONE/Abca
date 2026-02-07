// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Network / AbcaNetwork
// Purpose: The complete ABCA neural network.
//
//   Architecture:  Input(784) → Hidden(800) → Output(10)
//
//   Forward pass (bio-inspired):
//     1. Input encoding: rate coding (firing rate ∝ intensity) or binary spikes
//     2. Hidden: weighted sum of encoded input + bias → ReLU threshold → TopK
//        (bio: integrate-and-fire + lateral inhibition)
//     3. Output: Σ(W × hidden_activity) + bias → argmax
//        (bio: EPSP proportional to synaptic weight × pre-synaptic rate)
//
//   Learning (genuinely bio-inspired):
//     Hidden: Competitive Hebbian (winners move toward input pattern)
//     Output: Reward-modulated Hebbian (three-factor rule)
//       - ΔW = reward × pre_activity × post_activity × lr
//       - Correct → strengthen winner's synapses to active hidden cells
//       - Wrong → weaken wrong winner, strengthen correct class
//     Homeostasis: Adaptive bias to maintain target firing rate
//
//   What is NOT here (traditional ML artifacts removed):
//     ✗ Softmax normalization in training
//     ✗ Cross-entropy loss in training
//     ✗ Error = target - predicted (disguised gradient)
//     ✗ Weight decay / L2 regularization
//     ✗ Any form of backpropagation
// ============================================================================

using System.Runtime.CompilerServices;
using Abca.Compute;
using Abca.Learning;
using Abca.Memory;

namespace Abca.Network;

/// <summary>
/// Controls which layers update their weights during training.
/// </summary>
public enum TrainingMode
{
    /// <summary>Both hidden (unsupervised) and output (supervised) layers learn.</summary>
    Full,
    /// <summary>Only hidden layer learns via competitive Hebbian (unsupervised pre-training).</summary>
    HiddenOnly,
    /// <summary>Only output layer learns via reward-modulated Hebbian (hidden frozen).</summary>
    OutputOnly,
}

/// <summary>
/// The ABCA bio-inspired network: spike-based encoding, competitive feature
/// extraction, and reward-modulated Hebbian classification.
/// Zero backpropagation. All learning is local.
/// </summary>
public sealed class AbcaNetwork : IDisposable
{
    private readonly NetworkConfig _config;
    private readonly CellLayer _hiddenLayer;
    private readonly CellLayer _outputLayer;

    // ── Scratch buffers (pre-allocated, reused per forward pass) ──────────
    private readonly NativeBuffer<float> _inputEncoded;     // rate-coded [0,1] or binary spikes
    private readonly NativeBuffer<float> _hiddenPotential;  // raw hidden potentials
    private readonly NativeBuffer<float> _hiddenAct;        // post-inhibition activities
    private readonly NativeBuffer<float> _outputPotential;  // raw output potentials

    // ── Homeostasis state ────────────────────────────────────────────────
    private readonly NativeBuffer<float> _avgActivity;      // running avg activity per hidden cell
    private int _step;                                       // training step counter

    // ── Optional GPU acceleration ────────────────────────────────────────
    private readonly GpuAccelerator? _gpu;

    private bool _disposed;

    /// <summary>Network configuration.</summary>
    public NetworkConfig Config => _config;

    /// <summary>Hidden cell layer (for inspection / serialization).</summary>
    public CellLayer HiddenLayer => _hiddenLayer;

    /// <summary>Output cell layer (for inspection / serialization).</summary>
    public CellLayer OutputLayer => _outputLayer;

    /// <summary>GPU accelerator (null if using CPU only).</summary>
    public GpuAccelerator? Gpu => _gpu;

    // ─────────────────────────────────────────────────────────────────────
    //  Construction
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a new ABCA network with the specified configuration.
    /// All weights are initialized randomly; the network is ready to train.
    /// </summary>
    /// <param name="config">Network configuration.</param>
    /// <param name="gpu">Optional GPU accelerator for hidden layer forward pass.</param>
    public AbcaNetwork(NetworkConfig config, GpuAccelerator? gpu = null)
    {
        _config = config;
        _gpu = gpu;
        var rng = new Random(config.Seed);

        _hiddenLayer = new CellLayer(config.InputSize, config.HiddenSize, rng);
        _outputLayer = new CellLayer(config.HiddenSize, config.OutputSize, rng);

        // Scratch buffers
        _inputEncoded = new NativeBuffer<float>(config.InputSize);
        _hiddenPotential = new NativeBuffer<float>(config.HiddenSize);
        _hiddenAct = new NativeBuffer<float>(config.HiddenSize);
        _outputPotential = new NativeBuffer<float>(config.OutputSize);

        // Homeostasis
        _avgActivity = new NativeBuffer<float>(config.HiddenSize);
        float target = config.TopKFraction;
        Span<float> avg = _avgActivity.AsSpan();
        for (int i = 0; i < avg.Length; i++) avg[i] = target;
    }

    /// <summary>
    /// Internal constructor for deserialization (layers created externally).
    /// </summary>
    internal AbcaNetwork(NetworkConfig config, CellLayer hidden, CellLayer output, GpuAccelerator? gpu = null)
    {
        _config = config;
        _gpu = gpu;
        _hiddenLayer = hidden;
        _outputLayer = output;

        _inputEncoded = new NativeBuffer<float>(config.InputSize);
        _hiddenPotential = new NativeBuffer<float>(config.HiddenSize);
        _hiddenAct = new NativeBuffer<float>(config.HiddenSize);
        _outputPotential = new NativeBuffer<float>(config.OutputSize);
        _avgActivity = new NativeBuffer<float>(config.HiddenSize);
        float target = config.TopKFraction;
        Span<float> avg = _avgActivity.AsSpan();
        for (int i = 0; i < avg.Length; i++) avg[i] = target;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Forward Pass (Inference)
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs a forward pass and returns the predicted class index.
    ///
    /// Bio-inspired pipeline:
    ///   1. Spike encoding: continuous input → binary {0,1}
    ///   2. Hidden integration: each cell sums weighted spikes + bias
    ///   3. ReLU threshold: cells below zero are silent
    ///   4. Lateral inhibition: TopK competition, losers silenced
    ///   5. Output integration: each output cell sums weights from active hidden cells
    ///   6. Winner: cell with highest potential = prediction
    /// </summary>
    /// <param name="input">Input vector (e.g. 784 pixel values in [0,1]).</param>
    /// <returns>Predicted class index.</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int Forward(ReadOnlySpan<float> input)
    {
        // ── 1. Input encoding ────────────────────────────────────────────
        Span<float> encoded = _inputEncoded.AsSpan();
        if (_config.UseRateCoding)
        {
            // Rate coding: firing rate = stimulus intensity [0,1]
            // Bio: retinal ganglion cells fire at rates proportional to light
            input.CopyTo(encoded);
        }
        else
        {
            // Binary spikes: threshold binarization
            SimdMath.SpikeEncode(input, encoded, _config.SpikeThreshold);
        }
        ReadOnlySpan<float> enc = _inputEncoded.AsReadOnlySpan();

        // ── 2. Hidden layer integration (SIMD or GPU) ──────────────────
        // Each hidden cell: potential = dot(W[j,:], encoded) + bias[j]
        // Bio: dendritic integration of incoming spike rates
        if (_gpu is not null)
        {
            _gpu.MatVecMulBias(_hiddenLayer.Weights, enc,
                _hiddenLayer.Bias.AsReadOnlySpan(), _hiddenPotential.AsSpan());
        }
        else
        {
            _hiddenLayer.Forward(enc, _hiddenPotential.AsSpan());
        }

        // ── 3. ReLU threshold ────────────────────────────────────────────
        // Bio: action potential threshold — cells below zero are silent
        Span<float> hAct = _hiddenAct.AsSpan();
        _hiddenPotential.AsReadOnlySpan().CopyTo(hAct);
        SimdMath.ReLUInPlace(hAct);

        // ── 4. Lateral inhibition (TopK) ─────────────────────────────────
        // Bio: inhibitory interneurons suppress weak responders
        SimdMath.TopK(hAct, _config.TopK);

        // ── 5. Output layer integration (graded SIMD dot product) ────────
        // output[k] = dot(W[k,:], hidden_activity) + bias[k]
        // Bio: EPSP proportional to synaptic weight × pre-synaptic firing rate
        _outputLayer.Forward(_hiddenAct.AsReadOnlySpan(), _outputPotential.AsSpan());

        // ── 6. Winner selection ──────────────────────────────────────────
        // Bio: the most activated output cell "wins" (decision)
        return SimdMath.ArgMax(_outputPotential.AsReadOnlySpan());
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Training Step
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Performs a single training step: forward pass + local weight updates.
    ///
    /// Learning is purely bio-inspired:
    ///   - Hidden: Competitive Hebbian (unsupervised feature extraction)
    ///   - Output: Reward-modulated Hebbian (three-factor rule)
    ///   - Homeostasis: Adaptive excitability regulation
    ///
    /// There is NO softmax, NO error vector, NO gradient, NO weight decay.
    /// </summary>
    /// <param name="input">Input vector.</param>
    /// <param name="label">Correct class label (0-based).</param>
    /// <param name="mode">Which layers to update.</param>
    /// <returns>True if the prediction was correct.</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public bool TrainStep(ReadOnlySpan<float> input, int label, TrainingMode mode = TrainingMode.Full)
    {
        // Forward pass (populates _inputEncoded, _hiddenAct, _outputPotential)
        int prediction = Forward(input);
        bool correct = prediction == label;
        _step++;

        // ── Output layer: Reward-modulated Hebbian (SIMD) ────────────────
        // Three-factor rule: ΔW ∝ reward × pre_activity × lr
        // The ONLY supervised signal is a scalar reward (correct/wrong).
        // NO softmax, NO error = target - predicted, NO gradient.
        if (mode != TrainingMode.HiddenOnly)
        {
            RewardModulatedHebbian.Update(
                _outputLayer.Weights,
                _outputLayer.Bias,
                _hiddenAct.AsReadOnlySpan(),
                prediction,
                label,
                _config.OutputLearningRate);

            // Synaptic scaling (bio-plausible weight homeostasis)
            // Applied every 256 steps to prevent unbounded weight growth.
            // Bio: limited synaptic receptor density.
            if (_config.SynapticScaleTarget > 0f && (_step & 0xFF) == 0)
            {
                SimdMath.SynapticScale(_outputLayer.Weights, _config.SynapticScaleTarget);
            }
        }

        // ── Hidden layer: Competitive Hebbian (SIMD) ─────────────────────
        // Winners move their weight vectors toward the encoded input pattern.
        // This is unsupervised — no label information used.
        if (mode != TrainingMode.OutputOnly)
        {
            CompetitiveLearning.UpdateWeights(
                _hiddenLayer.Weights,
                _inputEncoded.AsReadOnlySpan(),
                _hiddenAct.AsReadOnlySpan(),
                _config.HiddenLearningRate);

            Homeostasis.Update(
                _hiddenLayer.Bias,
                _avgActivity,
                _hiddenAct.AsReadOnlySpan(),
                _config.TopKFraction,
                _config.HomeostasisRate,
                _config.HomeostasisDecay);

            // Synaptic scaling for hidden layer (every 1024 steps)
            if (_config.SynapticScaleTarget > 0f && (_step & 0x3FF) == 0)
            {
                SimdMath.SynapticScale(_hiddenLayer.Weights, _config.SynapticScaleTarget);
            }

            // Inform GPU that hidden weights have changed
            _gpu?.InvalidateWeights();
        }

        return correct;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Batch Evaluation
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Evaluates the network on a dataset (inference only, no learning).
    /// </summary>
    /// <param name="images">Flat array of images (count × inputSize).</param>
    /// <param name="labels">Label array.</param>
    /// <param name="count">Number of samples.</param>
    /// <returns>Accuracy as a fraction in [0, 1].</returns>
    public float Evaluate(ReadOnlySpan<float> images, ReadOnlySpan<byte> labels, int count)
    {
        int correct = 0;
        int inputSize = _config.InputSize;

        for (int i = 0; i < count; i++)
        {
            ReadOnlySpan<float> img = images.Slice(i * inputSize, inputSize);
            int pred = Forward(img);
            if (pred == labels[i]) correct++;
        }

        return (float)correct / count;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Dispose
    // ─────────────────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed)
        {
            _hiddenLayer.Dispose();
            _outputLayer.Dispose();
            _inputEncoded.Dispose();
            _hiddenPotential.Dispose();
            _hiddenAct.Dispose();
            _outputPotential.Dispose();
            _avgActivity.Dispose();
            _disposed = true;
        }
    }
}
