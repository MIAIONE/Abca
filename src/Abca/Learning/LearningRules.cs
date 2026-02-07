// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Learning / LearningRules
// Purpose: Bio-inspired local learning rules — NO backpropagation.
//
//   1. CompetitiveLearning — unsupervised Hebbian with winner-take-all.
//      Extracts input prototypes (like biological receptive fields).
//
//   2. RewardModulatedHebbian — genuine three-factor learning rule.
//      ΔW = reward × pre_activity × post_activity × lr
//      Bio-analog: dopaminergic modulation of Hebbian plasticity.
//      NO softmax, NO error vector, NO gradient computation.
//
//   3. Homeostasis — adaptive threshold to maintain target firing rate.
//      Prevents dead/over-active cells (intrinsic plasticity).
//
// Bio-inspired principles retained:
//   ✓ Local computation only (no cross-layer gradient)
//   ✓ Hebbian plasticity ("fire together, wire together")
//   ✓ Competitive inhibition (winner-take-all)
//   ✓ Reward modulation (global dopamine-like scalar signal)
//   ✓ Homeostatic adaptation (intrinsic plasticity)
//
// Explicitly excluded (traditional ML artifacts):
//   ✗ Softmax probability → error vector (disguised gradient)
//   ✗ Delta rule ΔW = η × (target - predicted) × input
//   ✗ Weight decay / L2 regularization
//   ✗ Any form of error back-propagation
// ============================================================================

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using Abca.Compute;
using Abca.Memory;

namespace Abca.Learning;

/// <summary>
/// Competitive Hebbian learning for unsupervised feature extraction.
/// Each "winner" cell moves its weight vector toward the input pattern.
/// Bio-analog: receptive field development through correlated activity.
/// Equivalent to online k-means / self-organizing map learning.
/// </summary>
public static class CompetitiveLearning
{
    /// <summary>
    /// Updates weight rows for cells that were active (activity > 0).
    /// For each active cell j:
    ///   W[j] ← (1 - η) × W[j] + η × input
    /// This moves the weight vector toward the input pattern.
    /// </summary>
    /// <param name="weights">Weight matrix [numCells, inputSize].</param>
    /// <param name="input">The input vector (binary spikes or continuous).</param>
    /// <param name="activity">Post-inhibition cell activities (sparse; most are zero).</param>
    /// <param name="learningRate">Learning rate η.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void UpdateWeights(
        NativeMatrix weights,
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> activity,
        float learningRate)
    {
        int numCells = weights.Rows;
        float oneMinusLr = 1f - learningRate;

        for (int j = 0; j < numCells; j++)
        {
            if (activity[j] <= 0f) continue; // Only winners learn

            Span<float> wRow = weights.GetRow(j);

            // W = (1-lr) × W + lr × input  (SIMD vectorized)
            // Bio: receptive field moves toward correlated input pattern
            TensorPrimitives.Multiply((ReadOnlySpan<float>)wRow, oneMinusLr, wRow);
            SimdMath.AddScaled(wRow, input, learningRate);
        }
    }
}

/// <summary>
/// Reward-modulated Hebbian learning — genuine three-factor rule.
/// Bio-analog: dopaminergic modulation of Hebbian synaptic plasticity.
///
/// The three factors:
///   1. Pre-synaptic activity  (did the hidden cell fire?)
///   2. Post-synaptic activity (did the output cell fire/win?)
///   3. Global reward signal   (correct → +1, wrong → −1)
///
/// Learning rule:
///   When correct (reward = +1):
///     Strengthen winner's synapses from active pre-synaptic cells.
///   When wrong (reward = −1):
///     Weaken wrong winner's synapses from active pre-synaptic cells.
///     Strengthen correct class's synapses from active pre-synaptic cells.
///
/// This is NOT gradient descent. There is:
///   - No softmax normalization
///   - No error vector (target - predicted)
///   - No loss function derivative
///   - Only a scalar reward signal modulating local Hebbian traces
/// </summary>
public static class RewardModulatedHebbian
{
    /// <summary>
    /// Updates output weights using the three-factor rule.
    /// </summary>
    /// <param name="weights">Output weight matrix [outputSize, hiddenSize].</param>
    /// <param name="bias">Output bias vector.</param>
    /// <param name="hiddenActivity">Post-inhibition hidden cell activities (sparse).</param>
    /// <param name="prediction">The winning output cell index.</param>
    /// <param name="correctLabel">The correct class label.</param>
    /// <param name="learningRate">Reward-modulated learning rate.</param>
    public static void Update(
        NativeMatrix weights,
        NativeBuffer<float> bias,
        ReadOnlySpan<float> hiddenActivity,
        int prediction,
        int correctLabel,
        float learningRate)
    {
        Span<float> b = bias.AsSpan();
        int hiddenSize = weights.Cols;

        if (prediction == correctLabel)
        {
            // ── Positive reinforcement ─────────────────────────────────
            // ΔW[winner,j] = +η × hidden[j]  (SIMD vectorized)
            // Bio: dopamine burst → Hebbian LTP proportional to
            // pre-synaptic firing rate. Stronger input → larger ΔW.
            Span<float> winnerRow = weights.GetRow(prediction);
            SimdMath.AddScaled(winnerRow, hiddenActivity, learningRate);
            b[prediction] += learningRate * 0.1f;
        }
        else
        {
            // ── Punishment + correction ────────────────────────────────
            // ΔW[wrong,j]   = −η × hidden[j]  (SIMD vectorized)
            // ΔW[correct,j] = +η × hidden[j]  (SIMD vectorized)
            // Bio: dopamine dip → LTD at wrong winner proportional to
            // pre-synaptic rate; teaching signal → LTP at correct cell.
            Span<float> wrongRow = weights.GetRow(prediction);
            Span<float> correctRow = weights.GetRow(correctLabel);

            SimdMath.AddScaled(wrongRow, hiddenActivity, -learningRate);
            SimdMath.AddScaled(correctRow, hiddenActivity, learningRate);

            b[prediction] -= learningRate * 0.1f;
            b[correctLabel] += learningRate * 0.1f;
        }
    }
}

/// <summary>
/// Homeostatic plasticity — adjusts hidden cell biases so that
/// each cell maintains a target firing rate over time.
/// Prevents "dead cells" (never fire) and "dominant cells" (always fire).
/// Bio-analog: intrinsic excitability regulation via ion channel expression.
/// </summary>
public static class Homeostasis
{
    /// <summary>
    /// Updates the running average activity and adjusts bias accordingly.
    ///   avg[j] = decay × avg[j] + (1 - decay) × wasActive[j]
    ///   Δbias[j] = η_h × (targetRate - avg[j])
    /// </summary>
    /// <param name="bias">Hidden layer bias (adjusted in-place).</param>
    /// <param name="avgActivity">Running average activity per cell (updated in-place).</param>
    /// <param name="currentActivity">Current post-inhibition activities.</param>
    /// <param name="targetRate">Desired fraction of time each cell should be active.</param>
    /// <param name="learningRate">Homeostasis learning rate η_h.</param>
    /// <param name="decay">Exponential moving average decay factor (e.g. 0.999).</param>
    public static void Update(
        NativeBuffer<float> bias,
        NativeBuffer<float> avgActivity,
        ReadOnlySpan<float> currentActivity,
        float targetRate,
        float learningRate,
        float decay)
    {
        Span<float> b = bias.AsSpan();
        Span<float> avg = avgActivity.AsSpan();
        float oneMinusDecay = 1f - decay;

        for (int j = 0; j < b.Length; j++)
        {
            // Update running average: was this cell active?
            float wasActive = currentActivity[j] > 0f ? 1f : 0f;
            avg[j] = decay * avg[j] + oneMinusDecay * wasActive;

            // Nudge bias toward target rate
            b[j] += learningRate * (targetRate - avg[j]);
        }
    }
}
