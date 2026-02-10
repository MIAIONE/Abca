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
    ///   W[j] ← (1 - η·a) × W[j] + η·a × input
    /// where a = normalized activity of cell j (activity-dependent plasticity).
    /// Bio-rationale: Stronger post-synaptic activation causes larger Ca²⁺
    /// influx through NMDA receptors, enabling more LTP. This is the 
    /// biological basis of activity-dependent plasticity.
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

        // Find max activity for normalization (bio: relative activation matters)
        float maxAct = 0f;
        for (int j = 0; j < numCells; j++)
        {
            if (activity[j] > maxAct) maxAct = activity[j];
        }
        if (maxAct < 1e-12f) return; // No active cells
        float invMax = 1f / maxAct;

        for (int j = 0; j < numCells; j++)
        {
            if (activity[j] <= 0f) continue; // Only winners learn

            // Activity-dependent plasticity: lr scales with normalized activity
            // Bio: Ca²⁺ concentration ∝ firing rate → more LTP at higher rates
            float normalizedAct = activity[j] * invMax; // [0, 1]
            float effectiveLr = learningRate * normalizedAct;
            float oneMinusLr = 1f - effectiveLr;

            Span<float> wRow = weights.GetRow(j);

            // W = (1-lr·a) × W + lr·a × input  (SIMD vectorized)
            // Bio: receptive field moves toward correlated input pattern,
            // proportional to the cell's activation strength
            TensorPrimitives.Multiply((ReadOnlySpan<float>)wRow, oneMinusLr, wRow);
            SimdMath.AddScaled(wRow, input, effectiveLr);
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
    /// <summary>LTP/LTD asymmetry ratio. Bio: D1-receptor mediated LTP
    /// is ~1.5× stronger than D2-receptor mediated LTD in striatal
    /// medium spiny neurons (Shen et al. 2008).</summary>
    private const float LtpLtdRatio = 1.5f;

    /// <summary>
    /// Updates output weights using the enhanced three-factor rule with
    /// margin-based competitor suppression.
    /// </summary>
    /// <param name="weights">Output weight matrix [outputSize, hiddenSize].</param>
    /// <param name="bias">Output bias vector.</param>
    /// <param name="hiddenActivity">Post-inhibition hidden cell activities (sparse).</param>
    /// <param name="outputPotentials">Raw output potentials for all classes.</param>
    /// <param name="prediction">The winning output cell index.</param>
    /// <param name="correctLabel">The correct class label.</param>
    /// <param name="learningRate">Reward-modulated learning rate.</param>
    public static void Update(
        NativeMatrix weights,
        NativeBuffer<float> bias,
        ReadOnlySpan<float> hiddenActivity,
        ReadOnlySpan<float> outputPotentials,
        int prediction,
        int correctLabel,
        float learningRate)
    {
        Span<float> b = bias.AsSpan();
        int numClasses = weights.Rows;

        if (prediction == correctLabel)
        {
            // ── Positive reinforcement (LTP dominant) ──────────────────
            // ΔW[winner,j] = +η × LTP_ratio × hidden[j]
            // Bio: dopamine burst via D1 receptors → strong LTP
            float ltpLr = learningRate * LtpLtdRatio;
            Span<float> winnerRow = weights.GetRow(prediction);
            SimdMath.AddScaled(winnerRow, hiddenActivity, ltpLr);
            b[prediction] += ltpLr * 0.1f;

            // ── Margin reinforcement: suppress close competitors ───────
            // If any other class has potential close to the winner,
            // apply mild LTD to that competitor.
            // Bio: lateral inhibition sharpens category boundaries.
            float winnerPot = outputPotentials[prediction];
            float marginLr = learningRate * 0.3f;  // Mild suppression
            for (int k = 0; k < numClasses; k++)
            {
                if (k == correctLabel) continue;
                // Suppress competitors that are within 80% of winner
                if (outputPotentials[k] > winnerPot * 0.8f)
                {
                    Span<float> compRow = weights.GetRow(k);
                    SimdMath.AddScaled(compRow, hiddenActivity, -marginLr);
                    b[k] -= marginLr * 0.1f;
                }
            }
        }
        else
        {
            // ── Punishment (LTD) + correction (LTP) ────────────────────
            float ltpLr = learningRate * LtpLtdRatio;
            Span<float> wrongRow = weights.GetRow(prediction);
            Span<float> correctRow = weights.GetRow(correctLabel);

            SimdMath.AddScaled(wrongRow, hiddenActivity, -learningRate); // LTD (1×)
            SimdMath.AddScaled(correctRow, hiddenActivity, ltpLr);       // LTP (1.5×)

            b[prediction] -= learningRate * 0.1f;
            b[correctLabel] += ltpLr * 0.1f;

            // ── Suppress other high-activation competitors ─────────────
            // Bio: when wrong, not just the winner is problematic;
            // all non-correct neurons with high activation should be
            // suppressed to sharpen class discrimination.
            float marginLr = learningRate * 0.2f;
            float threshold = outputPotentials[prediction] * 0.5f;
            for (int k = 0; k < numClasses; k++)
            {
                if (k == prediction || k == correctLabel) continue;
                if (outputPotentials[k] > threshold)
                {
                    Span<float> compRow = weights.GetRow(k);
                    SimdMath.AddScaled(compRow, hiddenActivity, -marginLr);
                    b[k] -= marginLr * 0.1f;
                }
            }
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
