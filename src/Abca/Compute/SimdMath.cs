// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Compute / SimdMath
// Purpose: SIMD-accelerated mathematical primitives for neural computation.
//          Uses hardware intrinsics (AVX2/AVX-512) where available,
//          with scalar fallbacks. Zero-allocation hot paths.
// ============================================================================

using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Abca.Memory;

namespace Abca.Compute;

/// <summary>
/// High-performance SIMD math operations for the ABCA network.
/// All methods are designed for zero-allocation, hot-path execution.
/// </summary>
public static class SimdMath
{
    // ─────────────────────────────────────────────────────────────────────
    //  Dot Product
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes the dot product of two float spans using hardware-accelerated
    /// SIMD via TensorPrimitives.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        return TensorPrimitives.Dot(a, b);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Matrix-Vector Multiplication
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Computes result[i] = dot(matrix.Row[i], vector) for each row.
    /// Uses TensorPrimitives.Dot per row for SIMD acceleration.
    /// </summary>
    public static void MatVecMul(NativeMatrix matrix, ReadOnlySpan<float> vector, Span<float> result)
    {
        int rows = matrix.Rows;
        int cols = matrix.Cols;
        for (int i = 0; i < rows; i++)
        {
            result[i] = TensorPrimitives.Dot(matrix.GetRowReadOnly(i), vector[..cols]);
        }
    }

    /// <summary>
    /// Computes result[i] = dot(matrix.Row[i], vector) + bias[i].
    /// Fused matrix-vector multiply with bias add.
    /// </summary>
    public static void MatVecMulBias(NativeMatrix matrix, ReadOnlySpan<float> vector,
                                     ReadOnlySpan<float> bias, Span<float> result)
    {
        int rows = matrix.Rows;
        int cols = matrix.Cols;
        for (int i = 0; i < rows; i++)
        {
            result[i] = TensorPrimitives.Dot(matrix.GetRowReadOnly(i), vector[..cols]) + bias[i];
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Activation Functions
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// In-place ReLU: x[i] = max(0, x[i]).
    /// Uses SIMD vectorized comparison and masking.
    /// </summary>
    public static void ReLUInPlace(Span<float> data)
    {
        int i = 0;
        int simdLen = Vector<float>.Count;
        int end = data.Length - simdLen + 1;
        ref float ptr = ref MemoryMarshal.GetReference(data);

        for (; i < end; i += simdLen)
        {
            var v = Unsafe.ReadUnaligned<Vector<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref ptr, i)));
            v = Vector.Max(v, Vector<float>.Zero);
            Unsafe.WriteUnaligned(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref ptr, i)), v);
        }

        for (; i < data.Length; i++)
        {
            if (data[i] < 0f) data[i] = 0f;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Spike Encoding (Bio-inspired input binarization)
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes a continuous input into binary spikes: dst[i] = src[i] > threshold ? 1 : 0.
    /// Bio-analog: retinal ganglion cells firing when stimulus exceeds threshold.
    /// </summary>
    public static void SpikeEncode(ReadOnlySpan<float> src, Span<float> dst, float threshold = 0f)
    {
        int n = src.Length;
        for (int i = 0; i < n; i++)
        {
            dst[i] = src[i] > threshold ? 1f : 0f;
        }
    }

    /// <summary>
    /// Graded sparse forward: output[k] = Σ_j W[k,j] × activity[j] + bias[k].
    /// Bio-honest: post-synaptic potential is proportional to BOTH
    /// synaptic weight AND pre-synaptic firing rate.
    /// Uses SIMD dot product (TensorPrimitives) — zeros in sparse activity
    /// naturally contribute zero, so this is correct for sparse vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SparseForward(
        NativeMatrix weights,
        ReadOnlySpan<float> activity,
        ReadOnlySpan<float> bias,
        Span<float> result)
    {
        int rows = weights.Rows;
        int cols = weights.Cols;

        for (int k = 0; k < rows; k++)
        {
            // SIMD dot product — zeros in sparse activity contribute zero
            result[k] = TensorPrimitives.Dot(weights.GetRowReadOnly(k)[..cols], activity[..cols]) + bias[k];
        }
    }

    /// <summary>
    /// Synaptic scaling: if a weight row's L2 norm exceeds targetNorm,
    /// scale it down to targetNorm. Bio-analog: limited synaptic receptor
    /// density constrains total synaptic strength per neuron.
    /// This is NOT L2 regularization (no loss penalty). It is a separate
    /// homeostatic mechanism that maintains weight magnitude bounds.
    /// </summary>
    public static void SynapticScale(NativeMatrix weights, float targetNorm)
    {
        if (targetNorm <= 0f) return;

        for (int r = 0; r < weights.Rows; r++)
        {
            Span<float> row = weights.GetRow(r);
            float norm = L2Norm(row);
            if (norm > targetNorm)
            {
                ScaleInPlace(row, targetNorm / norm);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Softmax (diagnostic tool — NOT used in training)
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Numerically stable softmax: p[i] = exp(x[i]-max) / sum(exp(x-max)).
    /// For small output layers (e.g. 10 classes), scalar code is optimal.
    /// </summary>
    public static void Softmax(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length;

        // Find max for numerical stability
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            if (src[i] > max) max = src[i];
        }

        // Compute exp(x - max) and sum
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            float e = MathF.Exp(src[i] - max);
            dst[i] = e;
            sum += e;
        }

        // Normalize
        float invSum = 1f / sum;
        for (int i = 0; i < n; i++)
        {
            dst[i] *= invSum;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  ArgMax
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the index of the maximum value in the span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ArgMax(ReadOnlySpan<float> data)
    {
        int bestIdx = 0;
        float bestVal = data[0];
        for (int i = 1; i < data.Length; i++)
        {
            if (data[i] > bestVal)
            {
                bestVal = data[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  TopK (Lateral Inhibition)
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Keeps only the top-K largest values in the span, zeroing out the rest.
    /// Implements bio-inspired lateral inhibition (winner-take-all).
    /// Uses Nth-element selection via partial sort.
    /// </summary>
    public static void TopK(Span<float> data, int k)
    {
        int n = data.Length;
        if (k >= n) return;
        if (k <= 0)
        {
            data.Clear();
            return;
        }

        // Rent a temporary array for sorting (avoids GC allocation)
        float[] rented = ArrayPool<float>.Shared.Rent(n);
        Span<float> temp = rented.AsSpan(0, n);
        try
        {
            data.CopyTo(temp);
            temp.Sort();

            // The threshold is the (n-k)th value in sorted order
            float threshold = temp[n - k];

            // Zero out values below threshold
            // Handle ties: count how many are at the threshold
            int countAbove = 0;
            for (int i = 0; i < n; i++)
            {
                if (data[i] > threshold) countAbove++;
            }
            int allowedAtThreshold = k - countAbove;

            for (int i = 0; i < n; i++)
            {
                if (data[i] > threshold)
                {
                    // Keep as is
                }
                else if (data[i] == threshold && allowedAtThreshold > 0)
                {
                    allowedAtThreshold--;
                    // Keep as is
                }
                else
                {
                    data[i] = 0f;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Vector Norms & Scaling
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>Computes the L2 (Euclidean) norm of a span.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float L2Norm(ReadOnlySpan<float> data)
    {
        return TensorPrimitives.Norm(data);
    }

    /// <summary>Scales every element in-place: data[i] *= factor.</summary>
    public static void ScaleInPlace(Span<float> data, float factor)
    {
        TensorPrimitives.Multiply(data, factor, data);
    }

    /// <summary>Normalizes a vector to unit L2 length in-place. No-op if norm ≈ 0.</summary>
    public static void NormalizeL2(Span<float> data)
    {
        float norm = L2Norm(data);
        if (norm > 1e-12f)
        {
            ScaleInPlace(data, 1f / norm);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Element-wise Operations
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>dst[i] += src[i] * scale (SIMD fused multiply-add).</summary>
    public static void AddScaled(Span<float> dst, ReadOnlySpan<float> src, float scale)
    {
        int n = dst.Length;
        int i = 0;
        int simdLen = Vector<float>.Count;
        int end = n - simdLen + 1;
        var vScale = new Vector<float>(scale);
        ref float dstRef = ref MemoryMarshal.GetReference(dst);
        ref float srcRef = ref MemoryMarshal.GetReference(src);

        for (; i < end; i += simdLen)
        {
            var s = Unsafe.ReadUnaligned<Vector<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref srcRef, i)));
            var d = Unsafe.ReadUnaligned<Vector<float>>(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref dstRef, i)));
            d += s * vScale;
            Unsafe.WriteUnaligned(
                ref Unsafe.As<float, byte>(ref Unsafe.Add(ref dstRef, i)), d);
        }

        for (; i < n; i++)
        {
            dst[i] += src[i] * scale;
        }
    }

    /// <summary>dst[i] = a[i] - b[i] (SIMD element-wise subtraction).</summary>
    public static void Subtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
    {
        TensorPrimitives.Subtract(a, b, dst);
    }

    /// <summary>Returns the sum of all elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(ReadOnlySpan<float> data)
    {
        return TensorPrimitives.Sum(data);
    }

    /// <summary>Computes cross-entropy loss = -sum(target * log(pred + eps)).</summary>
    public static float CrossEntropyLoss(ReadOnlySpan<float> target, ReadOnlySpan<float> predicted)
    {
        float loss = 0f;
        for (int i = 0; i < target.Length; i++)
        {
            if (target[i] > 0f)
            {
                loss -= target[i] * MathF.Log(MathF.Max(predicted[i], 1e-7f));
            }
        }
        return loss;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Random Initialization
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Fills a span with Xavier-uniform initialization: U(-s, s) where s = 1/sqrt(fanIn).
    /// </summary>
    public static void XavierUniform(Span<float> data, int fanIn, Random rng)
    {
        float scale = 1f / MathF.Sqrt(fanIn);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        }
    }

    /// <summary>
    /// Fills a matrix with small random initialization per row.
    /// Scale: U(-s, s) where s = 1/sqrt(fanIn) — ensures initial potentials
    /// are neither too large nor too small.
    /// </summary>
    public static void InitMatrix(NativeMatrix matrix, int fanIn, Random rng)
    {
        for (int r = 0; r < matrix.Rows; r++)
        {
            Span<float> row = matrix.GetRow(r);
            XavierUniform(row, fanIn, rng);
        }
    }
}
