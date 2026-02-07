// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Network / CellLayer
// Purpose: A single layer of cells with weights and biases.
//          Owns its native memory and performs forward computation.
// ============================================================================

using System.Runtime.CompilerServices;
using Abca.Compute;
using Abca.Memory;

namespace Abca.Network;

/// <summary>
/// A layer of cells in the ABCA network.
/// Each cell has a weight vector (receptive field) and a bias.
/// Forward pass: output[j] = dot(W[j,:], input) + bias[j].
/// </summary>
public sealed class CellLayer : IDisposable
{
    private readonly NativeMatrix _weights;
    private readonly NativeBuffer<float> _bias;
    private bool _disposed;

    /// <summary>Weight matrix [outputSize, inputSize].</summary>
    public NativeMatrix Weights => _weights;

    /// <summary>Bias vector [outputSize].</summary>
    public NativeBuffer<float> Bias => _bias;

    /// <summary>Number of input dimensions.</summary>
    public int InputSize => _weights.Cols;

    /// <summary>Number of output cells.</summary>
    public int OutputSize => _weights.Rows;

    /// <summary>
    /// Creates a new cell layer with the given dimensions.
    /// Weights are initialized with small random values; biases are zero.
    /// </summary>
    /// <param name="inputSize">Input dimension (fan-in).</param>
    /// <param name="outputSize">Number of cells (fan-out).</param>
    /// <param name="rng">Random number generator for weight init.</param>
    public CellLayer(int inputSize, int outputSize, Random rng)
    {
        _weights = new NativeMatrix(outputSize, inputSize);
        _bias = new NativeBuffer<float>(outputSize, zeroFill: true);
        SimdMath.InitMatrix(_weights, inputSize, rng);
    }

    /// <summary>
    /// Creates a cell layer with pre-allocated (empty) storage.
    /// Used by model deserialization.
    /// </summary>
    internal CellLayer(int inputSize, int outputSize)
    {
        _weights = new NativeMatrix(outputSize, inputSize);
        _bias = new NativeBuffer<float>(outputSize, zeroFill: true);
    }

    /// <summary>
    /// Forward pass: result[j] = dot(W[j], input) + bias[j].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Forward(ReadOnlySpan<float> input, Span<float> result)
    {
        SimdMath.MatVecMulBias(_weights, input, _bias.AsReadOnlySpan(), result);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _weights.Dispose();
            _bias.Dispose();
            _disposed = true;
        }
    }
}
