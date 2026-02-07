// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Memory / NativeMatrix
// Purpose: 2D row-major float matrix on native heap.
//          Each row is padded to a multiple of 16 floats (64 bytes) for
//          AVX-512 alignment.  Zero-GC, SIMD-optimal.
// ============================================================================

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Abca.Memory;

/// <summary>
/// A 2D float matrix stored on the native heap.
/// Row-major layout with each row padded to 64-byte (16-float) alignment
/// for maximal SIMD throughput.
/// </summary>
public sealed unsafe class NativeMatrix : IDisposable
{
    private float* _ptr;
    private readonly int _rows;
    private readonly int _cols;
    private readonly int _stride; // actual floats per row (cols rounded up to 16)
    private bool _disposed;

    /// <summary>Number of rows.</summary>
    public int Rows => _rows;

    /// <summary>Number of logical columns.</summary>
    public int Cols => _cols;

    /// <summary>Number of physical floats per row (padded for alignment).</summary>
    public int Stride => _stride;

    /// <summary>Raw pointer to the first element.</summary>
    public float* Ptr => _ptr;

    /// <summary>Total bytes allocated.</summary>
    public long ByteCount => (long)_rows * _stride * sizeof(float);

    /// <summary>
    /// Allocates a [<paramref name="rows"/>, <paramref name="cols"/>] matrix.
    /// Memory is cache-line aligned and zero-filled.
    /// </summary>
    public NativeMatrix(int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        _rows = rows;
        _cols = cols;
        _stride = (cols + 15) & ~15; // round up to multiple of 16
        nuint byteCount = (nuint)((long)_rows * _stride * sizeof(float));
        _ptr = (float*)NativeMemory.AlignedAlloc(byteCount, 64);
        if (_ptr == null)
            throw new OutOfMemoryException($"Failed to allocate {byteCount} bytes for NativeMatrix [{rows}x{cols}].");
        NativeMemory.Clear(_ptr, byteCount);
    }

    /// <summary>Element indexer.</summary>
    public ref float this[int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _ptr[row * _stride + col];
    }

    /// <summary>Returns a span over the logical columns of the given row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRow(int row) => new(_ptr + row * _stride, _cols);

    /// <summary>Returns a span over the full stride (including padding) of the given row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowPadded(int row) => new(_ptr + row * _stride, _stride);

    /// <summary>Returns a <see cref="ReadOnlySpan{T}"/> over the logical columns of the given row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetRowReadOnly(int row) => new(_ptr + row * _stride, _cols);

    /// <summary>Gets a raw float pointer to the start of a given row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPtr(int row) => _ptr + row * _stride;

    /// <summary>Writes the entire matrix to a binary stream (logical columns only, no padding).</summary>
    public void WriteTo(BinaryWriter writer)
    {
        for (int r = 0; r < _rows; r++)
        {
            ReadOnlySpan<byte> rowBytes = new(_ptr + r * _stride, _cols * sizeof(float));
            writer.Write(rowBytes);
        }
    }

    /// <summary>Reads the entire matrix from a binary stream (logical columns only).</summary>
    public void ReadFrom(BinaryReader reader)
    {
        for (int r = 0; r < _rows; r++)
        {
            Span<byte> rowBytes = new(_ptr + r * _stride, _cols * sizeof(float));
            reader.Read(rowBytes);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_ptr != null)
                NativeMemory.AlignedFree(_ptr);
            _ptr = null;
            _disposed = true;
        }
    }
}
