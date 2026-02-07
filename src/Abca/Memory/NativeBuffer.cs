// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Memory / NativeBuffer<T>
// Purpose: Cache-line aligned native memory buffer for unmanaged types.
//          Zero-GC: all memory lives on the native heap.
// ============================================================================

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Abca.Memory;

/// <summary>
/// A cache-line (64-byte) aligned native memory buffer for unmanaged value types.
/// Provides zero-GC, SIMD-friendly contiguous storage on the native heap.
/// </summary>
/// <typeparam name="T">An unmanaged value type (float, int, byte, etc.).</typeparam>
public sealed unsafe class NativeBuffer<T> : IDisposable where T : unmanaged
{
    private T* _ptr;
    private readonly int _length;
    private bool _disposed;

    /// <summary>The number of elements in this buffer.</summary>
    public int Length
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _length;
    }

    /// <summary>Raw pointer to the underlying memory. Use with caution.</summary>
    public T* Ptr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _ptr;
    }

    /// <summary>
    /// Allocates a new buffer of <paramref name="length"/> elements on the native heap,
    /// aligned to a 64-byte boundary (one cache line).
    /// </summary>
    /// <param name="length">Number of elements to allocate.</param>
    /// <param name="zeroFill">If true (default), zero-initialize all memory.</param>
    public NativeBuffer(int length, bool zeroFill = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        _length = length;
        nuint byteCount = (nuint)((long)length * sizeof(T));
        _ptr = (T*)NativeMemory.AlignedAlloc(byteCount, 64);
        if (_ptr == null)
            throw new OutOfMemoryException($"Failed to allocate {byteCount} bytes of aligned native memory.");
        if (zeroFill)
            NativeMemory.Clear(_ptr, byteCount);
    }

    /// <summary>Element indexer with aggressive inlining for hot-path access.</summary>
    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _ptr[index];
    }

    /// <summary>Returns a <see cref="Span{T}"/> over the entire buffer (zero-copy).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan() => new(_ptr, _length);

    /// <summary>Returns a <see cref="Span{T}"/> over a sub-range.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan(int start, int length) => new(_ptr + start, length);

    /// <summary>Returns a <see cref="ReadOnlySpan{T}"/> over the entire buffer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan() => new(_ptr, _length);

    /// <summary>Zero-fills the entire buffer.</summary>
    public void Clear()
    {
        NativeMemory.Clear(_ptr, (nuint)((long)_length * sizeof(T)));
    }

    /// <summary>Copies the contents of a span into this buffer.</summary>
    public void CopyFrom(ReadOnlySpan<T> source)
    {
        if (source.Length > _length)
            throw new ArgumentException("Source span is larger than buffer capacity.");
        source.CopyTo(AsSpan());
    }

    /// <summary>Writes all data to a binary stream.</summary>
    public void WriteTo(BinaryWriter writer)
    {
        ReadOnlySpan<byte> bytes = new(_ptr, _length * sizeof(T));
        writer.Write(bytes);
    }

    /// <summary>Reads data from a binary stream into this buffer.</summary>
    public void ReadFrom(BinaryReader reader)
    {
        Span<byte> bytes = new(_ptr, _length * sizeof(T));
        reader.Read(bytes);
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
