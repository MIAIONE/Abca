// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: IO / MnistLoader
// Purpose: Downloads (if needed) and parses the MNIST handwritten digit dataset.
//          IDX file format parser with GZip decompression.
// ============================================================================

using System.IO.Compression;
using System.Net.Http;

namespace Abca.IO;

/// <summary>
/// Loads the MNIST handwritten digit dataset.
/// Auto-downloads from a public mirror if local files are missing.
/// Returns raw float arrays (pixel values normalized to [0,1]).
/// </summary>
public static class MnistLoader
{
    private const string MirrorBase = "https://ossci-datasets.s3.amazonaws.com/mnist/";

    private static readonly string[] TrainFiles =
    [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz"
    ];

    private static readonly string[] TestFiles =
    [
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ];

    /// <summary>
    /// Loads the MNIST training dataset (60,000 images).
    /// </summary>
    /// <param name="directory">Local directory for cached files.</param>
    /// <returns>Tuple of (flat float array [60000×784], label array [60000], count).</returns>
    public static async Task<(float[] Images, byte[] Labels, int Count)> LoadTrainingDataAsync(string directory)
    {
        await EnsureDownloadedAsync(directory, TrainFiles);
        string imgPath = Path.Combine(directory, TrainFiles[0]);
        string lblPath = Path.Combine(directory, TrainFiles[1]);
        float[] images = ParseImages(imgPath);
        byte[] labels = ParseLabels(lblPath);
        return (images, labels, labels.Length);
    }

    /// <summary>
    /// Loads the MNIST test dataset (10,000 images).
    /// </summary>
    public static async Task<(float[] Images, byte[] Labels, int Count)> LoadTestDataAsync(string directory)
    {
        await EnsureDownloadedAsync(directory, TestFiles);
        string imgPath = Path.Combine(directory, TestFiles[0]);
        string lblPath = Path.Combine(directory, TestFiles[1]);
        float[] images = ParseImages(imgPath);
        byte[] labels = ParseLabels(lblPath);
        return (images, labels, labels.Length);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Download
    // ─────────────────────────────────────────────────────────────────────

    private static async Task EnsureDownloadedAsync(string directory, string[] fileNames)
    {
        Directory.CreateDirectory(directory);

        using var http = new HttpClient();
        http.Timeout = TimeSpan.FromMinutes(5);

        foreach (string fileName in fileNames)
        {
            string localPath = Path.Combine(directory, fileName);
            if (File.Exists(localPath))
            {
                var fi = new FileInfo(localPath);
                if (fi.Length > 0) continue; // Already downloaded
            }

            string url = MirrorBase + fileName;
            Console.Write($"  Downloading {fileName} ...");
            try
            {
                byte[] data = await http.GetByteArrayAsync(url);
                await File.WriteAllBytesAsync(localPath, data);
                Console.WriteLine($" OK ({data.Length:N0} bytes)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($" FAILED: {ex.Message}");
                throw new InvalidOperationException(
                    $"Failed to download MNIST file: {url}. " +
                    $"Please download manually and place in: {directory}", ex);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  IDX Format Parsing
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Parses an IDX3 image file (gzip compressed).
    /// Returns a flat float array: images[sample * 784 + pixel] ∈ [0, 1].
    /// </summary>
    private static float[] ParseImages(string gzipPath)
    {
        using var fs = File.OpenRead(gzipPath);
        using var gz = new GZipStream(fs, CompressionMode.Decompress);
        using var br = new BinaryReader(gz);

        int magic = ReadBigEndianInt32(br);
        if (magic != 2051)
            throw new FormatException($"Invalid MNIST image magic: {magic} (expected 2051)");

        int count = ReadBigEndianInt32(br);
        int rows = ReadBigEndianInt32(br);
        int cols = ReadBigEndianInt32(br);
        int pixelsPerImage = rows * cols;

        float[] result = new float[count * pixelsPerImage];
        byte[] buffer = new byte[pixelsPerImage];
        const float invByte = 1f / 255f;

        for (int i = 0; i < count; i++)
        {
            int offset = 0;
            while (offset < pixelsPerImage)
            {
                int read = gz.Read(buffer, offset, pixelsPerImage - offset);
                if (read == 0) throw new EndOfStreamException("Unexpected end of MNIST image data.");
                offset += read;
            }

            int baseIdx = i * pixelsPerImage;
            for (int p = 0; p < pixelsPerImage; p++)
            {
                result[baseIdx + p] = buffer[p] * invByte;
            }
        }

        return result;
    }

    /// <summary>
    /// Parses an IDX1 label file (gzip compressed).
    /// Returns a byte array of labels in [0, 9].
    /// </summary>
    private static byte[] ParseLabels(string gzipPath)
    {
        using var fs = File.OpenRead(gzipPath);
        using var gz = new GZipStream(fs, CompressionMode.Decompress);
        using var br = new BinaryReader(gz);

        int magic = ReadBigEndianInt32(br);
        if (magic != 2049)
            throw new FormatException($"Invalid MNIST label magic: {magic} (expected 2049)");

        int count = ReadBigEndianInt32(br);
        byte[] labels = new byte[count];

        int offset = 0;
        while (offset < count)
        {
            int read = gz.Read(labels, offset, count - offset);
            if (read == 0) throw new EndOfStreamException("Unexpected end of MNIST label data.");
            offset += read;
        }

        return labels;
    }

    /// <summary>Reads a 32-bit big-endian integer from the stream.</summary>
    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}
