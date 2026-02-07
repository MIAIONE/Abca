// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: IO / ModelSerializer
// Purpose: Binary model save/load with versioned header.
//          Supports full network state persistence and hot-reload.
// ============================================================================

using Abca.Network;

namespace Abca.IO;

/// <summary>
/// Binary model serializer for ABCA networks.
/// Format: [Header][Config][HiddenWeights][HiddenBias][OutputWeights][OutputBias]
/// </summary>
public static class ModelSerializer
{
    private const uint MagicNumber = 0x41424341; // "ABCA" in ASCII
    private const int FormatVersion = 2;

    /// <summary>
    /// Saves the network to a binary file.
    /// </summary>
    public static void Save(AbcaNetwork network, string path)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        var cfg = network.Config;

        // ── Header ──────────────────────────────────────────────────────
        bw.Write(MagicNumber);
        bw.Write(FormatVersion);

        // ── Config ──────────────────────────────────────────────────────
        bw.Write(cfg.InputSize);
        bw.Write(cfg.HiddenSize);
        bw.Write(cfg.OutputSize);
        bw.Write(cfg.TopKFraction);
        bw.Write(cfg.HiddenLearningRate);
        bw.Write(cfg.OutputLearningRate);
        bw.Write(cfg.HomeostasisRate);
        bw.Write(cfg.HomeostasisDecay);
        bw.Write(cfg.Epochs);
        bw.Write(cfg.Seed);
        bw.Write(cfg.SpikeThreshold);
        bw.Write(cfg.UseRateCoding);
        bw.Write(cfg.SynapticScaleTarget);

        // ── Hidden layer ────────────────────────────────────────────────
        network.HiddenLayer.Weights.WriteTo(bw);
        network.HiddenLayer.Bias.WriteTo(bw);

        // ── Output layer ────────────────────────────────────────────────
        network.OutputLayer.Weights.WriteTo(bw);
        network.OutputLayer.Bias.WriteTo(bw);

        bw.Flush();
    }

    /// <summary>
    /// Loads a network from a binary file.
    /// </summary>
    public static AbcaNetwork Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        // ── Header ──────────────────────────────────────────────────────
        uint magic = br.ReadUInt32();
        if (magic != MagicNumber)
            throw new FormatException($"Invalid ABCA model file (magic: 0x{magic:X8}).");

        int version = br.ReadInt32();
        if (version != FormatVersion)
            throw new FormatException($"Unsupported ABCA model version: {version} (expected {FormatVersion}).");

        // ── Config ──────────────────────────────────────────────────────
        var config = new NetworkConfig
        {
            InputSize = br.ReadInt32(),
            HiddenSize = br.ReadInt32(),
            OutputSize = br.ReadInt32(),
            TopKFraction = br.ReadSingle(),
            HiddenLearningRate = br.ReadSingle(),
            OutputLearningRate = br.ReadSingle(),
            HomeostasisRate = br.ReadSingle(),
            HomeostasisDecay = br.ReadSingle(),
            Epochs = br.ReadInt32(),
            Seed = br.ReadInt32(),
            SpikeThreshold = br.ReadSingle(),
            UseRateCoding = br.ReadBoolean(),
            SynapticScaleTarget = br.ReadSingle()
        };

        // ── Hidden layer ────────────────────────────────────────────────
        var hidden = new CellLayer(config.InputSize, config.HiddenSize);
        hidden.Weights.ReadFrom(br);
        hidden.Bias.ReadFrom(br);

        // ── Output layer ────────────────────────────────────────────────
        var output = new CellLayer(config.HiddenSize, config.OutputSize);
        output.Weights.ReadFrom(br);
        output.Bias.ReadFrom(br);

        return new AbcaNetwork(config, hidden, output);
    }
}
