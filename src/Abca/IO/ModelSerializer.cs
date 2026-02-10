// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// 模块: IO / ModelSerializer
// 目的: 稀疏、事件驱动网络的二进制持久化。
// 格式 v3: [Header][Config][HiddenLayer][OutputLayer]
//   HiddenLayer: 逐细胞 [fanIn][indices...][weights...] + 自适应阈值
//   OutputLayer: 同上，并包含资格迹。
// 内部使用平坦行主序数组，磁盘格式兼容 v3。
// ============================================================================

using Abca.Network;

namespace Abca.IO;

public static class ModelSerializer
{
    private const uint MagicNumber = 0x41424341; // "ABCA"
    private const int FormatVersion = 3;

    public static void Save(AbcaNetwork network, string path)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        var cfg = network.Config;

        // Header
        bw.Write(MagicNumber);
        bw.Write(FormatVersion);

        // Config（完整写出，以便精确复现）
        bw.Write(cfg.InputSize);
        bw.Write(cfg.HiddenSize);
        bw.Write(cfg.OutputSize);
        bw.Write(cfg.HiddenFanIn);
        bw.Write(cfg.OutputFanIn);
        bw.Write(cfg.HiddenMaxSpikesPerStep);
        bw.Write(cfg.OutputMaxSpikesPerStep);
        bw.Write(cfg.SynapticScaleTarget);
        bw.Write(cfg.WeightClamp);
        bw.Write(cfg.TimeSteps);
        bw.Write(cfg.HiddenDecay);
        bw.Write(cfg.OutputDecay);
        bw.Write(cfg.HiddenThreshold);
        bw.Write(cfg.OutputThreshold);
        bw.Write(cfg.HiddenRefractory);
        bw.Write(cfg.OutputRefractory);
        bw.Write(cfg.ResetPotential);
        bw.Write(cfg.HiddenLtpLearningRate);
        bw.Write(cfg.HiddenLtdLearningRate);
        bw.Write(cfg.OutputLearningRate);
        bw.Write(cfg.EligibilityDecay);
        bw.Write(cfg.TraceDecay);
        bw.Write(cfg.SynapticScalePeriod);
        bw.Write(cfg.HomeostasisRate);
        bw.Write(cfg.HomeostasisDecay);
        bw.Write(cfg.TargetFiringRate);
        bw.Write(cfg.ThresholdMin);
        bw.Write(cfg.ThresholdMax);
        bw.Write(cfg.RewardPositive);
        bw.Write(cfg.RewardNegative);
        bw.Write(cfg.UseRateCoding);
        bw.Write(cfg.SpikeThreshold);
        bw.Write(cfg.InputRateScale);
        bw.Write(cfg.Epochs);
        bw.Write(cfg.Seed);

        // Hidden layer（从平坦数组按行写出，磁盘格式不变）
        int[] hIn = network.GetHiddenConnections();
        float[] hW = network.GetHiddenWeights();
        int hFanIn = network.GetHiddenFanIn();
        float[] hThr = network.GetHiddenThresholds();
        int hiddenSize = cfg.HiddenSize;

        bw.Write(hiddenSize);
        for (int i = 0; i < hiddenSize; i++)
        {
            bw.Write(hFanIn);
            int baseIdx = i * hFanIn;
            for (int k = 0; k < hFanIn; k++) bw.Write(hIn[baseIdx + k]);
            for (int k = 0; k < hFanIn; k++) bw.Write(hW[baseIdx + k]);
        }
        // 自适应阈值
        for (int i = 0; i < hiddenSize; i++) bw.Write(hThr[i]);

        // Output layer（从平坦数组按行写出）
        int[] oIn = network.GetOutputConnections();
        float[] oW = network.GetOutputWeights();
        float[] oE = network.GetOutputEligibility();
        int oFanIn = network.GetOutputFanIn();
        int outputSize = cfg.OutputSize;

        bw.Write(outputSize);
        for (int i = 0; i < outputSize; i++)
        {
            bw.Write(oFanIn);
            int baseIdx = i * oFanIn;
            for (int k = 0; k < oFanIn; k++) bw.Write(oIn[baseIdx + k]);
            for (int k = 0; k < oFanIn; k++) bw.Write(oW[baseIdx + k]);
            for (int k = 0; k < oFanIn; k++) bw.Write(oE[baseIdx + k]);
        }

        bw.Flush();
    }

    public static AbcaNetwork Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        uint magic = br.ReadUInt32();
        if (magic != MagicNumber)
            throw new FormatException($"无效的 ABCA 模型文件 (magic=0x{magic:X8})。");

        int version = br.ReadInt32();
        if (version != FormatVersion)
            throw new FormatException($"模型版本不匹配: {version} ≠ {FormatVersion}");

        var cfg = new NetworkConfig
        {
            InputSize = br.ReadInt32(),
            HiddenSize = br.ReadInt32(),
            OutputSize = br.ReadInt32(),
            HiddenFanIn = br.ReadInt32(),
            OutputFanIn = br.ReadInt32(),
            HiddenMaxSpikesPerStep = br.ReadInt32(),
            OutputMaxSpikesPerStep = br.ReadInt32(),
            SynapticScaleTarget = br.ReadSingle(),
            WeightClamp = br.ReadSingle(),
            TimeSteps = br.ReadInt32(),
            HiddenDecay = br.ReadSingle(),
            OutputDecay = br.ReadSingle(),
            HiddenThreshold = br.ReadSingle(),
            OutputThreshold = br.ReadSingle(),
            HiddenRefractory = br.ReadInt32(),
            OutputRefractory = br.ReadInt32(),
            ResetPotential = br.ReadSingle(),
            HiddenLtpLearningRate = br.ReadSingle(),
            HiddenLtdLearningRate = br.ReadSingle(),
            OutputLearningRate = br.ReadSingle(),
            EligibilityDecay = br.ReadSingle(),
            TraceDecay = br.ReadSingle(),
            SynapticScalePeriod = br.ReadInt32(),
            HomeostasisRate = br.ReadSingle(),
            HomeostasisDecay = br.ReadSingle(),
            TargetFiringRate = br.ReadSingle(),
            ThresholdMin = br.ReadSingle(),
            ThresholdMax = br.ReadSingle(),
            RewardPositive = br.ReadSingle(),
            RewardNegative = br.ReadSingle(),
            UseRateCoding = br.ReadBoolean(),
            SpikeThreshold = br.ReadSingle(),
            InputRateScale = br.ReadSingle(),
            Epochs = br.ReadInt32(),
            Seed = br.ReadInt32()
        };

        // Hidden layer（逐行读取，组装为平坦数组）
        int hiddenCount = br.ReadInt32();
        int hFanIn = 0;
        int[] hIn = Array.Empty<int>();
        float[] hW = Array.Empty<float>();

        for (int i = 0; i < hiddenCount; i++)
        {
            int fan = br.ReadInt32();
            if (i == 0)
            {
                hFanIn = fan;
                hIn = new int[hiddenCount * fan];
                hW = new float[hiddenCount * fan];
            }
            int baseIdx = i * hFanIn;
            for (int k = 0; k < fan; k++) hIn[baseIdx + k] = br.ReadInt32();
            for (int k = 0; k < fan; k++) hW[baseIdx + k] = br.ReadSingle();
        }

        // 自适应阈值
        float[] hThr = new float[hiddenCount];
        for (int i = 0; i < hiddenCount; i++) hThr[i] = br.ReadSingle();

        // Output layer（逐行读取，组装为平坦数组）
        int outputCount = br.ReadInt32();
        int oFanIn = 0;
        int[] oIn = Array.Empty<int>();
        float[] oW = Array.Empty<float>();
        float[] oE = Array.Empty<float>();

        for (int i = 0; i < outputCount; i++)
        {
            int fan = br.ReadInt32();
            if (i == 0)
            {
                oFanIn = fan;
                oIn = new int[outputCount * fan];
                oW = new float[outputCount * fan];
                oE = new float[outputCount * fan];
            }
            int baseIdx = i * oFanIn;
            for (int k = 0; k < fan; k++) oIn[baseIdx + k] = br.ReadInt32();
            for (int k = 0; k < fan; k++) oW[baseIdx + k] = br.ReadSingle();
            for (int k = 0; k < fan; k++) oE[baseIdx + k] = br.ReadSingle();
        }

        return new AbcaNetwork(cfg, hIn, hW, hFanIn, oIn, oW, oE, oFanIn, hThr);
    }
}
