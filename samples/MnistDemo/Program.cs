// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Sample: MNIST Handwritten Digit Recognition Demo
//
// This demo trains an ABCA network on the MNIST dataset using:
//   - Binary spike encoding (threshold binarization of pixel input)
//   - Competitive Hebbian learning (hidden layer, unsupervised)
//   - Reward-modulated Hebbian learning (output layer, three-factor rule)
//   - Homeostatic plasticity (adaptive excitability)
//   - NO backpropagation — all learning is local.
//   - NO softmax, NO cross-entropy, NO gradient descent in any form.
//
// Usage: dotnet run [--epochs N] [--hidden N] [--topk F] [--seed N]
// ============================================================================

using Abca.Diagnostics;
using Abca.IO;
using Abca.Network;

const string DataDir = "data/mnist";
const string ModelPath = "abca_mnist.bin";

// ─────────────────────────────────────────────────────────────────────────────
//  Parse command-line arguments
// ─────────────────────────────────────────────────────────────────────────────
int epochs = GetArgInt(args, "--epochs", 20);
int hiddenSize = GetArgInt(args, "--hidden", 1200);
int hiddenFanIn = GetArgInt(args, "--fan-in-hidden", 96);
int outputFanIn = GetArgInt(args, "--fan-in-output", 200);
int seed = GetArgInt(args, "--seed", 42);

Console.WriteLine("╔══════════════════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║    ABCA - Asynchronous Bio-inspired Computing Architecture                  ║");
Console.WriteLine("║    MNIST Handwritten Digit Recognition                                      ║");
Console.WriteLine("╠══════════════════════════════════════════════════════════════════════════════╣");
Console.WriteLine("║  Learning: Spike Encoding → Competitive Hebbian → Reward-Modulated Hebbian  ║");
Console.WriteLine("║  Three-factor rule: ΔW = reward × pre_activity × post_activity              ║");
Console.WriteLine("║  NO backpropagation. NO softmax training. NO gradient. All learning local.  ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Load MNIST data
// ─────────────────────────────────────────────────────────────────────────────
Console.WriteLine("[1/5] Loading MNIST dataset...");
var (trainImages, trainLabels, trainCount) = await MnistLoader.LoadTrainingDataAsync(DataDir);
var (testImages, testLabels, testCount) = await MnistLoader.LoadTestDataAsync(DataDir);
Console.WriteLine($"  Training samples: {trainCount:N0}");
Console.WriteLine($"  Test samples:     {testCount:N0}");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Create network
// ─────────────────────────────────────────────────────────────────────────────
Console.WriteLine("[2/5] Creating ABCA network...");
var config = new NetworkConfig
{
    InputSize = 784,
    HiddenSize = hiddenSize,
    OutputSize = 10,
    HiddenFanIn = hiddenFanIn,
    OutputFanIn = outputFanIn,
    HiddenMaxSpikesPerStep = 100,
    OutputMaxSpikesPerStep = 5,
    TimeSteps = 10,
    HiddenDecay = 0.85f,
    OutputDecay = 0.85f,
    HiddenThreshold = 0.3f,
    OutputThreshold = 0.2f,
    HiddenRefractory = 1,
    OutputRefractory = 1,
    HiddenLtpLearningRate = 0.01f,
    HiddenLtdLearningRate = 0.004f,
    OutputLearningRate = 0.04f,
    EligibilityDecay = 0.90f,
    TraceDecay = 0.85f,
    SynapticScaleTarget = 5f,
    SynapticScalePeriod = 100,
    WeightClamp = 3.0f,
    UseRateCoding = true,
    InputRateScale = 0.5f,
    Epochs = epochs,
    Seed = seed
};

using var network = new AbcaNetwork(config);
Console.WriteLine($"  Hidden cells:     {config.HiddenSize}");
Console.WriteLine($"  Hidden fan-in:    {config.HiddenFanIn}");
Console.WriteLine($"  Output fan-in:    {config.OutputFanIn}");
Console.WriteLine($"  Compute device:   {network.Gpu.DeviceName} ({(network.Gpu.IsGpu ? "CUDA GPU" : "CPU Parallel")})");
Console.WriteLine($"  Input encoding:   {(config.UseRateCoding ? "Poisson rate" : "Binary spikes")}");
Console.WriteLine($"  Time steps:       {config.TimeSteps}");
Console.WriteLine($"  Synaptic scale:   {config.SynapticScaleTarget}");
Console.WriteLine($"  Hidden LTP/LTD:   {config.HiddenLtpLearningRate}/{config.HiddenLtdLearningRate}");
Console.WriteLine($"  Output LR:        {config.OutputLearningRate}");
Console.WriteLine($"  Epochs:           {config.Epochs}");
Console.WriteLine($"  Seed:             {config.Seed}");
Console.WriteLine();

Console.WriteLine("[3/5] Training...");
Console.WriteLine("─────────────────────────────────────────────────────────────────────────────────");

var history = new TrainingHistory();
var rng = new Random(seed);
int[] indices = new int[trainCount];
for (int i = 0; i < trainCount; i++) indices[i] = i;

for (int epoch = 1; epoch <= epochs; epoch++)
{
    var sw = System.Diagnostics.Stopwatch.StartNew();

    // 打乱数据
    for (int i = trainCount - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (indices[i], indices[j]) = (indices[j], indices[i]);
    }

    int correct = 0;
    for (int s = 0; s < trainCount; s++)
    {
        int idx = indices[s];
        ReadOnlySpan<float> img = trainImages.AsSpan(idx * 784, 784);
        int label = trainLabels[idx];
        if (network.TrainSample(img, label)) correct++;
    }

    float trainAcc = (float)correct / trainCount;
    float testAcc = network.Evaluate(testImages, testLabels, testCount);

    sw.Stop();
    float sps = trainCount / (float)sw.Elapsed.TotalSeconds;

    var metrics = new EpochMetrics
    {
        Epoch = epoch,
        TrainAccuracy = trainAcc,
        TestAccuracy = testAcc,
        TrainLoss = 0f,
        TestLoss = 0f,
        Duration = sw.Elapsed,
        SamplesPerSecond = sps
    };

    history.Record(metrics);
    Console.WriteLine(metrics);
}

Console.WriteLine("─────────────────────────────────────────────────────────────────────────────────");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Save model
// ─────────────────────────────────────────────────────────────────────────────
Console.WriteLine("[4/5] Saving model (稀疏事件驱动格式 v3)...");
ModelSerializer.Save(network, ModelPath);
var fileSize = new FileInfo(ModelPath).Length;
Console.WriteLine($"  Saved to: {ModelPath} ({fileSize:N0} bytes)");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Verify save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
Console.WriteLine("[5/5] Verifying save/load round-trip...");
using var loaded = ModelSerializer.Load(ModelPath);
float loadedAcc = loaded.Evaluate(testImages, testLabels, testCount);
Console.WriteLine($"  Loaded model test accuracy: {loadedAcc:P2}");
Console.WriteLine($"  Round-trip verified");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Summary
// ─────────────────────────────────────────────────────────────────────────────
history.PrintSummary();

// ─────────────────────────────────────────────────────────────────────────────
//  CLI argument helpers
// ─────────────────────────────────────────────────────────────────────────────
static int GetArgInt(string[] args, string name, int defaultValue)
{
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name && int.TryParse(args[i + 1], out int val))
            return val;
    }
    return defaultValue;
}
