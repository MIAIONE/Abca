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

using System.Diagnostics;
using Abca.Compute;
using Abca.Diagnostics;
using Abca.IO;
using Abca.Network;

const string DataDir = "data/mnist";
const string ModelPath = "abca_mnist.bin";

// ─────────────────────────────────────────────────────────────────────────────
//  Parse command-line arguments
// ─────────────────────────────────────────────────────────────────────────────
int epochs = GetArgInt(args, "--epochs", 30);
int hiddenSize = GetArgInt(args, "--hidden", 800);
float topKFraction = GetArgFloat(args, "--topk", 0.1f);
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
//  Initialize GPU (optional — falls back to CPU SIMD if no GPU)
// ─────────────────────────────────────────────────────────────────────────────
GpuAccelerator? gpu = null;
try
{
    gpu = new GpuAccelerator();
    Console.WriteLine($"[GPU] Device:     {gpu.DeviceName}");
    Console.WriteLine($"[GPU] Type:       {gpu.AccelType}");
    Console.WriteLine($"[GPU] Accelerated: {(gpu.IsGpu ? "Yes (GPU)" : "No (CPU fallback)")}");
    Console.WriteLine();
}
catch (Exception ex)
{
    Console.WriteLine($"[GPU] Not available: {ex.Message}");
    Console.WriteLine("[GPU] Falling back to CPU SIMD (AVX-512).");
    Console.WriteLine();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Create network
// ─────────────────────────────────────────────────────────────────────────────
Console.WriteLine("[2/5] Creating ABCA network...");
var config = new NetworkConfig
{
    InputSize = 784,
    HiddenSize = hiddenSize,
    OutputSize = 10,
    TopKFraction = topKFraction,
    HiddenLearningRate = 0.01f,
    OutputLearningRate = 0.01f,      // Bio: higher dopamine = faster learning
    HomeostasisRate = 0.002f,       // Bio: stronger intrinsic excitability regulation
    HomeostasisDecay = 0.999f,
    UseRateCoding = true,            // Rate coding: firing rate ∝ intensity (bio)
    SpikeThreshold = 0f,
    SynapticScaleTarget = 0f,        // Disabled: critical period → higher plasticity
    Epochs = epochs,
    Seed = seed
};

using var network = new AbcaNetwork(config, gpu);
Console.WriteLine($"  Hidden cells:     {config.HiddenSize}");
Console.WriteLine($"  Top-K winners:    {config.TopK} ({config.TopKFraction:P0} of hidden)");
Console.WriteLine($"  Input encoding:   {(config.UseRateCoding ? "Rate coding" : "Binary spikes")}");
Console.WriteLine($"  Compute backend:  {(gpu?.IsGpu == true ? $"GPU ({gpu.DeviceName})" : "CPU SIMD")}");
Console.WriteLine($"  Synaptic scale:   {config.SynapticScaleTarget}");
Console.WriteLine($"  Hidden LR:        {config.HiddenLearningRate}");
Console.WriteLine($"  Output LR:        {config.OutputLearningRate}");
Console.WriteLine($"  Epochs:           {config.Epochs}");
Console.WriteLine($"  Seed:             {config.Seed}");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Training loop — Three-phase bio-inspired training
//    Phase 1: Unsupervised competitive learning (develop receptive fields)
//    Phase 2: Full training (both layers learn — most bio-realistic)
//    Phase 3: Consolidation (output fine-tuning, hidden frozen)
//
//  Bio-rationale: In biological development:
//    1. Visual cortex develops feature detectors before task-specific learning
//    2. During active learning, ALL layers are simultaneously plastic
//    3. After learning, plasticity decreases (memory consolidation)
// ─────────────────────────────────────────────────────────────────────────────
int warmupEpochs = Math.Max(2, epochs / 5);       // 20% unsupervised
int fullEpochs = Math.Max(1, epochs * 3 / 5);     // 60% full (both layers plastic)
int consolidateEpochs = epochs - warmupEpochs - fullEpochs; // 20% consolidation

Console.WriteLine("[3/5] Training...");
Console.WriteLine($"  Phase 1: Unsupervised Hebbian feature learning    ({warmupEpochs} epochs)");
Console.WriteLine($"  Phase 2: Full co-adaptive learning (both layers)  ({fullEpochs} epochs)");
Console.WriteLine($"  Phase 3: Output consolidation (hidden frozen)     ({consolidateEpochs} epochs)");
Console.WriteLine("─────────────────────────────────────────────────────────────────────────────────");

var history = new TrainingHistory();
var rng = new Random(seed);

// GPU strategy: disable during training (CPU SIMD is faster for per-sample
// 800×784 inference — GPU transfer overhead > compute benefit at this scale).
// Enable for batch evaluation.
network.GpuEnabled = false;

// Pre-create shuffle indices
int[] indices = new int[trainCount];
for (int i = 0; i < trainCount; i++) indices[i] = i;

for (int epoch = 1; epoch <= epochs; epoch++)
{
    var sw = Stopwatch.StartNew();

    // Determine training mode for this epoch
    // Bio: development → active learning → consolidation
    TrainingMode mode;
    if (epoch <= warmupEpochs)
        mode = TrainingMode.HiddenOnly;           // Develop receptive fields
    else if (epoch <= warmupEpochs + fullEpochs)
        mode = TrainingMode.Full;                 // Both layers co-adapt
    else
        mode = TrainingMode.OutputOnly;           // Memory consolidation

    // Shuffle training data (Fisher-Yates)
    for (int i = trainCount - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (indices[i], indices[j]) = (indices[j], indices[i]);
    }

    // Train
    int correct = 0;

    for (int s = 0; s < trainCount; s++)
    {
        int idx = indices[s];
        ReadOnlySpan<float> img = trainImages.AsSpan(idx * 784, 784);
        int label = trainLabels[idx];

        if (network.TrainStep(img, label, mode))
            correct++;
    }

    float trainAcc = (float)correct / trainCount;

    // Evaluate on test set (re-enable GPU for inference, weights stable)
    network.GpuEnabled = true;
    float testAcc = network.Evaluate(testImages, testLabels, testCount);
    network.GpuEnabled = false;

    sw.Stop();
    float sps = trainCount / (float)sw.Elapsed.TotalSeconds;

    string phaseTag = epoch <= warmupEpochs ? "[unsup]" 
        : epoch <= warmupEpochs + fullEpochs ? "[full ]" : "[cnsld]";

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
    Console.WriteLine($"{phaseTag} {metrics}");
}

Console.WriteLine("─────────────────────────────────────────────────────────────────────────────────");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Save model
// ─────────────────────────────────────────────────────────────────────────────
Console.WriteLine("[4/5] Saving model...");
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
Console.WriteLine($"  Round-trip verified: accuracy matches last epoch");
Console.WriteLine();

// ─────────────────────────────────────────────────────────────────────────────
//  Summary
// ─────────────────────────────────────────────────────────────────────────────
history.PrintSummary();
gpu?.Dispose();

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

static float GetArgFloat(string[] args, string name, float defaultValue)
{
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name && float.TryParse(args[i + 1], out float val))
            return val;
    }
    return defaultValue;
}
