// ============================================================================
// ABCA - Unit Tests
// Covers: Memory (NativeBuffer, NativeMatrix), Compute (SimdMath),
//         Learning rules, Network forward/backward, IO round-trip.
// ============================================================================

using Abca.Compute;
using Abca.Learning;
using Abca.Memory;
using Abca.Network;
using Abca.IO;

namespace Abca.Tests;

// ═══════════════════════════════════════════════════════════════════════════
//  GPU Accelerator Tests
// ═══════════════════════════════════════════════════════════════════════════

public class GpuAcceleratorTests
{
    [Fact]
    public void GpuAccelerator_CanInitialize()
    {
        using var gpu = new GpuAccelerator();
        Assert.NotNull(gpu.DeviceName);
        // Should be either GPU or CPU fallback
        Assert.True(gpu.IsGpu || !gpu.IsGpu); // Always true — just verify no crash
    }

    [Fact]
    public void GpuMatVecMulBias_MatchesCpu()
    {
        using var gpu = new GpuAccelerator();
        using var mat = new NativeMatrix(4, 8);
        using var bias = new NativeBuffer<float>(4);

        // Fill with known values
        var rng = new Random(42);
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 8; c++)
                mat[r, c] = (float)(rng.NextDouble() * 2 - 1);
            bias[r] = (float)(rng.NextDouble() * 0.1);
        }

        float[] input = new float[8];
        for (int i = 0; i < 8; i++) input[i] = (float)rng.NextDouble();

        // CPU result
        float[] cpuResult = new float[4];
        SimdMath.MatVecMulBias(mat, input, bias.AsReadOnlySpan(), cpuResult);

        // GPU result
        float[] gpuResult = new float[4];
        gpu.MatVecMulBias(mat, input, bias.AsReadOnlySpan(), gpuResult);

        // Verify match
        for (int i = 0; i < 4; i++)
            Assert.Equal(cpuResult[i], gpuResult[i], 1e-3f);
    }
}

public class NativeBufferTests
{
    [Fact]
    public void AllocateAndAccess()
    {
        using var buf = new NativeBuffer<float>(100);
        Assert.Equal(100, buf.Length);

        // Should be zero-filled
        for (int i = 0; i < 100; i++)
            Assert.Equal(0f, buf[i]);

        // Write and read
        buf[0] = 42f;
        buf[99] = -1f;
        Assert.Equal(42f, buf[0]);
        Assert.Equal(-1f, buf[99]);
    }

    [Fact]
    public void SpanAccess()
    {
        using var buf = new NativeBuffer<float>(10);
        buf[3] = 7f;

        Span<float> span = buf.AsSpan();
        Assert.Equal(7f, span[3]);

        ReadOnlySpan<float> ro = buf.AsReadOnlySpan();
        Assert.Equal(7f, ro[3]);
    }

    [Fact]
    public void Clear()
    {
        using var buf = new NativeBuffer<float>(50);
        buf[10] = 999f;
        buf.Clear();
        Assert.Equal(0f, buf[10]);
    }

    [Fact]
    public void CopyFrom()
    {
        using var buf = new NativeBuffer<float>(5);
        float[] src = [1f, 2f, 3f, 4f, 5f];
        buf.CopyFrom(src);
        Assert.Equal(3f, buf[2]);
    }

    [Fact]
    public void BinaryRoundTrip()
    {
        using var buf = new NativeBuffer<float>(10);
        for (int i = 0; i < 10; i++) buf[i] = i * 1.5f;

        using var ms = new MemoryStream();
        using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            buf.WriteTo(bw);

        ms.Position = 0;
        using var buf2 = new NativeBuffer<float>(10);
        using (var br = new BinaryReader(ms))
            buf2.ReadFrom(br);

        for (int i = 0; i < 10; i++)
            Assert.Equal(buf[i], buf2[i]);
    }
}

public class NativeMatrixTests
{
    [Fact]
    public void AllocateAndAccess()
    {
        using var mat = new NativeMatrix(3, 5);
        Assert.Equal(3, mat.Rows);
        Assert.Equal(5, mat.Cols);
        Assert.True(mat.Stride >= 5);
        Assert.True(mat.Stride % 16 == 0); // Padded to 16 floats

        mat[1, 2] = 42f;
        Assert.Equal(42f, mat[1, 2]);
    }

    [Fact]
    public void RowAccess()
    {
        using var mat = new NativeMatrix(2, 8);
        mat[0, 3] = 10f;
        mat[1, 7] = 20f;

        Span<float> row0 = mat.GetRow(0);
        Assert.Equal(8, row0.Length);
        Assert.Equal(10f, row0[3]);

        Span<float> row1 = mat.GetRow(1);
        Assert.Equal(20f, row1[7]);
    }

    [Fact]
    public void BinaryRoundTrip()
    {
        using var mat = new NativeMatrix(3, 7);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 7; c++)
                mat[r, c] = r * 10 + c;

        using var ms = new MemoryStream();
        using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            mat.WriteTo(bw);

        ms.Position = 0;
        using var mat2 = new NativeMatrix(3, 7);
        using (var br = new BinaryReader(ms))
            mat2.ReadFrom(br);

        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 7; c++)
                Assert.Equal(mat[r, c], mat2[r, c]);
    }
}

public class SimdMathTests
{
    [Fact]
    public void DotProduct()
    {
        float[] a = [1f, 2f, 3f, 4f];
        float[] b = [5f, 6f, 7f, 8f];
        float result = SimdMath.Dot(a, b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        Assert.Equal(70f, result, 1e-5f);
    }

    [Fact]
    public void DotProductLarge()
    {
        int n = 784;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = 1f; b[i] = 1f; }
        float result = SimdMath.Dot(a, b);
        Assert.Equal(784f, result, 0.1f);
    }

    [Fact]
    public void MatVecMul()
    {
        using var mat = new NativeMatrix(2, 3);
        mat[0, 0] = 1f; mat[0, 1] = 2f; mat[0, 2] = 3f;
        mat[1, 0] = 4f; mat[1, 1] = 5f; mat[1, 2] = 6f;
        float[] vec = [1f, 1f, 1f];
        float[] result = new float[2];

        SimdMath.MatVecMul(mat, vec, result);
        Assert.Equal(6f, result[0], 1e-5f);  // 1+2+3
        Assert.Equal(15f, result[1], 1e-5f); // 4+5+6
    }

    [Fact]
    public void ReLU()
    {
        float[] data = [-3f, -1f, 0f, 1f, 5f, -0.5f, 2f, -100f];
        SimdMath.ReLUInPlace(data);
        Assert.Equal(0f, data[0]);
        Assert.Equal(0f, data[1]);
        Assert.Equal(0f, data[2]);
        Assert.Equal(1f, data[3]);
        Assert.Equal(5f, data[4]);
        Assert.Equal(0f, data[5]);
        Assert.Equal(2f, data[6]);
        Assert.Equal(0f, data[7]);
    }

    [Fact]
    public void Softmax()
    {
        float[] src = [1f, 2f, 3f];
        float[] dst = new float[3];
        SimdMath.Softmax(src, dst);

        // Probabilities should sum to 1
        float sum = dst[0] + dst[1] + dst[2];
        Assert.Equal(1f, sum, 1e-5f);

        // Monotonicity
        Assert.True(dst[0] < dst[1]);
        Assert.True(dst[1] < dst[2]);
    }

    [Fact]
    public void ArgMax()
    {
        float[] data = [1f, 5f, 3f, 2f, 4f];
        Assert.Equal(1, SimdMath.ArgMax(data));
    }

    [Fact]
    public void TopK_KeepsCorrectCount()
    {
        float[] data = [5f, 1f, 4f, 2f, 3f, 8f, 7f, 6f, 9f, 0f];
        SimdMath.TopK(data, 3);

        // Top 3 should be 9, 8, 7; rest should be 0
        int nonZero = data.Count(x => x > 0);
        Assert.Equal(3, nonZero);
        Assert.True(data.Contains(9f));
        Assert.True(data.Contains(8f));
        Assert.True(data.Contains(7f));
    }

    [Fact]
    public void L2Norm()
    {
        float[] data = [3f, 4f];
        float norm = SimdMath.L2Norm(data);
        Assert.Equal(5f, norm, 1e-5f);
    }

    [Fact]
    public void NormalizeL2()
    {
        float[] data = [3f, 4f];
        SimdMath.NormalizeL2(data);
        Assert.Equal(0.6f, data[0], 1e-5f);
        Assert.Equal(0.8f, data[1], 1e-5f);
    }

    [Fact]
    public void AddScaled()
    {
        float[] dst = [1f, 2f, 3f, 4f];
        float[] src = [10f, 20f, 30f, 40f];
        SimdMath.AddScaled(dst, src, 0.1f);
        Assert.Equal(2f, dst[0], 1e-5f);
        Assert.Equal(4f, dst[1], 1e-5f);
    }

    [Fact]
    public void SpikeEncode()
    {
        float[] src = [0f, 0.3f, 0f, 0.8f, 1f, 0f];
        float[] dst = new float[6];
        SimdMath.SpikeEncode(src, dst, threshold: 0f);
        // Anything > 0 should be 1
        Assert.Equal(0f, dst[0]);
        Assert.Equal(1f, dst[1]);
        Assert.Equal(0f, dst[2]);
        Assert.Equal(1f, dst[3]);
        Assert.Equal(1f, dst[4]);
        Assert.Equal(0f, dst[5]);
    }

    [Fact]
    public void SpikeEncode_WithThreshold()
    {
        float[] src = [0f, 0.3f, 0.5f, 0.8f, 1f];
        float[] dst = new float[5];
        SimdMath.SpikeEncode(src, dst, threshold: 0.5f);
        Assert.Equal(0f, dst[0]);
        Assert.Equal(0f, dst[1]);
        Assert.Equal(0f, dst[2]);  // 0.5 is not > 0.5
        Assert.Equal(1f, dst[3]);
        Assert.Equal(1f, dst[4]);
    }

    [Fact]
    public void SparseForward()
    {
        using var mat = new NativeMatrix(2, 4);
        mat[0, 0] = 1f; mat[0, 1] = 2f; mat[0, 2] = 3f; mat[0, 3] = 4f;
        mat[1, 0] = 5f; mat[1, 1] = 6f; mat[1, 2] = 7f; mat[1, 3] = 8f;

        float[] activity = [0.5f, 0f, 2f, 0f]; // Graded: positions 0 and 2 active
        float[] bias = [0.1f, 0.2f];
        float[] result = new float[2];

        SimdMath.SparseForward(mat, activity, bias, result);
        // Row 0: W[0,0]*0.5 + W[0,2]*2 + bias = 1*0.5 + 3*2 + 0.1 = 6.6
        Assert.Equal(6.6f, result[0], 1e-4f);
        // Row 1: W[1,0]*0.5 + W[1,2]*2 + bias = 5*0.5 + 7*2 + 0.2 = 16.7
        Assert.Equal(16.7f, result[1], 1e-4f);
    }

    [Fact]
    public void CrossEntropyLoss()
    {
        float[] target = [0f, 1f, 0f];
        float[] pred = [0.1f, 0.8f, 0.1f];
        float loss = SimdMath.CrossEntropyLoss(target, pred);
        float expected = -MathF.Log(0.8f);
        Assert.Equal(expected, loss, 1e-4f);
    }
}

public class LearningRuleTests
{
    [Fact]
    public void CompetitiveLearning_MovesTowardInput()
    {
        using var weights = new NativeMatrix(2, 4);
        // Cell 0 weight vector
        weights[0, 0] = 0.5f; weights[0, 1] = 0.5f;
        weights[0, 2] = 0.5f; weights[0, 3] = 0.5f;
        // Cell 1 (inactive, shouldn't change)
        weights[1, 0] = 0.1f; weights[1, 1] = 0.2f;
        weights[1, 2] = 0.3f; weights[1, 3] = 0.4f;

        float[] input = [1f, 0f, 0f, 0f];
        float[] activity = [1f, 0f]; // Only cell 0 is active

        CompetitiveLearning.UpdateWeights(weights, input, activity, 0.5f);

        // Cell 0 should have moved toward [1,0,0,0]
        // W[0] = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        Assert.True(weights[0, 0] > 0.5f);
        // W[1..3] should have decreased
        Assert.True(weights[0, 1] < 0.5f);

        // Cell 1 should be unchanged (was inactive)
        Assert.Equal(0.1f, weights[1, 0], 1e-5f);
    }

    [Fact]
    public void RewardModulatedHebbian_StrengthensOnCorrect()
    {
        using var weights = new NativeMatrix(3, 4);  // 3 classes, 4 hidden
        using var bias = new NativeBuffer<float>(3);

        // Some initial weights
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                weights[r, c] = 0.1f;

        // Graded activity: cell 0 strong, cell 2 weak, cells 1,3 silent
        float[] hiddenActivity = [2f, 0f, 0.5f, 0f];

        // Prediction is correct (winner == label)
        RewardModulatedHebbian.Update(weights, bias, hiddenActivity,
            prediction: 1, correctLabel: 1, learningRate: 0.1f);

        // Winner cell 1's weights: proportional to activity
        // ΔW[1,0] = 0.1 * 2.0 = 0.2 → new = 0.3
        Assert.Equal(0.3f, weights[1, 0], 1e-4f);
        Assert.Equal(0.1f, weights[1, 1], 1e-4f); // Inactive, no change
        // ΔW[1,2] = 0.1 * 0.5 = 0.05 → new = 0.15
        Assert.Equal(0.15f, weights[1, 2], 1e-4f);
        Assert.Equal(0.1f, weights[1, 3], 1e-4f); // Inactive, no change

        // Other cells should be unchanged
        Assert.Equal(0.1f, weights[0, 0], 1e-4f);
        Assert.Equal(0.1f, weights[2, 0], 1e-4f);

        // Bias of winner should have increased
        Assert.True(bias[1] > 0);
    }

    [Fact]
    public void RewardModulatedHebbian_WeakensOnWrong()
    {
        using var weights = new NativeMatrix(3, 4);  // 3 classes, 4 hidden
        using var bias = new NativeBuffer<float>(3);

        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                weights[r, c] = 0.5f;

        // Graded activity: proportional changes
        float[] hiddenActivity = [1f, 0f, 0.5f, 0f];

        // Prediction is wrong: winner=0, correct=2
        RewardModulatedHebbian.Update(weights, bias, hiddenActivity,
            prediction: 0, correctLabel: 2, learningRate: 0.1f);

        // Wrong winner (0): ΔW = -0.1 * activity
        Assert.Equal(0.4f, weights[0, 0], 1e-4f);  // -0.1*1.0
        Assert.Equal(0.45f, weights[0, 2], 1e-4f); // -0.1*0.5

        // Correct class (2): ΔW = +0.1 * activity
        Assert.Equal(0.6f, weights[2, 0], 1e-4f);  // +0.1*1.0
        Assert.Equal(0.55f, weights[2, 2], 1e-4f); // +0.1*0.5

        // Uninvolved class (1) should be unchanged
        Assert.Equal(0.5f, weights[1, 0], 1e-4f);

        // Bias: wrong winner decreased, correct increased
        Assert.True(bias[0] < 0);
        Assert.True(bias[2] > 0);
    }

    [Fact]
    public void Homeostasis_IncreasesThresholdForOveractive()
    {
        using var bias = new NativeBuffer<float>(3);
        using var avgActivity = new NativeBuffer<float>(3);

        // Cell 0: over-active (avg > target)
        avgActivity[0] = 0.5f;
        // Cell 1: under-active (avg < target)
        avgActivity[1] = 0.01f;
        // Cell 2: on target
        avgActivity[2] = 0.1f;

        float[] currentActivity = [1f, 0f, 0f];
        float targetRate = 0.1f;

        Homeostasis.Update(bias, avgActivity, currentActivity, targetRate, 0.01f, 0.99f);

        // Over-active cell should get decreased bias (harder to fire)
        Assert.True(bias[0] < 0);
        // Under-active cell should get increased bias (easier to fire)
        Assert.True(bias[1] > 0);
    }
}

public class NetworkTests
{
    [Fact]
    public void ForwardPassReturnsPrediction()
    {
        var config = new NetworkConfig
        {
            InputSize = 16,
            HiddenSize = 32,
            OutputSize = 5,
            TopKFraction = 0.25f,
            Seed = 123
        };

        using var net = new AbcaNetwork(config);
        float[] input = new float[16];
        for (int i = 0; i < 16; i++) input[i] = 0.5f;

        int pred = net.Forward(input);
        Assert.InRange(pred, 0, 4);
    }

    [Fact]
    public void TrainStepDoesNotCrash()
    {
        var config = new NetworkConfig
        {
            InputSize = 16,
            HiddenSize = 32,
            OutputSize = 5,
            Seed = 42
        };

        using var net = new AbcaNetwork(config);
        float[] input = new float[16];
        var rng = new Random(0);
        for (int i = 0; i < 16; i++) input[i] = (float)rng.NextDouble();

        // Train for several steps
        for (int step = 0; step < 100; step++)
        {
            int label = step % 5;
            net.TrainStep(input, label);
        }

        int pred = net.Forward(input);
        Assert.InRange(pred, 0, 4);
    }

    [Fact]
    public void TrainImproves_XorLikeTask()
    {
        // A simple 4-class task: input quadrant → class
        var config = new NetworkConfig
        {
            InputSize = 2,
            HiddenSize = 20,
            OutputSize = 4,
            TopKFraction = 0.5f,
            HiddenLearningRate = 0.05f,
            OutputLearningRate = 0.05f,
            HomeostasisRate = 0.001f,
            SpikeThreshold = 0.3f,
            Seed = 42
        };

        // Generate simple training data
        float[][] inputs =
        [
            [0.8f, 0.8f],  // Quadrant 0
            [0.2f, 0.8f],  // Quadrant 1
            [0.2f, 0.2f],  // Quadrant 2
            [0.8f, 0.2f],  // Quadrant 3
        ];
        int[] labels = [0, 1, 2, 3];

        using var net = new AbcaNetwork(config);

        // Train for many iterations
        int lastCorrect = 0;
        for (int epoch = 0; epoch < 200; epoch++)
        {
            int correct = 0;
            for (int i = 0; i < 4; i++)
            {
                if (net.TrainStep(inputs[i], labels[i]))
                    correct++;
            }
            lastCorrect = correct;
        }

        // After 200 epochs on 4 examples, should get at least 2/4 correct
        Assert.True(lastCorrect >= 2, $"Expected at least 2/4 correct, got {lastCorrect}");
    }

    [Fact]
    public void ModelSaveLoad_RoundTrip()
    {
        var config = new NetworkConfig
        {
            InputSize = 8,
            HiddenSize = 16,
            OutputSize = 3,
            Seed = 99
        };

        string tempPath = Path.GetTempFileName();
        try
        {
            // Create and train briefly
            using var net = new AbcaNetwork(config);
            float[] input = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f];
            for (int i = 0; i < 10; i++)
                net.TrainStep(input, i % 3);

            int predBefore = net.Forward(input);

            // Save
            ModelSerializer.Save(net, tempPath);

            // Load
            using var loaded = ModelSerializer.Load(tempPath);
            int predAfter = loaded.Forward(input);

            Assert.Equal(predBefore, predAfter);
        }
        finally
        {
            File.Delete(tempPath);
        }
    }
}
