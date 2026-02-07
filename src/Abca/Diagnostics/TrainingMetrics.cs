// ============================================================================
// ABCA - Asynchronous Bio-inspired Computing Architecture
// Module: Diagnostics / TrainingMetrics
// Purpose: Epoch-level training metrics and performance monitoring.
// ============================================================================

using System.Diagnostics;

namespace Abca.Diagnostics;

/// <summary>
/// Records metrics for a single training epoch.
/// </summary>
public sealed class EpochMetrics
{
    public int Epoch { get; init; }
    public float TrainAccuracy { get; init; }
    public float TestAccuracy { get; init; }
    public float TrainLoss { get; init; }
    public float TestLoss { get; init; }
    public TimeSpan Duration { get; init; }
    public float SamplesPerSecond { get; init; }

    public override string ToString() =>
        $"Epoch {Epoch,3} | " +
        $"Train: {TrainAccuracy,7:P2} loss={TrainLoss,8:F4} | " +
        $"Test: {TestAccuracy,7:P2} loss={TestLoss,8:F4} | " +
        $"{Duration.TotalSeconds,6:F2}s ({SamplesPerSecond,8:F0} samples/s)";
}

/// <summary>
/// Accumulates training metrics across epochs and provides summary statistics.
/// </summary>
public sealed class TrainingHistory
{
    private readonly List<EpochMetrics> _epochs = [];

    /// <summary>All recorded epoch metrics.</summary>
    public IReadOnlyList<EpochMetrics> Epochs => _epochs;

    /// <summary>Best test accuracy achieved.</summary>
    public float BestTestAccuracy { get; private set; }

    /// <summary>Epoch number that achieved the best test accuracy.</summary>
    public int BestEpoch { get; private set; }

    /// <summary>Total training wall-clock time.</summary>
    public TimeSpan TotalDuration => TimeSpan.FromTicks(
        _epochs.Sum(e => e.Duration.Ticks));

    /// <summary>Records a new epoch's metrics.</summary>
    public void Record(EpochMetrics metrics)
    {
        _epochs.Add(metrics);
        if (metrics.TestAccuracy > BestTestAccuracy)
        {
            BestTestAccuracy = metrics.TestAccuracy;
            BestEpoch = metrics.Epoch;
        }
    }

    /// <summary>Prints a summary of the training history.</summary>
    public void PrintSummary()
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                        ABCA Training Summary                               ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Total epochs:        {_epochs.Count,5}                                              ║");
        Console.WriteLine($"║  Total time:          {TotalDuration.TotalSeconds,8:F2}s                                        ║");
        Console.WriteLine($"║  Best test accuracy:  {BestTestAccuracy,8:P2}  (epoch {BestEpoch})                          ║");

        if (_epochs.Count > 0)
        {
            float avgSps = _epochs.Average(e => e.SamplesPerSecond);
            Console.WriteLine($"║  Avg throughput:      {avgSps,8:F0} samples/s                                ║");
        }

        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
    }
}

/// <summary>
/// High-resolution performance timer using Stopwatch.
/// </summary>
public sealed class PerfTimer : IDisposable
{
    private readonly Stopwatch _sw = Stopwatch.StartNew();
    private readonly string _label;

    public PerfTimer(string label)
    {
        _label = label;
    }

    public TimeSpan Elapsed => _sw.Elapsed;

    public void Dispose()
    {
        _sw.Stop();
        Console.WriteLine($"  [{_label}] {_sw.Elapsed.TotalMilliseconds:F1}ms");
    }
}
