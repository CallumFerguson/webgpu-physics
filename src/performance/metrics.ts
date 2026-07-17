export interface DurationSummary {
  readonly sampleCount: number;
  readonly minimumMilliseconds: number;
  readonly maximumMilliseconds: number;
  readonly meanMilliseconds: number;
  readonly p50Milliseconds: number;
  readonly p95Milliseconds: number;
  readonly p99Milliseconds: number;
  readonly percentileMethod: "nearest-rank";
}

export interface FrameTimeSummary extends DurationSummary {
  readonly slowestOnePercentSampleCount: number;
  readonly slowestOnePercentMeanMilliseconds: number;
  readonly averageFramesPerSecond: number | null;
  readonly onePercentLowFramesPerSecond: number | null;
  readonly onePercentLowMethod: "reciprocal-of-slowest-one-percent-mean";
}

function reciprocalFramesPerSecond(milliseconds: number): number | null {
  return milliseconds > 0 ? 1_000 / milliseconds : null;
}

/** Validate and sort a nonnegative operation-duration distribution. */
function summarizeDurationsInternal(
  samplesMilliseconds: readonly number[],
): {
  readonly summary: DurationSummary;
  readonly sorted: readonly number[];
} {
  if (samplesMilliseconds.length === 0) {
    throw new RangeError("Cannot summarize an empty duration sample.");
  }
  for (const sample of samplesMilliseconds) {
    if (!Number.isFinite(sample) || sample < 0) {
      throw new RangeError(
        "Duration samples must be finite nonnegative milliseconds.",
      );
    }
  }

  const sorted = [...samplesMilliseconds].sort((left, right) => left - right);
  const percentile = (fraction: number): number => {
    const index = Math.max(0, Math.ceil(fraction * sorted.length) - 1);
    return sorted[index]!;
  };
  const meanMilliseconds =
    samplesMilliseconds.reduce((sum, sample) => sum + sample, 0) /
    samplesMilliseconds.length;

  return {
    summary: {
      sampleCount: samplesMilliseconds.length,
      minimumMilliseconds: sorted[0]!,
      maximumMilliseconds: sorted.at(-1)!,
      meanMilliseconds,
      p50Milliseconds: percentile(0.5),
      p95Milliseconds: percentile(0.95),
      p99Milliseconds: percentile(0.99),
      percentileMethod: "nearest-rank",
    },
    sorted,
  };
}

/** Summarize a nonnegative operation-duration distribution in milliseconds. */
export function summarizeDurations(
  samplesMilliseconds: readonly number[],
): DurationSummary {
  return summarizeDurationsInternal(samplesMilliseconds).summary;
}

/**
 * Summarize nonnegative end-to-end frame durations without averaging
 * instantaneous FPS values. Average FPS is 1000 / mean(frame time). The 1%
 * low is 1000 / mean(the slowest ceil(1%) of frame times).
 */
export function summarizeFrameTimes(
  samplesMilliseconds: readonly number[],
): FrameTimeSummary {
  const { summary, sorted } = summarizeDurationsInternal(samplesMilliseconds);
  const slowestOnePercentSampleCount = Math.max(
    1,
    Math.ceil(samplesMilliseconds.length * 0.01),
  );
  const slowestOnePercent = sorted.slice(-slowestOnePercentSampleCount);
  const slowestOnePercentMeanMilliseconds =
    slowestOnePercent.reduce((sum, sample) => sum + sample, 0) /
    slowestOnePercent.length;

  return {
    ...summary,
    slowestOnePercentSampleCount,
    slowestOnePercentMeanMilliseconds,
    averageFramesPerSecond: reciprocalFramesPerSecond(
      summary.meanMilliseconds,
    ),
    onePercentLowFramesPerSecond: reciprocalFramesPerSecond(
      slowestOnePercentMeanMilliseconds,
    ),
    onePercentLowMethod: "reciprocal-of-slowest-one-percent-mean",
  };
}
