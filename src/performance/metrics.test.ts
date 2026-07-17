import { describe, expect, it } from "vitest";

import { summarizeDurations, summarizeFrameTimes } from "./metrics";

describe("standard frame-time metrics", () => {
  it("reports latency percentiles, average FPS, and the conventional 1% low", () => {
    const summary = summarizeFrameTimes(
      Array.from({ length: 100 }, (_unused, index) => index + 1),
    );

    expect(summary).toMatchObject({
      sampleCount: 100,
      minimumMilliseconds: 1,
      maximumMilliseconds: 100,
      meanMilliseconds: 50.5,
      p50Milliseconds: 50,
      p95Milliseconds: 95,
      p99Milliseconds: 99,
      slowestOnePercentSampleCount: 1,
      slowestOnePercentMeanMilliseconds: 100,
      percentileMethod: "nearest-rank",
      onePercentLowMethod: "reciprocal-of-slowest-one-percent-mean",
    });
    expect(summary.averageFramesPerSecond).toBeCloseTo(1_000 / 50.5, 12);
    expect(summary.onePercentLowFramesPerSecond).toBe(10);
  });

  it("rounds the slow tail up so a 120-frame run averages two samples", () => {
    const summary = summarizeFrameTimes(
      Array.from({ length: 120 }, (_unused, index) => index + 1),
    );

    expect(summary.slowestOnePercentSampleCount).toBe(2);
    expect(summary.slowestOnePercentMeanMilliseconds).toBe(119.5);
    expect(summary.onePercentLowFramesPerSecond).toBeCloseTo(1_000 / 119.5, 12);
  });

  it("keeps zero-duration timer samples honest instead of producing Infinity", () => {
    const summary = summarizeFrameTimes([0, 0, 0]);

    expect(summary.averageFramesPerSecond).toBeNull();
    expect(summary.onePercentLowFramesPerSecond).toBeNull();
  });

  it("does not label CPU/GPU operation throughput as FPS", () => {
    const summary = summarizeDurations([1, 2, 3]);

    expect(summary.meanMilliseconds).toBe(2);
    expect(summary.p95Milliseconds).toBe(3);
    expect(summary).not.toHaveProperty("averageFramesPerSecond");
    expect(summary).not.toHaveProperty("onePercentLowFramesPerSecond");
  });

  it("rejects missing, negative, or non-finite samples", () => {
    expect(() => summarizeFrameTimes([])).toThrow(/empty/);
    expect(() => summarizeFrameTimes([1, -1])).toThrow(/nonnegative/);
    expect(() => summarizeFrameTimes([Number.NaN])).toThrow(/finite/);
    expect(() => summarizeFrameTimes([Number.POSITIVE_INFINITY])).toThrow(
      /finite/,
    );
    expect(() => summarizeDurations([])).toThrow(/empty/);
  });
});
