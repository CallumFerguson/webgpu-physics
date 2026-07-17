import { describe, expect, it } from "vitest";

import { LivePerformanceCollector } from "./live-metrics";

describe("live production performance metrics", () => {
  it("reports produced-frame FPS, 1% low, CPU submit time, and time rate", () => {
    const collector = new LivePerformanceCollector(1 / 120, 120, 100);
    for (let frame = 0; frame <= 120; frame += 1) {
      collector.recordProducedFrame(frame * 16, 0.4, 0.1);
    }

    const snapshot = collector.snapshot();
    expect(snapshot.frameIntervalSampleCount).toBe(120);
    expect(snapshot.cpuSampleCount).toBe(120);
    expect(snapshot.frameInterval?.averageFramesPerSecond).toBe(62.5);
    expect(snapshot.onePercentLowFramesPerSecond).toBe(62.5);
    expect(snapshot.deliveredSimulationStepsPerSecond).toBe(62.5);
    expect(snapshot.simulationTimeRate).toBeCloseTo(62.5 / 120, 12);
    expect(snapshot.cpuFrameSubmission?.meanMilliseconds).toBeCloseTo(0.5, 12);
    expect(snapshot.cpuSimulationSubmission?.meanMilliseconds).toBeCloseTo(
      0.4,
      12,
    );
    expect(snapshot.cpuRenderSubmission?.meanMilliseconds).toBeCloseTo(0.1, 12);
  });

  it("bounds every rolling distribution and delays 1% low until warmup", () => {
    const collector = new LivePerformanceCollector(1 / 60, 5, 4);
    for (let frame = 0; frame < 4; frame += 1) {
      collector.recordProducedFrame(frame * 10, 1, 2);
    }
    expect(collector.snapshot().onePercentLowFramesPerSecond).toBeNull();

    for (let frame = 4; frame < 12; frame += 1) {
      collector.recordProducedFrame(frame * 10, 1, 2);
    }
    const snapshot = collector.snapshot();
    expect(snapshot.frameIntervalSampleCount).toBe(5);
    expect(snapshot.cpuSampleCount).toBe(5);
    expect(snapshot.onePercentLowFramesPerSecond).toBe(100);
  });

  it("adds asynchronous GPU batches without labeling them as FPS", () => {
    const collector = new LivePerformanceCollector(1 / 60, 4, 2);
    collector.recordGpuTimingBatch({
      frameMilliseconds: [2, 3, 4],
      simulationStepMilliseconds: [1, 2, 3],
      renderMilliseconds: [0.25, 0.5, 0.75],
    });

    const snapshot = collector.snapshot();
    expect(snapshot.gpuSampleCount).toBe(3);
    expect(snapshot.gpuFrame?.meanMilliseconds).toBe(3);
    expect(snapshot.gpuSimulationStep?.p95Milliseconds).toBe(3);
    expect(snapshot.gpuRender?.meanMilliseconds).toBe(0.5);
    expect(snapshot.gpuFrame).not.toHaveProperty("averageFramesPerSecond");
  });

  it("resets cleanly and rejects malformed or non-monotonic samples", () => {
    const collector = new LivePerformanceCollector(1 / 60, 10, 2);
    collector.recordProducedFrame(10, 1, 1);
    expect(() => collector.recordProducedFrame(9, 1, 1)).toThrow(/monotonic/);
    expect(() => collector.recordProducedFrame(11, -1, 1)).toThrow(
      /nonnegative/,
    );
    expect(() =>
      collector.recordGpuTimingBatch({
        frameMilliseconds: [1],
        simulationStepMilliseconds: [],
        renderMilliseconds: [1],
      }),
    ).toThrow(/matching nonempty/);

    collector.reset();
    expect(collector.snapshot()).toMatchObject({
      frameIntervalSampleCount: 0,
      cpuSampleCount: 0,
      gpuSampleCount: 0,
      frameInterval: null,
      cpuFrameSubmission: null,
      gpuFrame: null,
    });
  });
});
