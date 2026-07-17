import { describe, expect, it } from "vitest";

import {
  assessJGS2ComputeBudget,
  buildJGS2PerformanceBenchmark,
  validateJGS2PerformanceProfileOptions,
  type JGS2CpuFrameProfile,
  type JGS2GpuFrameProfile,
} from "./jgs2-benchmark";

const base = {
  workloadId: "minimal",
  timestepSeconds: 1 / 30,
  iterationsPerStep: 7,
  initialSimulationFrame: 0,
  finalSimulationFrame: 4,
  warmupFrameCount: 1,
  measuredFrameCount: 3,
  diagnosticReadbacksBefore: 0,
  diagnosticReadbacksAfter: 2,
  finalState: {
    positions: [1, 2, 3, 1],
    velocities: [4, 5, 6, 0],
  },
} as const;

function cpuProfile(): JGS2CpuFrameProfile {
  return {
    ...base,
    samples: {
      endToEndFrameMilliseconds: [4, 5, 6],
      cpuSimulationSubmissionMilliseconds: [1, 1.5, 2],
      cpuRenderSubmissionMilliseconds: [0.5, 0.75, 1],
      cpuFrameSubmissionMilliseconds: [1.5, 2.25, 3],
    },
  };
}

function gpuProfile(): JGS2GpuFrameProfile {
  return {
    ...base,
    timestamp: {
      feature: "timestamp-query",
      supported: true,
      featureEnabled: true,
      reason: null,
      timestampMapCount: 1,
    },
    samples: {
      gpuFrameMilliseconds: [3, 3.5, 4],
      gpuSimulationStepMilliseconds: [2, 2.5, 3],
      gpuRenderMilliseconds: [0.5, 0.6, 0.7],
    },
  };
}

describe("JGS2 E2E performance benchmark schema", () => {
  it("keeps wall, CPU, and GPU timing domains separate", () => {
    const report = buildJGS2PerformanceBenchmark(cpuProfile(), gpuProfile());

    expect(report.schemaVersion).toBe(1);
    expect(report.endToEndFrame.meanMilliseconds).toBe(5);
    expect(report.cpuSimulationSubmission.meanMilliseconds).toBe(1.5);
    expect(report.cpuRenderSubmission.meanMilliseconds).toBe(0.75);
    expect(report.cpuFrameSubmission.meanMilliseconds).toBe(2.25);
    expect(report.gpuFrame?.meanMilliseconds).toBe(3.5);
    expect(report.gpuSimulationStep?.meanMilliseconds).toBe(2.5);
    expect(report.gpuRender?.meanMilliseconds).toBeCloseTo(0.6, 12);
    expect(report.stateEquivalent).toBe(true);
    expect(assessJGS2ComputeBudget(report)).toMatchObject({
      serializedStepBudgetMilliseconds: 1_000 / 60,
      meetsSerializedWallMeanBudget: true,
      meetsGpuFrameP95Budget: true,
      passesNecessaryComputeBudget: true,
    });
  });

  it("represents unsupported timestamps explicitly without inventing GPU FPS", () => {
    const unsupported: JGS2GpuFrameProfile = {
      ...gpuProfile(),
      timestamp: {
        feature: "timestamp-query",
        supported: false,
        featureEnabled: false,
        reason: "timestamp-query unavailable",
        timestampMapCount: 0,
      },
      samples: {
        gpuFrameMilliseconds: [],
        gpuSimulationStepMilliseconds: [],
        gpuRenderMilliseconds: [],
      },
    };

    const report = buildJGS2PerformanceBenchmark(cpuProfile(), unsupported);
    expect(report.gpuFrame).toBeNull();
    expect(report.gpuSimulationStep).toBeNull();
    expect(report.gpuRender).toBeNull();
    expect(assessJGS2ComputeBudget(report)).toMatchObject({
      gpuFrameP95Milliseconds: null,
      meetsGpuFrameP95Budget: null,
      passesNecessaryComputeBudget: true,
    });

    expect(() =>
      buildJGS2PerformanceBenchmark(cpuProfile(), {
        ...unsupported,
        samples: {
          ...unsupported.samples,
          gpuFrameMilliseconds: [Number.NaN],
        },
      }),
    ).toThrow(/empty when it is unsupported/);

    expect(() =>
      buildJGS2PerformanceBenchmark(cpuProfile(), {
        ...unsupported,
        timestamp: {
          ...unsupported.timestamp,
          timestampMapCount: 2 as unknown as 0 | 1,
        },
      }),
    ).toThrow(/must not map/);
  });

  it("gates sustained wall throughput and GPU-tail compute independently", () => {
    const fastGpu = gpuProfile();
    const slowWall: JGS2CpuFrameProfile = {
      ...cpuProfile(),
      samples: {
        ...cpuProfile().samples,
        endToEndFrameMilliseconds: [9, 10, 11],
      },
    };
    const wallLimited = buildJGS2PerformanceBenchmark(
      { ...slowWall, timestepSeconds: 1 / 120 },
      { ...fastGpu, timestepSeconds: 1 / 120 },
    );
    expect(assessJGS2ComputeBudget(wallLimited)).toMatchObject({
      meetsSerializedWallMeanBudget: false,
      meetsGpuFrameP95Budget: true,
      passesNecessaryComputeBudget: false,
    });

    const gpuLimited = buildJGS2PerformanceBenchmark(
      { ...cpuProfile(), timestepSeconds: 1 / 120 },
      {
        ...gpuProfile(),
        timestepSeconds: 1 / 120,
        samples: {
          ...gpuProfile().samples,
          gpuFrameMilliseconds: [7, 8, 9],
        },
      },
    );
    expect(assessJGS2ComputeBudget(gpuLimited)).toMatchObject({
      meetsSerializedWallMeanBudget: true,
      meetsGpuFrameP95Budget: false,
      passesNecessaryComputeBudget: false,
    });

    const queueDrainJitter = buildJGS2PerformanceBenchmark(
      {
        ...cpuProfile(),
        timestepSeconds: 1 / 120,
        samples: {
          ...cpuProfile().samples,
          endToEndFrameMilliseconds: [3, 4, 10],
        },
      },
      { ...gpuProfile(), timestepSeconds: 1 / 120 },
    );
    expect(assessJGS2ComputeBudget(queueDrainJitter)).toMatchObject({
      serializedWallP95Milliseconds: 10,
      meetsSerializedWallMeanBudget: true,
      meetsGpuFrameP95Budget: true,
      passesNecessaryComputeBudget: true,
    });
  });

  it("detects any state change caused by timestamp instrumentation", () => {
    const changed: JGS2GpuFrameProfile = {
      ...gpuProfile(),
      finalState: {
        positions: [1, 2, 3.001, 1],
        velocities: [4, 5, 6, 0],
      },
    };

    expect(
      buildJGS2PerformanceBenchmark(cpuProfile(), changed).stateEquivalent,
    ).toBe(false);
  });

  it("fails closed on invalid counts, readbacks, or incomplete samples", () => {
    expect(() =>
      validateJGS2PerformanceProfileOptions({
        warmupFrameCount: 0,
        measuredFrameCount: 0,
      }),
    ).toThrow(/measuredFrameCount/);

    expect(() =>
      buildJGS2PerformanceBenchmark(
        {
          ...cpuProfile(),
          samples: {
            ...cpuProfile().samples,
            cpuSimulationSubmissionMilliseconds: [],
          },
        },
        gpuProfile(),
      ),
    ).toThrow(/measuredFrameCount/);

    expect(() =>
      buildJGS2PerformanceBenchmark(
        {
          ...cpuProfile(),
          diagnosticReadbacksAfter: 1,
        },
        gpuProfile(),
      ),
    ).toThrow(/exactly two/);

    expect(() =>
      buildJGS2PerformanceBenchmark(
        {
          ...cpuProfile(),
          finalState: {
            positions: [1, Number.NaN, 3, 1],
            velocities: [4, 5, 6, 0],
          },
        },
        gpuProfile(),
      ),
    ).toThrow(/finite/);

    expect(() =>
      buildJGS2PerformanceBenchmark(
        cpuProfile(),
        {
          ...gpuProfile(),
          timestamp: {
            ...gpuProfile().timestamp,
            featureEnabled: false,
          },
        },
      ),
    ).toThrow(/inconsistent/);
  });

  it("separates timestamp sample limits from warmup and absolute frame indices", () => {
    expect(() =>
      validateJGS2PerformanceProfileOptions({
        warmupFrameCount: 10_000,
        measuredFrameCount: 2_048,
      }),
    ).not.toThrow();
    expect(() =>
      validateJGS2PerformanceProfileOptions({
        warmupFrameCount: 0,
        measuredFrameCount: 2_049,
      }),
    ).toThrow(/2048/);

    expect(() =>
      buildJGS2PerformanceBenchmark(
        {
          ...cpuProfile(),
          warmupFrameCount: 10_000,
          finalSimulationFrame: 10_003,
        },
        {
          ...gpuProfile(),
          warmupFrameCount: 10_000,
          finalSimulationFrame: 10_003,
        },
      ),
    ).not.toThrow();
  });
});
