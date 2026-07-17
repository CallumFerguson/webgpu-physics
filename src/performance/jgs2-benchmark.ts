import {
  summarizeDurations,
  summarizeFrameTimes,
  type DurationSummary,
  type FrameTimeSummary,
} from "./metrics";
import { MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT } from "../simulation/gpu/gpu-timestamp";

export interface JGS2PerformanceProfileOptions {
  readonly warmupFrameCount: number;
  readonly measuredFrameCount: number;
}

export const DEFAULT_JGS2_E2E_PERFORMANCE_OPTIONS = {
  warmupFrameCount: 120,
  measuredFrameCount: 600,
} as const satisfies JGS2PerformanceProfileOptions;

/**
 * Short, informational telemetry for individual scene E2E tests. Formal
 * performance baselines should continue to use
 * DEFAULT_JGS2_E2E_PERFORMANCE_OPTIONS.
 */
export const LIGHTWEIGHT_JGS2_E2E_TELEMETRY_OPTIONS = {
  warmupFrameCount: 30,
  measuredFrameCount: 120,
} as const satisfies JGS2PerformanceProfileOptions;

export interface JGS2PerformanceState {
  readonly positions: readonly number[];
  readonly velocities: readonly number[];
}

interface JGS2PerformanceProfileBase {
  readonly workloadId: string;
  readonly timestepSeconds: number;
  readonly iterationsPerStep: number;
  readonly initialSimulationFrame: number;
  readonly finalSimulationFrame: number;
  readonly warmupFrameCount: number;
  readonly measuredFrameCount: number;
  readonly diagnosticReadbacksBefore: number;
  readonly diagnosticReadbacksAfter: number;
  readonly finalState: JGS2PerformanceState;
}

export interface JGS2CpuFrameProfile extends JGS2PerformanceProfileBase {
  readonly samples: {
    readonly endToEndFrameMilliseconds: readonly number[];
    readonly cpuSimulationSubmissionMilliseconds: readonly number[];
    readonly cpuRenderSubmissionMilliseconds: readonly number[];
    readonly cpuFrameSubmissionMilliseconds: readonly number[];
  };
}

export interface JGS2GpuFrameProfile extends JGS2PerformanceProfileBase {
  readonly timestamp: {
    readonly feature: "timestamp-query";
    readonly supported: boolean;
    readonly featureEnabled: boolean;
    readonly reason: string | null;
    readonly timestampMapCount: 0 | 1;
  };
  readonly samples: {
    readonly gpuFrameMilliseconds: readonly number[];
    readonly gpuSimulationStepMilliseconds: readonly number[];
    readonly gpuRenderMilliseconds: readonly number[];
  };
}

export interface JGS2PerformanceBenchmark {
  readonly schema: "org.jgs2.e2e-scene-performance";
  readonly schemaVersion: 1;
  readonly workloadId: string;
  readonly timestepSeconds: number;
  readonly iterationsPerStep: number;
  readonly warmupFrameCount: number;
  readonly measuredFrameCount: number;
  readonly definitions: {
    readonly endToEndFrame: string;
    readonly cpuSimulationSubmission: string;
    readonly cpuRenderSubmission: string;
    readonly cpuFrameSubmission: string;
    readonly gpuFrame: string;
    readonly gpuSimulationStep: string;
    readonly gpuRender: string;
    readonly averageFramesPerSecond: string;
    readonly onePercentLowFramesPerSecond: string;
  };
  readonly endToEndFrame: FrameTimeSummary;
  readonly cpuSimulationSubmission: DurationSummary;
  readonly cpuRenderSubmission: DurationSummary;
  readonly cpuFrameSubmission: DurationSummary;
  readonly gpuFrame: DurationSummary | null;
  readonly gpuSimulationStep: DurationSummary | null;
  readonly gpuRender: DurationSummary | null;
  readonly gpuTimestamp: JGS2GpuFrameProfile["timestamp"];
  readonly stateEquivalent: boolean;
  readonly cpuProfile: JGS2CpuFrameProfile;
  readonly gpuProfile: JGS2GpuFrameProfile;
}

export interface JGS2ComputeBudgetAssessment {
  readonly definition: string;
  readonly serializedStepBudgetMilliseconds: number;
  readonly serializedWallMeanMilliseconds: number;
  readonly serializedWallP95Milliseconds: number;
  readonly meetsSerializedWallMeanBudget: boolean;
  readonly gpuFrameP95Milliseconds: number | null;
  readonly meetsGpuFrameP95Budget: boolean | null;
  readonly passesNecessaryComputeBudget: boolean;
}

const MAX_BENCHMARK_WARMUP_FRAME_COUNT = 10_000;
const MAX_BENCHMARK_ITERATION_COUNT = 10_000;

function validateCount(
  value: number,
  name: string,
  allowZero: boolean,
  maximum: number,
): void {
  if (
    !Number.isSafeInteger(value) ||
    value < (allowZero ? 0 : 1) ||
    value > maximum
  ) {
    throw new RangeError(
      `${name} must be ${allowZero ? "a nonnegative" : "a positive"} safe integer no greater than ${maximum}.`,
    );
  }
}

function validateFrameIndex(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a nonnegative safe integer.`);
  }
}

export function validateJGS2PerformanceProfileOptions(
  options: JGS2PerformanceProfileOptions,
): void {
  validateCount(
    options.warmupFrameCount,
    "warmupFrameCount",
    true,
    MAX_BENCHMARK_WARMUP_FRAME_COUNT,
  );
  validateCount(
    options.measuredFrameCount,
    "measuredFrameCount",
    false,
    MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT,
  );
}

function validateBase(profile: JGS2PerformanceProfileBase): void {
  if (!profile.workloadId) {
    throw new RangeError("workloadId must not be empty.");
  }
  if (!(Number.isFinite(profile.timestepSeconds) && profile.timestepSeconds > 0)) {
    throw new RangeError("timestepSeconds must be finite and positive.");
  }
  validateCount(
    profile.iterationsPerStep,
    "iterationsPerStep",
    false,
    MAX_BENCHMARK_ITERATION_COUNT,
  );
  validateFrameIndex(profile.initialSimulationFrame, "initialSimulationFrame");
  validateFrameIndex(profile.finalSimulationFrame, "finalSimulationFrame");
  validateJGS2PerformanceProfileOptions(profile);
  const expectedFinalFrame =
    profile.initialSimulationFrame +
    profile.warmupFrameCount +
    profile.measuredFrameCount;
  if (!Number.isSafeInteger(expectedFinalFrame)) {
    throw new RangeError("The profiled simulation frame range must remain a safe integer.");
  }
  if (profile.finalSimulationFrame !== expectedFinalFrame) {
    throw new RangeError(
      `finalSimulationFrame ${profile.finalSimulationFrame} does not match expected frame ${expectedFinalFrame}.`,
    );
  }
  if (
    profile.diagnosticReadbacksAfter - profile.diagnosticReadbacksBefore !==
    2
  ) {
    throw new RangeError(
      "Each profile must perform exactly two final state-buffer readbacks outside its timing interval.",
    );
  }
  if (
    profile.finalState.positions.length === 0 ||
    profile.finalState.positions.length !== profile.finalState.velocities.length
  ) {
    throw new RangeError(
      "Final performance state must contain matching nonempty position and velocity buffers.",
    );
  }
  if (
    !profile.finalState.positions.every(Number.isFinite) ||
    !profile.finalState.velocities.every(Number.isFinite)
  ) {
    throw new RangeError("Final performance state must contain only finite values.");
  }
}

function arraysEqual(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => Object.is(value, right[index]))
  );
}

export function buildJGS2PerformanceBenchmark(
  cpuProfile: JGS2CpuFrameProfile,
  gpuProfile: JGS2GpuFrameProfile,
): JGS2PerformanceBenchmark {
  validateBase(cpuProfile);
  validateBase(gpuProfile);
  const commonFields = [
    "workloadId",
    "timestepSeconds",
    "iterationsPerStep",
    "initialSimulationFrame",
    "finalSimulationFrame",
    "warmupFrameCount",
    "measuredFrameCount",
  ] as const;
  for (const field of commonFields) {
    if (cpuProfile[field] !== gpuProfile[field]) {
      throw new RangeError(`CPU and GPU profiles disagree on ${field}.`);
    }
  }

  const cpuSamples = [
    cpuProfile.samples.endToEndFrameMilliseconds,
    cpuProfile.samples.cpuSimulationSubmissionMilliseconds,
    cpuProfile.samples.cpuRenderSubmissionMilliseconds,
    cpuProfile.samples.cpuFrameSubmissionMilliseconds,
  ];
  if (
    cpuSamples.some(
      (samples) => samples.length !== cpuProfile.measuredFrameCount,
    )
  ) {
    throw new RangeError(
      "Every wall/CPU sample set must match measuredFrameCount.",
    );
  }
  const gpuSamples = [
    gpuProfile.samples.gpuFrameMilliseconds,
    gpuProfile.samples.gpuSimulationStepMilliseconds,
    gpuProfile.samples.gpuRenderMilliseconds,
  ];
  const expectedGpuSampleCount = gpuProfile.timestamp.supported
    ? gpuProfile.measuredFrameCount
    : 0;
  if (
    !gpuSamples.every(
      (samples) => samples.length === expectedGpuSampleCount,
    )
  ) {
    throw new RangeError(
      "GPU sample sets must all be complete when timestamp-query is supported and empty when it is unsupported.",
    );
  }
  const expectedTimestampMapCount = gpuProfile.timestamp.supported ? 1 : 0;
  if (
    gpuProfile.timestamp.timestampMapCount !== expectedTimestampMapCount
  ) {
    throw new RangeError(
      "A supported GPU profile must map timestamps exactly once; unsupported profiles must not map.",
    );
  }
  if (
    gpuProfile.timestamp.supported !== gpuProfile.timestamp.featureEnabled ||
    (gpuProfile.timestamp.supported && gpuProfile.timestamp.reason !== null) ||
    (!gpuProfile.timestamp.supported && !gpuProfile.timestamp.reason)
  ) {
    throw new RangeError(
      "GPU timestamp support, enabled state, and unavailability reason are inconsistent.",
    );
  }

  const stateEquivalent =
    arraysEqual(
      cpuProfile.finalState.positions,
      gpuProfile.finalState.positions,
    ) &&
    arraysEqual(
      cpuProfile.finalState.velocities,
      gpuProfile.finalState.velocities,
    );

  return {
    schema: "org.jgs2.e2e-scene-performance",
    schemaVersion: 1,
    workloadId: cpuProfile.workloadId,
    timestepSeconds: cpuProfile.timestepSeconds,
    iterationsPerStep: cpuProfile.iterationsPerStep,
    warmupFrameCount: cpuProfile.warmupFrameCount,
    measuredFrameCount: cpuProfile.measuredFrameCount,
    definitions: {
      endToEndFrame:
        "Wall latency of one serialized simulation step plus one render submission and GPU queue completion.",
      cpuSimulationSubmission:
        "CPU time to prepare, encode, and submit one JGS2 simulation step; GPU execution and waiting are excluded.",
      cpuRenderSubmission:
        "CPU time to encode and submit one scene render; GPU execution and waiting are excluded.",
      cpuFrameSubmission:
        "Sum of CPU simulation-submission and render-submission time for the same measured frame.",
      gpuFrame:
        "GPU timestamp interval from the beginning of simulation through the end of rendering for one frame.",
      gpuSimulationStep:
        "GPU timestamp interval around one JGS2 compute step; rendering and timestamp readback are excluded.",
      gpuRender:
        "GPU timestamp interval around the scene render pass; simulation and timestamp readback are excluded.",
      averageFramesPerSecond:
        "1000 divided by mean milliseconds; this is serialized benchmark throughput, not monitor refresh rate.",
      onePercentLowFramesPerSecond:
        "1000 divided by the mean of the slowest ceil(1%) frame/step durations.",
    },
    endToEndFrame: summarizeFrameTimes(
      cpuProfile.samples.endToEndFrameMilliseconds,
    ),
    cpuSimulationSubmission: summarizeDurations(
      cpuProfile.samples.cpuSimulationSubmissionMilliseconds,
    ),
    cpuRenderSubmission: summarizeDurations(
      cpuProfile.samples.cpuRenderSubmissionMilliseconds,
    ),
    cpuFrameSubmission: summarizeDurations(
      cpuProfile.samples.cpuFrameSubmissionMilliseconds,
    ),
    gpuFrame: gpuProfile.timestamp.supported
      ? summarizeDurations(gpuProfile.samples.gpuFrameMilliseconds)
      : null,
    gpuSimulationStep: gpuProfile.timestamp.supported
      ? summarizeDurations(gpuProfile.samples.gpuSimulationStepMilliseconds)
      : null,
    gpuRender: gpuProfile.timestamp.supported
      ? summarizeDurations(gpuProfile.samples.gpuRenderMilliseconds)
      : null,
    gpuTimestamp: gpuProfile.timestamp,
    stateEquivalent,
    cpuProfile,
    gpuProfile,
  };
}

/**
 * Assess compute capacity without treating a queue-drain latency sample as a
 * production animation frame. The serialized mean measures sustained
 * throughput; GPU p95 measures execution-tail capacity when timestamps exist.
 */
export function assessJGS2ComputeBudget(
  benchmark: JGS2PerformanceBenchmark,
): JGS2ComputeBudgetAssessment {
  const serializedStepBudgetMilliseconds = Math.min(
    1_000 / 60,
    benchmark.timestepSeconds * 1_000,
  );
  const meetsSerializedWallMeanBudget =
    benchmark.endToEndFrame.meanMilliseconds <=
    serializedStepBudgetMilliseconds;
  const gpuFrameP95Milliseconds =
    benchmark.gpuFrame?.p95Milliseconds ?? null;
  const meetsGpuFrameP95Budget =
    gpuFrameP95Milliseconds === null
      ? null
      : gpuFrameP95Milliseconds <= serializedStepBudgetMilliseconds;
  return {
    definition:
      "Necessary compute-capacity condition: serialized wall mean and, when timestamp-query is available, GPU-frame p95 must each fit both a 60 Hz display interval and one simulation timestep. Serialized wall p95 remains diagnostic because per-sample queue drains include browser/event-loop synchronization jitter. This does not measure production requestAnimationFrame scheduling or simulation time rate.",
    serializedStepBudgetMilliseconds,
    serializedWallMeanMilliseconds:
      benchmark.endToEndFrame.meanMilliseconds,
    serializedWallP95Milliseconds: benchmark.endToEndFrame.p95Milliseconds,
    meetsSerializedWallMeanBudget,
    gpuFrameP95Milliseconds,
    meetsGpuFrameP95Budget,
    passesNecessaryComputeBudget:
      meetsSerializedWallMeanBudget && meetsGpuFrameP95Budget !== false,
  };
}
