import {
  summarizeDurations,
  summarizeFrameTimes,
  type DurationSummary,
  type FrameTimeSummary,
} from "./metrics";

export const DEFAULT_LIVE_PERFORMANCE_WINDOW_SIZE = 300;
export const DEFAULT_LIVE_ONE_PERCENT_LOW_SAMPLE_COUNT = 100;

export interface LiveGpuTimingBatch {
  readonly frameMilliseconds: readonly number[];
  readonly simulationStepMilliseconds: readonly number[];
  readonly renderMilliseconds: readonly number[];
  readonly simulationStepCounts: readonly number[];
}

export interface LivePerformanceSnapshot {
  readonly windowSize: number;
  readonly frameIntervalSampleCount: number;
  readonly cpuSampleCount: number;
  readonly gpuSampleCount: number;
  readonly frameInterval: FrameTimeSummary | null;
  readonly onePercentLowFramesPerSecond: number | null;
  readonly deliveredSimulationStepsPerSecond: number | null;
  readonly simulationTimeRate: number | null;
  readonly cpuFrameSubmission: DurationSummary | null;
  readonly cpuSimulationSubmission: DurationSummary | null;
  readonly cpuRenderSubmission: DurationSummary | null;
  readonly gpuFrame: DurationSummary | null;
  readonly gpuSimulationStep: DurationSummary | null;
  readonly gpuRender: DurationSummary | null;
}

function validateDuration(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be finite nonnegative milliseconds.`);
  }
}

function appendRolling(
  destination: number[],
  values: readonly number[],
  maximumLength: number,
): void {
  destination.push(...values);
  if (destination.length > maximumLength) {
    destination.splice(0, destination.length - maximumLength);
  }
}

function summarizeOrNull(samples: readonly number[]): DurationSummary | null {
  return samples.length > 0 ? summarizeDurations(samples) : null;
}

/**
 * Rolling metrics for frames actually submitted by the production loop.
 * requestAnimationFrame callbacks that do not produce a simulation/render
 * frame are intentionally excluded.
 */
export class LivePerformanceCollector {
  private readonly frameIntervals: number[] = [];
  private readonly cpuFrames: number[] = [];
  private readonly cpuSimulationSteps: number[] = [];
  private readonly cpuRenders: number[] = [];
  private readonly gpuFrames: number[] = [];
  private readonly gpuSimulationSteps: number[] = [];
  private readonly gpuRenders: number[] = [];
  private readonly intervalSimulationStepCounts: number[] = [];
  private previousProducedFrameStartMilliseconds: number | null = null;
  private previousProducedSimulationStepCount: number | null = null;

  constructor(
    private readonly timestepSeconds: number,
    private readonly windowSize = DEFAULT_LIVE_PERFORMANCE_WINDOW_SIZE,
    private readonly onePercentLowMinimumSamples =
      DEFAULT_LIVE_ONE_PERCENT_LOW_SAMPLE_COUNT,
  ) {
    if (!(Number.isFinite(timestepSeconds) && timestepSeconds > 0)) {
      throw new RangeError("timestepSeconds must be finite and positive.");
    }
    if (!Number.isSafeInteger(windowSize) || windowSize < 1) {
      throw new RangeError("windowSize must be a positive safe integer.");
    }
    if (
      !Number.isSafeInteger(onePercentLowMinimumSamples) ||
      onePercentLowMinimumSamples < 1 ||
      onePercentLowMinimumSamples > windowSize
    ) {
      throw new RangeError(
        "onePercentLowMinimumSamples must be a positive safe integer no greater than windowSize.",
      );
    }
  }

  recordProducedFrame(
    frameStartMilliseconds: number,
    cpuSimulationSubmissionMilliseconds: number,
    cpuRenderSubmissionMilliseconds: number,
    simulationStepCount = 1,
  ): void {
    validateDuration(frameStartMilliseconds, "frameStartMilliseconds");
    validateDuration(
      cpuSimulationSubmissionMilliseconds,
      "cpuSimulationSubmissionMilliseconds",
    );
    validateDuration(
      cpuRenderSubmissionMilliseconds,
      "cpuRenderSubmissionMilliseconds",
    );
    if (!Number.isSafeInteger(simulationStepCount) || simulationStepCount < 1) {
      throw new RangeError("simulationStepCount must be a positive safe integer.");
    }
    if (
      this.previousProducedFrameStartMilliseconds !== null &&
      frameStartMilliseconds < this.previousProducedFrameStartMilliseconds
    ) {
      throw new RangeError("Produced-frame start times must be monotonic.");
    }
    if (this.previousProducedFrameStartMilliseconds !== null) {
      appendRolling(
        this.frameIntervals,
        [frameStartMilliseconds - this.previousProducedFrameStartMilliseconds],
        this.windowSize,
      );
      appendRolling(
        this.intervalSimulationStepCounts,
        [this.previousProducedSimulationStepCount!],
        this.windowSize,
      );
    }
    this.previousProducedFrameStartMilliseconds = frameStartMilliseconds;
    this.previousProducedSimulationStepCount = simulationStepCount;
    appendRolling(
      this.cpuSimulationSteps,
      Array.from(
        { length: simulationStepCount },
        () => cpuSimulationSubmissionMilliseconds / simulationStepCount,
      ),
      this.windowSize,
    );
    appendRolling(
      this.cpuRenders,
      [cpuRenderSubmissionMilliseconds],
      this.windowSize,
    );
    appendRolling(
      this.cpuFrames,
      [
        cpuSimulationSubmissionMilliseconds +
          cpuRenderSubmissionMilliseconds,
      ],
      this.windowSize,
    );
  }

  recordGpuTimingBatch(batch: LiveGpuTimingBatch): void {
    const sampleCount = batch.frameMilliseconds.length;
    if (
      sampleCount === 0 ||
      batch.simulationStepMilliseconds.length !== sampleCount ||
      batch.renderMilliseconds.length !== sampleCount ||
      batch.simulationStepCounts.length !== sampleCount
    ) {
      throw new RangeError(
        "Live GPU timing batches must contain matching nonempty sample arrays.",
      );
    }
    for (const [name, samples] of [
      ["frameMilliseconds", batch.frameMilliseconds],
      ["simulationStepMilliseconds", batch.simulationStepMilliseconds],
      ["renderMilliseconds", batch.renderMilliseconds],
    ] as const) {
      for (const sample of samples) {
        validateDuration(sample, name);
      }
    }
    const normalizedSimulationSteps: number[] = [];
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const simulationStepCount = batch.simulationStepCounts[sample]!;
      if (!Number.isSafeInteger(simulationStepCount) || simulationStepCount < 1) {
        throw new RangeError(
          "Live GPU simulation step counts must be positive safe integers.",
        );
      }
      const averageStepMilliseconds =
        batch.simulationStepMilliseconds[sample]! / simulationStepCount;
      for (let step = 0; step < simulationStepCount; step += 1) {
        normalizedSimulationSteps.push(averageStepMilliseconds);
      }
    }
    appendRolling(this.gpuFrames, batch.frameMilliseconds, this.windowSize);
    appendRolling(
      this.gpuSimulationSteps,
      normalizedSimulationSteps,
      this.windowSize,
    );
    appendRolling(this.gpuRenders, batch.renderMilliseconds, this.windowSize);
  }

  snapshot(): LivePerformanceSnapshot {
    const frameInterval =
      this.frameIntervals.length > 0
        ? summarizeFrameTimes(this.frameIntervals)
        : null;
    const totalFrameIntervalMilliseconds = this.frameIntervals.reduce(
      (sum, interval) => sum + interval,
      0,
    );
    const deliveredSimulationStepsPerSecond =
      totalFrameIntervalMilliseconds > 0
        ? (1000 *
            this.intervalSimulationStepCounts.reduce(
              (sum, count) => sum + count,
              0,
            )) /
          totalFrameIntervalMilliseconds
        : null;
    return {
      windowSize: this.windowSize,
      frameIntervalSampleCount: this.frameIntervals.length,
      cpuSampleCount: this.cpuFrames.length,
      gpuSampleCount: this.gpuFrames.length,
      frameInterval,
      onePercentLowFramesPerSecond:
        this.frameIntervals.length >= this.onePercentLowMinimumSamples
          ? (frameInterval?.onePercentLowFramesPerSecond ?? null)
          : null,
      deliveredSimulationStepsPerSecond,
      simulationTimeRate:
        deliveredSimulationStepsPerSecond === null
          ? null
          : deliveredSimulationStepsPerSecond * this.timestepSeconds,
      cpuFrameSubmission: summarizeOrNull(this.cpuFrames),
      cpuSimulationSubmission: summarizeOrNull(this.cpuSimulationSteps),
      cpuRenderSubmission: summarizeOrNull(this.cpuRenders),
      gpuFrame: summarizeOrNull(this.gpuFrames),
      gpuSimulationStep: summarizeOrNull(this.gpuSimulationSteps),
      gpuRender: summarizeOrNull(this.gpuRenders),
    };
  }

  reset(): void {
    this.frameIntervals.length = 0;
    this.cpuFrames.length = 0;
    this.cpuSimulationSteps.length = 0;
    this.cpuRenders.length = 0;
    this.gpuFrames.length = 0;
    this.gpuSimulationSteps.length = 0;
    this.gpuRenders.length = 0;
    this.intervalSimulationStepCounts.length = 0;
    this.previousProducedFrameStartMilliseconds = null;
    this.previousProducedSimulationStepCount = null;
  }
}
