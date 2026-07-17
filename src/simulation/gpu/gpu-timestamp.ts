export const GPU_TIMESTAMP_FEATURE = "timestamp-query" satisfies GPUFeatureName;

export interface GpuTimestampMeasurement {
  readonly feature: typeof GPU_TIMESTAMP_FEATURE;
  readonly supported: boolean;
  readonly featureEnabled: boolean;
  readonly gpuNanoseconds: number | null;
  readonly gpuMilliseconds: number | null;
  readonly reason: string | null;
}

export interface GpuTimestampBatchMeasurement {
  readonly feature: typeof GPU_TIMESTAMP_FEATURE;
  readonly supported: boolean;
  readonly featureEnabled: boolean;
  readonly gpuNanoseconds: readonly number[] | null;
  readonly gpuMilliseconds: readonly number[] | null;
  readonly reason: string | null;
  readonly frameCount: number;
  readonly timestampMapCount: 0 | 1;
}

export interface GpuTimestampIntervalWrites {
  readonly querySet: GPUQuerySet;
  readonly startWriteIndex: number;
  readonly endWriteIndex: number;
}

export interface GpuTimestampFrameWrites {
  readonly simulation: GpuTimestampIntervalWrites;
  readonly render: GpuTimestampIntervalWrites;
}

export interface GpuTimestampFrameProfileMeasurement {
  readonly feature: typeof GPU_TIMESTAMP_FEATURE;
  readonly supported: boolean;
  readonly featureEnabled: boolean;
  readonly gpuFrameMilliseconds: readonly number[] | null;
  readonly gpuSimulationStepMilliseconds: readonly number[] | null;
  readonly gpuRenderMilliseconds: readonly number[] | null;
  readonly reason: string | null;
  readonly frameCount: number;
  readonly timestampMapCount: 0 | 1;
}

interface GpuTimestampResources {
  readonly querySet: GPUQuerySet;
  readonly resolve: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly queryCount: number;
}

interface GpuTimestampFrameProfileResources {
  readonly querySet: GPUQuerySet;
  readonly resolve: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly queryCount: number;
}

const TIMESTAMP_QUERY_COUNT = 2;
export const MAX_GPU_TIMESTAMP_QUERY_COUNT = 8_192;
export const MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT =
  MAX_GPU_TIMESTAMP_QUERY_COUNT / TIMESTAMP_QUERY_COUNT;

function timestampByteLength(queryCount: number): number {
  return queryCount * BigUint64Array.BYTES_PER_ELEMENT;
}

export function decodeGpuTimestampMeasurement(
  startNanoseconds: bigint,
  endNanoseconds: bigint,
): GpuTimestampMeasurement {
  if (endNanoseconds < startNanoseconds) {
    throw new Error(
      `GPU timestamp interval ended before it began (${endNanoseconds} < ${startNanoseconds}).`,
    );
  }
  const gpuNanoseconds = Number(endNanoseconds - startNanoseconds);
  const gpuMilliseconds = gpuNanoseconds / 1_000_000;
  if (!Number.isFinite(gpuMilliseconds) || gpuMilliseconds < 0) {
    throw new Error(`GPU timestamp interval is invalid: ${gpuMilliseconds} ms.`);
  }
  return {
    feature: GPU_TIMESTAMP_FEATURE,
    supported: true,
    featureEnabled: true,
    gpuNanoseconds,
    gpuMilliseconds,
    reason: null,
  };
}

/**
 * Explicit, benchmark-only timer for one submitted GPU command stream.
 *
 * Resources are allocated lazily. Production callers that never invoke
 * measure() create no query set, resolve buffer, readback buffer, timestamp
 * passes, or synchronization point.
 */
export class GpuTimestampFrameTimer {
  private resources?: GpuTimestampResources;
  private measuring = false;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
  ) {}

  get supported(): boolean {
    return this.device.features.has(GPU_TIMESTAMP_FEATURE);
  }

  async measure(
    encodeCommandStream: (encoder: GPUCommandEncoder) => void,
  ): Promise<GpuTimestampMeasurement> {
    const batch = await this.measureFrames(1, (encoder) => {
      encodeCommandStream(encoder);
    });
    if (!batch.supported) {
      return {
        feature: batch.feature,
        supported: false,
        featureEnabled: batch.featureEnabled,
        gpuNanoseconds: null,
        gpuMilliseconds: null,
        reason: batch.reason,
      };
    }
    return {
      feature: batch.feature,
      supported: true,
      featureEnabled: batch.featureEnabled,
      gpuNanoseconds: batch.gpuNanoseconds![0]!,
      gpuMilliseconds: batch.gpuMilliseconds![0]!,
      reason: null,
    };
  }

  /**
   * Measure several consecutive GPU frames with one query resolve and map.
   * This avoids inserting a CPU/GPU synchronization point between samples.
   */
  async measureFrames(
    frameCount: number,
    encodeFrame: (encoder: GPUCommandEncoder, frameIndex: number) => void,
  ): Promise<GpuTimestampBatchMeasurement> {
    this.assertUsable();
    if (
      !Number.isSafeInteger(frameCount) ||
      frameCount < 1 ||
      frameCount > MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT
    ) {
      throw new RangeError(
        `frameCount must be a positive safe integer no greater than ${MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT}.`,
      );
    }
    if (this.measuring) {
      throw new Error("A GPU timestamp measurement is already in progress.");
    }
    this.measuring = true;

    try {
      const encoder = this.device.createCommandEncoder({
        label: `${this.label}-timed-command-encoder`,
      });

      if (!this.supported) {
        for (let frame = 0; frame < frameCount; frame += 1) {
          encodeFrame(encoder, frame);
        }
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        return {
          feature: GPU_TIMESTAMP_FEATURE,
          supported: false,
          featureEnabled: false,
          gpuNanoseconds: null,
          gpuMilliseconds: null,
          reason: "The WebGPU device does not have the timestamp-query feature enabled.",
          frameCount,
          timestampMapCount: 0,
        };
      }

      const queryCount = frameCount * TIMESTAMP_QUERY_COUNT;
      const { querySet, resolve, readback } = this.getResources(queryCount);

      for (let frame = 0; frame < frameCount; frame += 1) {
        const startQuery = frame * TIMESTAMP_QUERY_COUNT;
        const endQuery = startQuery + 1;
        // Empty passes place timestamps directly around the production
        // simulation commands without modifying those passes.
        encoder
          .beginComputePass({
            label: `${this.label}-timestamp-start-pass-${frame}`,
            timestampWrites: {
              querySet,
              endOfPassWriteIndex: startQuery,
            },
          })
          .end();
        encodeFrame(encoder, frame);
        encoder
          .beginComputePass({
            label: `${this.label}-timestamp-end-pass-${frame}`,
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: endQuery,
            },
          })
          .end();
      }
      encoder.resolveQuerySet(querySet, 0, queryCount, resolve, 0);
      encoder.copyBufferToBuffer(
        resolve,
        0,
        readback,
        0,
        timestampByteLength(queryCount),
      );
      this.device.queue.submit([encoder.finish()]);

      let mapped = false;
      try {
        await readback.mapAsync(GPUMapMode.READ);
        mapped = true;
        const timestamps = new BigUint64Array(readback.getMappedRange());
        const gpuNanoseconds: number[] = [];
        const gpuMilliseconds: number[] = [];
        for (let frame = 0; frame < frameCount; frame += 1) {
          const measurement = decodeGpuTimestampMeasurement(
            timestamps[frame * TIMESTAMP_QUERY_COUNT]!,
            timestamps[frame * TIMESTAMP_QUERY_COUNT + 1]!,
          );
          gpuNanoseconds.push(measurement.gpuNanoseconds!);
          gpuMilliseconds.push(measurement.gpuMilliseconds!);
        }
        return {
          feature: GPU_TIMESTAMP_FEATURE,
          supported: true,
          featureEnabled: true,
          gpuNanoseconds,
          gpuMilliseconds,
          reason: null,
          frameCount,
          timestampMapCount: 1,
        };
      } finally {
        if (mapped) {
          readback.unmap();
        }
      }
    } finally {
      this.measuring = false;
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.resources?.querySet.destroy();
    this.resources?.resolve.destroy();
    this.resources?.readback.destroy();
    this.resources = undefined;
  }

  private getResources(queryCount: number): GpuTimestampResources {
    if (this.resources?.queryCount === queryCount) {
      return this.resources;
    }
    this.resources?.querySet.destroy();
    this.resources?.resolve.destroy();
    this.resources?.readback.destroy();
    const querySet = this.device.createQuerySet({
      label: `${this.label}-timestamp-query-set`,
      type: "timestamp",
      count: queryCount,
    });
    const byteLength = timestampByteLength(queryCount);
    const resolve = this.device.createBuffer({
      label: `${this.label}-timestamp-resolve`,
      size: byteLength,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: `${this.label}-timestamp-readback`,
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.resources = { querySet, resolve, readback, queryCount };
    return this.resources;
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("GpuTimestampFrameTimer has been destroyed.");
    }
  }
}

const FRAME_PROFILE_QUERIES_PER_FRAME = 4;
export const MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT =
  MAX_GPU_TIMESTAMP_QUERY_COUNT / FRAME_PROFILE_QUERIES_PER_FRAME;

/**
 * Benchmark-only profiler for production-shaped simulation and render
 * submissions. Each frame receives distinct simulation and render timestamp
 * intervals; all queries are resolved and mapped once after the run.
 */
export class GpuTimestampFrameProfiler {
  private resources?: GpuTimestampFrameProfileResources;
  private measuring = false;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
  ) {}

  get supported(): boolean {
    return this.device.features.has(GPU_TIMESTAMP_FEATURE);
  }

  async measureFrames(
    frameCount: number,
    submitFrame: (
      writes: GpuTimestampFrameWrites | null,
      frameIndex: number,
    ) => Promise<void>,
  ): Promise<GpuTimestampFrameProfileMeasurement> {
    this.assertUsable();
    if (
      !Number.isSafeInteger(frameCount) ||
      frameCount < 1 ||
      frameCount > MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT
    ) {
      throw new RangeError(
        `frameCount must be a positive safe integer no greater than ${MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT}.`,
      );
    }
    if (this.measuring) {
      throw new Error("A GPU frame timestamp profile is already in progress.");
    }
    this.measuring = true;
    try {
      if (!this.supported) {
        for (let frame = 0; frame < frameCount; frame += 1) {
          await submitFrame(null, frame);
        }
        return {
          feature: GPU_TIMESTAMP_FEATURE,
          supported: false,
          featureEnabled: false,
          gpuFrameMilliseconds: null,
          gpuSimulationStepMilliseconds: null,
          gpuRenderMilliseconds: null,
          reason:
            "The WebGPU device does not have the timestamp-query feature enabled.",
          frameCount,
          timestampMapCount: 0,
        };
      }

      const queryCount = frameCount * FRAME_PROFILE_QUERIES_PER_FRAME;
      const { querySet, resolve, readback } = this.getResources(queryCount);
      for (let frame = 0; frame < frameCount; frame += 1) {
        const queryBase = frame * FRAME_PROFILE_QUERIES_PER_FRAME;
        await submitFrame(
          {
            simulation: {
              querySet,
              startWriteIndex: queryBase,
              endWriteIndex: queryBase + 1,
            },
            render: {
              querySet,
              startWriteIndex: queryBase + 2,
              endWriteIndex: queryBase + 3,
            },
          },
          frame,
        );
      }

      const byteLength = timestampByteLength(queryCount);
      const encoder = this.device.createCommandEncoder({
        label: `${this.label}-frame-profile-resolve-encoder`,
      });
      encoder.resolveQuerySet(querySet, 0, queryCount, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, readback, 0, byteLength);
      this.device.queue.submit([encoder.finish()]);

      let mapped = false;
      try {
        await readback.mapAsync(GPUMapMode.READ);
        mapped = true;
        const timestamps = new BigUint64Array(readback.getMappedRange());
        const gpuFrameMilliseconds: number[] = [];
        const gpuSimulationStepMilliseconds: number[] = [];
        const gpuRenderMilliseconds: number[] = [];
        for (let frame = 0; frame < frameCount; frame += 1) {
          const queryBase = frame * FRAME_PROFILE_QUERIES_PER_FRAME;
          const simulation = decodeGpuTimestampMeasurement(
            timestamps[queryBase]!,
            timestamps[queryBase + 1]!,
          );
          const render = decodeGpuTimestampMeasurement(
            timestamps[queryBase + 2]!,
            timestamps[queryBase + 3]!,
          );
          const wholeFrame = decodeGpuTimestampMeasurement(
            timestamps[queryBase]!,
            timestamps[queryBase + 3]!,
          );
          gpuSimulationStepMilliseconds.push(simulation.gpuMilliseconds!);
          gpuRenderMilliseconds.push(render.gpuMilliseconds!);
          gpuFrameMilliseconds.push(wholeFrame.gpuMilliseconds!);
        }
        return {
          feature: GPU_TIMESTAMP_FEATURE,
          supported: true,
          featureEnabled: true,
          gpuFrameMilliseconds,
          gpuSimulationStepMilliseconds,
          gpuRenderMilliseconds,
          reason: null,
          frameCount,
          timestampMapCount: 1,
        };
      } finally {
        if (mapped) {
          readback.unmap();
        }
      }
    } finally {
      this.measuring = false;
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.resources?.querySet.destroy();
    this.resources?.resolve.destroy();
    this.resources?.readback.destroy();
    this.resources = undefined;
  }

  private getResources(queryCount: number): GpuTimestampFrameProfileResources {
    if (this.resources?.queryCount === queryCount) {
      return this.resources;
    }
    this.resources?.querySet.destroy();
    this.resources?.resolve.destroy();
    this.resources?.readback.destroy();
    const byteLength = timestampByteLength(queryCount);
    const querySet = this.device.createQuerySet({
      label: `${this.label}-frame-profile-query-set`,
      type: "timestamp",
      count: queryCount,
    });
    const resolve = this.device.createBuffer({
      label: `${this.label}-frame-profile-resolve`,
      size: byteLength,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: `${this.label}-frame-profile-readback`,
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.resources = { querySet, resolve, readback, queryCount };
    return this.resources;
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("GpuTimestampFrameProfiler has been destroyed.");
    }
  }
}
