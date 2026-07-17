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

export const DEFAULT_LIVE_GPU_TIMESTAMP_BATCH_FRAME_COUNT = 60;
export const LIVE_GPU_TIMESTAMP_SLOT_COUNT = 3;

export interface GpuTimestampQueryResolve {
  readonly querySet: GPUQuerySet;
  readonly queryCount: number;
  readonly resolveBuffer: GPUBuffer;
  readonly readbackBuffer: GPUBuffer;
  readonly byteLength: number;
}

export interface GpuTimestampLiveFramePlan {
  readonly token: number;
  readonly writes: GpuTimestampFrameWrites;
  readonly resolveAfterRender: GpuTimestampQueryResolve | null;
}

export interface GpuTimestampLiveBatch {
  readonly feature: typeof GPU_TIMESTAMP_FEATURE;
  readonly frameCount: number;
  readonly gpuFrameMilliseconds: readonly number[];
  readonly gpuSimulationStepMilliseconds: readonly number[];
  readonly gpuRenderMilliseconds: readonly number[];
  readonly timestampMapCount: 1;
}

type GpuTimestampLiveSlotState = "idle" | "recording" | "mapping";

interface GpuTimestampLiveSlot {
  readonly querySet: GPUQuerySet;
  readonly resolve: GPUBuffer;
  readonly readback: GPUBuffer;
  state: GpuTimestampLiveSlotState;
  frameCount: number;
  generation: number;
}

/**
 * Non-blocking live GPU profiler. Timestamp queries are accumulated across a
 * batch of produced frames, resolved in the final render command buffer, and
 * mapped asynchronously. Three rotating slots let rendering continue while
 * prior readbacks are pending; if all are busy, frames remain uninstrumented.
 */
export class GpuTimestampLiveProfiler {
  private readonly slots: GpuTimestampLiveSlot[];
  private readonly completedBatches: GpuTimestampLiveBatch[] = [];
  private recordingSlotIndex: number | null = null;
  private pendingPlan: { readonly token: number; readonly slotIndex: number } | null =
    null;
  private nextToken = 1;
  private mapCount = 0;
  private skippedFrames = 0;
  private measurementGeneration = 0;
  private failureReason: string | null = null;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    readonly batchFrameCount = DEFAULT_LIVE_GPU_TIMESTAMP_BATCH_FRAME_COUNT,
  ) {
    if (
      !Number.isSafeInteger(batchFrameCount) ||
      batchFrameCount < 1 ||
      batchFrameCount > MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT
    ) {
      throw new RangeError(
        `batchFrameCount must be a positive safe integer no greater than ${MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT}.`,
      );
    }
    this.slots = this.featureSupported
      ? Array.from({ length: LIVE_GPU_TIMESTAMP_SLOT_COUNT }, (_unused, index) =>
          this.createSlot(index),
        )
      : [];
  }

  get featureSupported(): boolean {
    return this.device.features.has(GPU_TIMESTAMP_FEATURE);
  }

  get available(): boolean {
    return this.featureSupported && this.failureReason === null && !this.destroyed;
  }

  get reason(): string | null {
    if (!this.featureSupported) {
      return "The WebGPU device does not have the timestamp-query feature enabled.";
    }
    return this.failureReason;
  }

  get timestampMapCount(): number {
    return this.mapCount;
  }

  get skippedFrameCount(): number {
    return this.skippedFrames;
  }

  beginFrame(): GpuTimestampLiveFramePlan | null {
    this.assertUsable();
    if (!this.available) {
      return null;
    }
    if (this.pendingPlan) {
      throw new Error("The prior live GPU timestamp frame has not been finished.");
    }
    if (this.recordingSlotIndex === null) {
      const availableSlotIndex = this.slots.findIndex(
        (slot) => slot.state === "idle",
      );
      if (availableSlotIndex < 0) {
        this.skippedFrames += 1;
        return null;
      }
      const slot = this.slots[availableSlotIndex]!;
      slot.state = "recording";
      slot.frameCount = 0;
      slot.generation = this.measurementGeneration;
      this.recordingSlotIndex = availableSlotIndex;
    }

    const slotIndex = this.recordingSlotIndex;
    const slot = this.slots[slotIndex]!;
    const queryBase = slot.frameCount * FRAME_PROFILE_QUERIES_PER_FRAME;
    const token = this.nextToken;
    this.nextToken += 1;
    const isFinalFrame = slot.frameCount + 1 === this.batchFrameCount;
    const queryCount =
      this.batchFrameCount * FRAME_PROFILE_QUERIES_PER_FRAME;
    const plan: GpuTimestampLiveFramePlan = {
      token,
      writes: {
        simulation: {
          querySet: slot.querySet,
          startWriteIndex: queryBase,
          endWriteIndex: queryBase + 1,
        },
        render: {
          querySet: slot.querySet,
          startWriteIndex: queryBase + 2,
          endWriteIndex: queryBase + 3,
        },
      },
      resolveAfterRender: isFinalFrame
        ? {
            querySet: slot.querySet,
            queryCount,
            resolveBuffer: slot.resolve,
            readbackBuffer: slot.readback,
            byteLength: timestampByteLength(queryCount),
          }
        : null,
    };
    this.pendingPlan = { token, slotIndex };
    return plan;
  }

  finishFrame(plan: GpuTimestampLiveFramePlan): void {
    this.assertUsable();
    if (
      !this.pendingPlan ||
      this.pendingPlan.token !== plan.token ||
      this.pendingPlan.slotIndex !== this.recordingSlotIndex
    ) {
      throw new Error("The live GPU timestamp frame plan is stale or unknown.");
    }
    const slotIndex = this.pendingPlan.slotIndex;
    const slot = this.slots[slotIndex]!;
    this.pendingPlan = null;
    slot.frameCount += 1;
    if (slot.frameCount < this.batchFrameCount) {
      if (plan.resolveAfterRender !== null) {
        throw new Error("A partial live GPU timestamp batch cannot be resolved.");
      }
      return;
    }
    if (plan.resolveAfterRender === null) {
      throw new Error("The final live GPU timestamp frame must resolve its batch.");
    }
    slot.state = "mapping";
    this.recordingSlotIndex = null;
    this.mapCount += 1;
    void this.mapSlot(slot, slot.generation);
  }

  /**
   * Start a fresh visible measurement epoch without waiting for older maps.
   * Completed or in-flight batches from the prior epoch are discarded.
   */
  resetMeasurementWindow(): void {
    this.assertUsable();
    if (this.pendingPlan) {
      throw new Error(
        "Cannot reset live GPU timestamps while a frame plan is pending.",
      );
    }
    this.measurementGeneration += 1;
    this.completedBatches.length = 0;
    this.skippedFrames = 0;
    if (this.recordingSlotIndex !== null) {
      const slot = this.slots[this.recordingSlotIndex]!;
      slot.state = "idle";
      slot.frameCount = 0;
      slot.generation = this.measurementGeneration;
      this.recordingSlotIndex = null;
    }
  }

  /**
   * Permanently disable this profiler when a caller cannot complete a frame
   * plan. Partially written queries are deliberately abandoned and never
   * resolved or reused.
   */
  abortFrame(plan: GpuTimestampLiveFramePlan, reason: unknown): void {
    this.assertUsable();
    if (!this.pendingPlan || this.pendingPlan.token !== plan.token) {
      throw new Error("The live GPU timestamp frame plan is stale or unknown.");
    }
    const slot = this.slots[this.pendingPlan.slotIndex]!;
    this.pendingPlan = null;
    this.recordingSlotIndex = null;
    slot.state = "idle";
    slot.frameCount = 0;
    this.disable(
      reason instanceof Error
        ? `Live GPU timestamp frame aborted: ${reason.message}`
        : `Live GPU timestamp frame aborted: ${String(reason)}`,
    );
  }

  /** Disable live timing without destroying resources that may still be in use. */
  disable(reason: unknown): void {
    if (this.destroyed || this.failureReason !== null) {
      return;
    }
    this.failureReason =
      reason instanceof Error ? reason.message : String(reason);
    this.pendingPlan = null;
    this.recordingSlotIndex = null;
  }

  consumeCompletedBatches(): readonly GpuTimestampLiveBatch[] {
    if (this.completedBatches.length === 0) {
      return [];
    }
    return this.completedBatches.splice(0, this.completedBatches.length);
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.completedBatches.length = 0;
    this.pendingPlan = null;
    this.recordingSlotIndex = null;
    for (const slot of this.slots) {
      slot.querySet.destroy();
      slot.resolve.destroy();
      slot.readback.destroy();
    }
  }

  private createSlot(index: number): GpuTimestampLiveSlot {
    const queryCount =
      this.batchFrameCount * FRAME_PROFILE_QUERIES_PER_FRAME;
    const byteLength = timestampByteLength(queryCount);
    return {
      querySet: this.device.createQuerySet({
        label: `${this.label}-live-query-set-${index}`,
        type: "timestamp",
        count: queryCount,
      }),
      resolve: this.device.createBuffer({
        label: `${this.label}-live-resolve-${index}`,
        size: byteLength,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      }),
      readback: this.device.createBuffer({
        label: `${this.label}-live-readback-${index}`,
        size: byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      state: "idle",
      frameCount: 0,
      generation: this.measurementGeneration,
    };
  }

  private async mapSlot(
    slot: GpuTimestampLiveSlot,
    batchGeneration: number,
  ): Promise<void> {
    let mapped = false;
    try {
      await slot.readback.mapAsync(GPUMapMode.READ);
      mapped = true;
      const timestamps = new BigUint64Array(slot.readback.getMappedRange());
      const gpuFrameMilliseconds: number[] = [];
      const gpuSimulationStepMilliseconds: number[] = [];
      const gpuRenderMilliseconds: number[] = [];
      for (let frame = 0; frame < this.batchFrameCount; frame += 1) {
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
        gpuFrameMilliseconds.push(wholeFrame.gpuMilliseconds!);
        gpuSimulationStepMilliseconds.push(simulation.gpuMilliseconds!);
        gpuRenderMilliseconds.push(render.gpuMilliseconds!);
      }
      if (
        !this.destroyed &&
        batchGeneration === this.measurementGeneration
      ) {
        this.completedBatches.push({
          feature: GPU_TIMESTAMP_FEATURE,
          frameCount: this.batchFrameCount,
          gpuFrameMilliseconds,
          gpuSimulationStepMilliseconds,
          gpuRenderMilliseconds,
          timestampMapCount: 1,
        });
      }
    } catch (reason) {
      if (!this.destroyed) {
        this.disable(
          reason instanceof Error
            ? `Live GPU timestamp profiling failed: ${reason.message}`
            : `Live GPU timestamp profiling failed: ${String(reason)}`,
        );
      }
    } finally {
      if (mapped) {
        slot.readback.unmap();
      }
      if (!this.destroyed) {
        slot.state = "idle";
        slot.frameCount = 0;
      }
    }
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("GpuTimestampLiveProfiler has been destroyed.");
    }
  }
}
