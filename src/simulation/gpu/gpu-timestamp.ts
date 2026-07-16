export const GPU_TIMESTAMP_FEATURE = "timestamp-query" satisfies GPUFeatureName;

export interface GpuTimestampMeasurement {
  readonly feature: typeof GPU_TIMESTAMP_FEATURE;
  readonly supported: boolean;
  readonly featureEnabled: boolean;
  readonly gpuNanoseconds: number | null;
  readonly gpuMilliseconds: number | null;
  readonly reason: string | null;
}

interface GpuTimestampResources {
  readonly querySet: GPUQuerySet;
  readonly resolve: GPUBuffer;
  readonly readback: GPUBuffer;
}

const TIMESTAMP_QUERY_COUNT = 2;
const TIMESTAMP_BYTE_LENGTH = TIMESTAMP_QUERY_COUNT * BigUint64Array.BYTES_PER_ELEMENT;

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
    this.assertUsable();
    if (this.measuring) {
      throw new Error("A GPU timestamp measurement is already in progress.");
    }
    this.measuring = true;

    try {
      const encoder = this.device.createCommandEncoder({
        label: `${this.label}-timed-command-encoder`,
      });

      if (!this.supported) {
        encodeCommandStream(encoder);
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        return {
          feature: GPU_TIMESTAMP_FEATURE,
          supported: false,
          featureEnabled: false,
          gpuNanoseconds: null,
          gpuMilliseconds: null,
          reason: "The WebGPU device does not have the timestamp-query feature enabled.",
        };
      }

      const { querySet, resolve, readback } = this.getResources();

      // The end of an empty pass is the boundary immediately before the
      // simulation command stream. This avoids adding timestamps to any of
      // the production passes themselves.
      encoder
        .beginComputePass({
          label: `${this.label}-timestamp-start-pass`,
          timestampWrites: {
            querySet,
            endOfPassWriteIndex: 0,
          },
        })
        .end();

      encodeCommandStream(encoder);

      // The beginning of this empty pass is the boundary immediately after
      // the simulation command stream.
      encoder
        .beginComputePass({
          label: `${this.label}-timestamp-end-pass`,
          timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: 1,
          },
        })
        .end();
      encoder.resolveQuerySet(querySet, 0, TIMESTAMP_QUERY_COUNT, resolve, 0);
      encoder.copyBufferToBuffer(
        resolve,
        0,
        readback,
        0,
        TIMESTAMP_BYTE_LENGTH,
      );
      this.device.queue.submit([encoder.finish()]);

      let mapped = false;
      try {
        await readback.mapAsync(GPUMapMode.READ);
        mapped = true;
        const timestamps = new BigUint64Array(readback.getMappedRange());
        return decodeGpuTimestampMeasurement(timestamps[0]!, timestamps[1]!);
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

  private getResources(): GpuTimestampResources {
    if (this.resources) {
      return this.resources;
    }
    const querySet = this.device.createQuerySet({
      label: `${this.label}-timestamp-query-set`,
      type: "timestamp",
      count: TIMESTAMP_QUERY_COUNT,
    });
    const resolve = this.device.createBuffer({
      label: `${this.label}-timestamp-resolve`,
      size: TIMESTAMP_BYTE_LENGTH,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: `${this.label}-timestamp-readback`,
      size: TIMESTAMP_BYTE_LENGTH,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.resources = { querySet, resolve, readback };
    return this.resources;
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("GpuTimestampFrameTimer has been destroyed.");
    }
  }
}
