import { describe, expect, it } from "vitest";

import {
  decodeGpuTimestampMeasurement,
  GpuTimestampFrameProfiler,
  GpuTimestampFrameTimer,
  GpuTimestampLiveProfiler,
  LIVE_GPU_TIMESTAMP_SLOT_COUNT,
  MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT,
  MAX_GPU_TIMESTAMP_QUERY_COUNT,
  MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT,
} from "./gpu-timestamp";

describe("GPU timestamp measurement", () => {
  it("converts WebGPU nanoseconds to finite nonnegative milliseconds", () => {
    expect(decodeGpuTimestampMeasurement(10_000n, 2_510_000n)).toEqual({
      feature: "timestamp-query",
      supported: true,
      featureEnabled: true,
      gpuNanoseconds: 2_500_000,
      gpuMilliseconds: 2.5,
      reason: null,
    });
    expect(decodeGpuTimestampMeasurement(42n, 42n).gpuMilliseconds).toBe(0);
  });

  it("rejects reversed intervals rather than reporting misleading timing", () => {
    expect(() => decodeGpuTimestampMeasurement(2n, 1n)).toThrow(
      /ended before it began/,
    );
  });

  it("guards concurrent measurements on an unsupported device", async () => {
    let finishSubmittedWork!: () => void;
    const submittedWork = new Promise<void>((resolve) => {
      finishSubmittedWork = resolve;
    });
    const finish = {} as GPUCommandBuffer;
    const device = {
      features: new Set<GPUFeatureName>(),
      createCommandEncoder: () => ({ finish: () => finish }),
      queue: {
        submit: () => undefined,
        onSubmittedWorkDone: () => submittedWork,
      },
    } as unknown as GPUDevice;
    const timer = new GpuTimestampFrameTimer(device, "test");

    const first = timer.measure(() => undefined);
    await expect(timer.measure(() => undefined)).rejects.toThrow(
      /already in progress/,
    );
    finishSubmittedWork();
    await expect(first).resolves.toMatchObject({
      supported: false,
      featureEnabled: false,
      gpuMilliseconds: null,
    });
    await expect(
      timer.measureFrames(2, () => undefined),
    ).resolves.toMatchObject({
      supported: false,
      frameCount: 2,
      timestampMapCount: 0,
    });
    timer.destroy();
  });

  it("profiles simulation, render, and whole-frame intervals with one map", async () => {
    const previousBufferUsage = globalThis.GPUBufferUsage;
    const previousMapMode = globalThis.GPUMapMode;
    Object.assign(globalThis, {
      GPUBufferUsage: {
        QUERY_RESOLVE: 1,
        COPY_SRC: 2,
        COPY_DST: 4,
        MAP_READ: 8,
      },
      GPUMapMode: { READ: 1 },
    });
    const timestamps = new BigUint64Array([
      100n,
      200n,
      220n,
      260n,
      300n,
      500n,
      520n,
      570n,
    ]);
    const queryCounts: number[] = [];
    const writes: number[][] = [];
    let mapCount = 0;
    const readback = {
      mapAsync: async () => {
        mapCount += 1;
      },
      getMappedRange: () => timestamps.buffer,
      unmap: () => undefined,
      destroy: () => undefined,
    } as unknown as GPUBuffer;
    let bufferIndex = 0;
    const device = {
      features: new Set<GPUFeatureName>(["timestamp-query"]),
      createQuerySet: (descriptor: GPUQuerySetDescriptor) => {
        queryCounts.push(descriptor.count);
        return { destroy: () => undefined } as unknown as GPUQuerySet;
      },
      createBuffer: () => {
        bufferIndex += 1;
        return bufferIndex % 2 === 0
          ? readback
          : ({ destroy: () => undefined } as unknown as GPUBuffer);
      },
      createCommandEncoder: () =>
        ({
          resolveQuerySet: () => undefined,
          copyBufferToBuffer: () => undefined,
          finish: () => ({}) as GPUCommandBuffer,
        }) as unknown as GPUCommandEncoder,
      queue: { submit: () => undefined },
    } as unknown as GPUDevice;

    try {
      const profiler = new GpuTimestampFrameProfiler(device, "test-frame");
      const profile = await profiler.measureFrames(2, async (frameWrites) => {
        expect(frameWrites).not.toBeNull();
        writes.push([
          frameWrites!.simulation.startWriteIndex,
          frameWrites!.simulation.endWriteIndex,
          frameWrites!.render.startWriteIndex,
          frameWrites!.render.endWriteIndex,
        ]);
      });

      expect(queryCounts).toEqual([8]);
      expect(writes).toEqual([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
      ]);
      expect(mapCount).toBe(1);
      expect(profile.timestampMapCount).toBe(1);
      expect(profile.gpuSimulationStepMilliseconds).toEqual([0.0001, 0.0002]);
      expect(profile.gpuRenderMilliseconds).toEqual([0.00004, 0.00005]);
      expect(profile.gpuFrameMilliseconds).toEqual([0.00016, 0.00027]);
      profiler.destroy();
    } finally {
      Object.assign(globalThis, {
        GPUBufferUsage: previousBufferUsage,
        GPUMapMode: previousMapMode,
      });
    }
  });

  it("reports unsupported frame profiling without queries and guards concurrency", async () => {
    let releaseFirstFrame!: () => void;
    const firstFrameBlocked = new Promise<void>((resolve) => {
      releaseFirstFrame = resolve;
    });
    let submittedFrames = 0;
    const device = {
      features: new Set<GPUFeatureName>(),
      createQuerySet: () => {
        throw new Error("unsupported profiling must stay allocation-free");
      },
    } as unknown as GPUDevice;
    const profiler = new GpuTimestampFrameProfiler(device, "unsupported-frame");

    const first = profiler.measureFrames(2, async (writes) => {
      expect(writes).toBeNull();
      submittedFrames += 1;
      if (submittedFrames === 1) {
        await firstFrameBlocked;
      }
    });
    await Promise.resolve();
    await expect(
      profiler.measureFrames(1, async () => undefined),
    ).rejects.toThrow(/already in progress/);
    releaseFirstFrame();
    await expect(first).resolves.toMatchObject({
      supported: false,
      timestampMapCount: 0,
      gpuFrameMilliseconds: null,
      frameCount: 2,
    });
    expect(submittedFrames).toBe(2);
    profiler.destroy();
  });

  it("fails closed at the WebGPU query-set frame limits", async () => {
    expect(MAX_GPU_TIMESTAMP_QUERY_COUNT).toBe(8_192);
    expect(MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT).toBe(4_096);
    expect(MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT).toBe(2_048);

    let timedFrames = 0;
    const finish = {} as GPUCommandBuffer;
    const device = {
      features: new Set<GPUFeatureName>(),
      createCommandEncoder: () => ({ finish: () => finish }),
      queue: {
        submit: () => undefined,
        onSubmittedWorkDone: async () => undefined,
      },
    } as unknown as GPUDevice;
    const timer = new GpuTimestampFrameTimer(device, "limit-timer");
    await expect(
      timer.measureFrames(MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT, () => {
        timedFrames += 1;
      }),
    ).resolves.toMatchObject({
      supported: false,
      frameCount: MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT,
    });
    expect(timedFrames).toBe(MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT);
    await expect(
      timer.measureFrames(
        MAX_GPU_TIMESTAMP_TIMER_FRAME_COUNT + 1,
        () => undefined,
      ),
    ).rejects.toThrow(/4096/);

    let profiledFrames = 0;
    const profiler = new GpuTimestampFrameProfiler(device, "limit-profiler");
    await expect(
      profiler.measureFrames(
        MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT,
        async () => {
          profiledFrames += 1;
        },
      ),
    ).resolves.toMatchObject({
      supported: false,
      frameCount: MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT,
    });
    expect(profiledFrames).toBe(MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT);
    await expect(
      profiler.measureFrames(
        MAX_GPU_TIMESTAMP_PROFILE_FRAME_COUNT + 1,
        async () => undefined,
      ),
    ).rejects.toThrow(/2048/);

    timer.destroy();
    profiler.destroy();
  });

  it("keeps unsupported live profiling allocation-free", () => {
    let allocations = 0;
    const device = {
      features: new Set<GPUFeatureName>(),
      createQuerySet: () => {
        allocations += 1;
        return {} as GPUQuerySet;
      },
      createBuffer: () => {
        allocations += 1;
        return {} as GPUBuffer;
      },
    } as unknown as GPUDevice;

    const profiler = new GpuTimestampLiveProfiler(device, "live-unsupported", 2);
    expect(profiler.featureSupported).toBe(false);
    expect(profiler.available).toBe(false);
    expect(profiler.reason).toMatch(/does not have/);
    expect(profiler.beginFrame()).toBeNull();
    expect(profiler.timestampMapCount).toBe(0);
    expect(allocations).toBe(0);
    profiler.destroy();
  });

  it("batches live frame, simulation, and render timestamps into one async map", async () => {
    const previousBufferUsage = globalThis.GPUBufferUsage;
    const previousMapMode = globalThis.GPUMapMode;
    Object.assign(globalThis, {
      GPUBufferUsage: {
        QUERY_RESOLVE: 1,
        COPY_SRC: 2,
        COPY_DST: 4,
        MAP_READ: 8,
      },
      GPUMapMode: { READ: 1 },
    });
    const timestamps = new BigUint64Array([
      100n,
      200n,
      220n,
      260n,
      300n,
      500n,
      520n,
      570n,
    ]);
    let mapCount = 0;
    let querySetCount = 0;
    const destroyed: string[] = [];
    const device = {
      features: new Set<GPUFeatureName>(["timestamp-query"]),
      createQuerySet: () => {
        querySetCount += 1;
        return {
          destroy: () => destroyed.push("query"),
        } as unknown as GPUQuerySet;
      },
      createBuffer: (descriptor: GPUBufferDescriptor) =>
        descriptor.usage & GPUBufferUsage.MAP_READ
          ? ({
              mapAsync: async () => {
                mapCount += 1;
              },
              getMappedRange: () => timestamps.buffer,
              unmap: () => undefined,
              destroy: () => destroyed.push("readback"),
            } as unknown as GPUBuffer)
          : ({
              destroy: () => destroyed.push("resolve"),
            } as unknown as GPUBuffer),
    } as unknown as GPUDevice;

    try {
      const profiler = new GpuTimestampLiveProfiler(device, "live", 2);
      expect(querySetCount).toBe(LIVE_GPU_TIMESTAMP_SLOT_COUNT);
      const first = profiler.beginFrame()!;
      expect([
        first.writes.simulation.startWriteIndex,
        first.writes.simulation.endWriteIndex,
        first.writes.render.startWriteIndex,
        first.writes.render.endWriteIndex,
      ]).toEqual([0, 1, 2, 3]);
      expect(first.resolveAfterRender).toBeNull();
      profiler.finishFrame(first);

      const second = profiler.beginFrame()!;
      expect([
        second.writes.simulation.startWriteIndex,
        second.writes.simulation.endWriteIndex,
        second.writes.render.startWriteIndex,
        second.writes.render.endWriteIndex,
      ]).toEqual([4, 5, 6, 7]);
      expect(second.resolveAfterRender).toMatchObject({
        queryCount: 8,
        byteLength: 64,
      });
      profiler.finishFrame(second);
      expect(mapCount).toBe(1);
      expect(profiler.timestampMapCount).toBe(1);

      await Promise.resolve();
      await Promise.resolve();
      const batches = profiler.consumeCompletedBatches();
      expect(batches).toHaveLength(1);
      expect(batches[0]!.gpuSimulationStepMilliseconds).toEqual([
        0.0001,
        0.0002,
      ]);
      expect(batches[0]!.gpuRenderMilliseconds).toEqual([0.00004, 0.00005]);
      expect(batches[0]!.gpuFrameMilliseconds).toEqual([0.00016, 0.00027]);
      expect(profiler.consumeCompletedBatches()).toEqual([]);
      profiler.destroy();
      expect(destroyed).toHaveLength(LIVE_GPU_TIMESTAMP_SLOT_COUNT * 3);
    } finally {
      Object.assign(globalThis, {
        GPUBufferUsage: previousBufferUsage,
        GPUMapMode: previousMapMode,
      });
    }
  });

  it("rotates live query slots and skips instrumentation instead of stalling", async () => {
    const previousBufferUsage = globalThis.GPUBufferUsage;
    const previousMapMode = globalThis.GPUMapMode;
    Object.assign(globalThis, {
      GPUBufferUsage: {
        QUERY_RESOLVE: 1,
        COPY_SRC: 2,
        COPY_DST: 4,
        MAP_READ: 8,
      },
      GPUMapMode: { READ: 1 },
    });
    const releases: Array<() => void> = [];
    const timestamps = new BigUint64Array([100n, 200n, 220n, 260n]);
    const device = {
      features: new Set<GPUFeatureName>(["timestamp-query"]),
      createQuerySet: () =>
        ({ destroy: () => undefined }) as unknown as GPUQuerySet,
      createBuffer: (descriptor: GPUBufferDescriptor) =>
        descriptor.usage & GPUBufferUsage.MAP_READ
          ? ({
              mapAsync: () =>
                new Promise<void>((resolve) => {
                  releases.push(resolve);
                }),
              getMappedRange: () => timestamps.buffer,
              unmap: () => undefined,
              destroy: () => undefined,
            } as unknown as GPUBuffer)
          : ({ destroy: () => undefined }) as unknown as GPUBuffer,
    } as unknown as GPUDevice;

    try {
      const profiler = new GpuTimestampLiveProfiler(device, "live-slots", 1);
      for (let slot = 0; slot < LIVE_GPU_TIMESTAMP_SLOT_COUNT; slot += 1) {
        const plan = profiler.beginFrame();
        expect(plan).not.toBeNull();
        profiler.finishFrame(plan!);
      }
      expect(releases).toHaveLength(LIVE_GPU_TIMESTAMP_SLOT_COUNT);
      expect(profiler.beginFrame()).toBeNull();
      expect(profiler.skippedFrameCount).toBe(1);
      profiler.resetMeasurementWindow();
      expect(profiler.skippedFrameCount).toBe(0);

      releases[0]!();
      await Promise.resolve();
      await Promise.resolve();
      expect(profiler.consumeCompletedBatches()).toEqual([]);
      const next = profiler.beginFrame();
      expect(next).not.toBeNull();
      profiler.finishFrame(next!);
      expect(profiler.timestampMapCount).toBe(
        LIVE_GPU_TIMESTAMP_SLOT_COUNT + 1,
      );
      profiler.destroy();
      for (const release of releases.slice(1)) {
        release();
      }
    } finally {
      Object.assign(globalThis, {
        GPUBufferUsage: previousBufferUsage,
        GPUMapMode: previousMapMode,
      });
    }
  });

  it("fails closed when a caller aborts a partially encoded live frame", () => {
    const previousBufferUsage = globalThis.GPUBufferUsage;
    Object.assign(globalThis, {
      GPUBufferUsage: {
        QUERY_RESOLVE: 1,
        COPY_SRC: 2,
        COPY_DST: 4,
        MAP_READ: 8,
      },
    });
    const device = {
      features: new Set<GPUFeatureName>(["timestamp-query"]),
      createQuerySet: () =>
        ({ destroy: () => undefined }) as unknown as GPUQuerySet,
      createBuffer: () =>
        ({ destroy: () => undefined }) as unknown as GPUBuffer,
    } as unknown as GPUDevice;

    try {
      const profiler = new GpuTimestampLiveProfiler(device, "live-abort", 2);
      const plan = profiler.beginFrame()!;
      profiler.abortFrame(plan, new Error("render submission failed"));

      expect(profiler.available).toBe(false);
      expect(profiler.reason).toMatch(/frame aborted.*render submission failed/i);
      expect(profiler.beginFrame()).toBeNull();
      expect(() => profiler.finishFrame(plan)).toThrow(/stale or unknown/);
      profiler.destroy();
    } finally {
      Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage });
    }
  });
});
