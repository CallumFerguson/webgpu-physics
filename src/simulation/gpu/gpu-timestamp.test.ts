import { describe, expect, it } from "vitest";

import {
  decodeGpuTimestampMeasurement,
  GpuTimestampFrameTimer,
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
    timer.destroy();
  });
});
