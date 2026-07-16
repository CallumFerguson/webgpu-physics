import { describe, expect, it } from "vitest";

import {
  assertHardwareWebGPUAdapter,
  selectWebGPUDeviceFeatures,
  WebGPUUnavailableError,
} from "./device";

function adapterInfo(
  overrides: Partial<GPUAdapterInfo> = {},
): GPUAdapterInfo {
  return {
    vendor: "Example GPU Vendor",
    architecture: "Example Architecture",
    device: "Example Hardware GPU",
    description: "Example Hardware GPU",
    isFallbackAdapter: false,
    ...overrides,
  } as GPUAdapterInfo;
}

describe("assertHardwareWebGPUAdapter", () => {
  it("accepts a hardware adapter", () => {
    expect(() => assertHardwareWebGPUAdapter(adapterInfo())).not.toThrow();
  });

  it("rejects an adapter marked as a fallback", () => {
    expect(() =>
      assertHardwareWebGPUAdapter(
        adapterInfo({
          description: "Software fallback",
          isFallbackAdapter: true,
        }),
      ),
    ).toThrow(WebGPUUnavailableError);
  });

  it("rejects SwiftShader even when it is not marked as a fallback", () => {
    expect(() =>
      assertHardwareWebGPUAdapter(
        adapterInfo({ description: "Google SwiftShader Vulkan" }),
      ),
    ).toThrow(/SwiftShader and other fallback adapters are disabled/);
  });
});

describe("selectWebGPUDeviceFeatures", () => {
  const adapterWithFeatures = (...features: GPUFeatureName[]) =>
    ({
      features: new Set(features) as unknown as GPUSupportedFeatures,
    }) satisfies Pick<GPUAdapter, "features">;

  it("requests timestamp queries when the adapter advertises them", () => {
    expect(
      selectWebGPUDeviceFeatures(adapterWithFeatures("timestamp-query")),
    ).toEqual(["timestamp-query"]);
  });

  it("does not require timestamp queries from unsupported adapters", () => {
    expect(selectWebGPUDeviceFeatures(adapterWithFeatures())).toEqual([]);
  });
});
