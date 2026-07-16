import { describe, expect, it } from "vitest";

import {
  assertHardwareWebGPUAdapter,
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
