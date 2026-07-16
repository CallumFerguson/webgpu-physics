export interface WebGPUDeviceContext {
  readonly gpu: GPU;
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
}

export class WebGPUUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGPUUnavailableError";
  }
}

function describeAdapter(info: GPUAdapterInfo): string {
  const details = [
    info.description,
    info.vendor,
    info.architecture,
    info.device,
  ].filter((detail, index, all) => detail && all.indexOf(detail) === index);

  return details.join(" / ") || "unknown adapter";
}

export function assertHardwareWebGPUAdapter(info: GPUAdapterInfo): void {
  const description = describeAdapter(info);
  const isSwiftShader = /swiftshader/i.test(description);

  if (info.isFallbackAdapter || isSwiftShader) {
    throw new WebGPUUnavailableError(
      `A hardware WebGPU adapter is required, but Chrome selected a software ` +
        `adapter (${description}). SwiftShader and other fallback adapters are disabled.`,
    );
  }
}

/** Select optional features only when the chosen adapter advertises them. */
export function selectWebGPUDeviceFeatures(
  adapter: Pick<GPUAdapter, "features">,
): GPUFeatureName[] {
  return adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
}

export async function requestWebGPUDevice(): Promise<WebGPUDeviceContext> {
  const gpu = navigator.gpu;

  if (!gpu) {
    throw new WebGPUUnavailableError(
      "WebGPU is unavailable in this browser. Use a current browser with WebGPU enabled.",
    );
  }

  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });

  if (!adapter) {
    throw new WebGPUUnavailableError(
      "WebGPU is present, but no hardware GPU adapter could be created. " +
        "Software fallback adapters such as SwiftShader are disabled.",
    );
  }

  assertHardwareWebGPUAdapter(adapter.info);

  const device = await adapter.requestDevice({
    requiredFeatures: selectWebGPUDeviceFeatures(adapter),
  });

  return { gpu, adapter, device };
}
