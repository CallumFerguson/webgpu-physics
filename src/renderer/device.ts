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

export async function requestWebGPUDevice(): Promise<WebGPUDeviceContext> {
  const gpu = navigator.gpu;

  if (!gpu) {
    throw new WebGPUUnavailableError(
      "WebGPU is unavailable in this browser. Use a current browser with WebGPU enabled.",
    );
  }

  const adapter = await gpu.requestAdapter({ powerPreference: "low-power" });

  if (!adapter) {
    throw new WebGPUUnavailableError(
      "WebGPU is present, but no compatible GPU adapter could be created.",
    );
  }

  const device = await adapter.requestDevice();

  return { gpu, adapter, device };
}
