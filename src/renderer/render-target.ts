export type RenderTargetKind = "canvas" | "texture";

export interface RenderTarget {
  readonly kind: RenderTargetKind;
  readonly format: GPUTextureFormat;
  acquireView(): GPUTextureView;
}

export interface OffscreenTextureOptions {
  readonly width: number;
  readonly height: number;
  readonly format: GPUTextureFormat;
  readonly label?: string;
}

export class CanvasRenderTarget implements RenderTarget {
  readonly kind = "canvas";

  private constructor(
    private readonly context: GPUCanvasContext,
    readonly format: GPUTextureFormat,
  ) {}

  static create(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    format: GPUTextureFormat,
  ): CanvasRenderTarget {
    const context = canvas.getContext("webgpu");

    if (!context) {
      throw new Error("The canvas could not create a WebGPU rendering context.");
    }

    context.configure({
      device,
      format,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    return new CanvasRenderTarget(context, format);
  }

  acquireView(): GPUTextureView {
    return this.context.getCurrentTexture().createView();
  }
}

export class OffscreenTextureRenderTarget implements RenderTarget {
  readonly kind = "texture";

  private constructor(
    readonly texture: GPUTexture,
    readonly format: GPUTextureFormat,
  ) {}

  static create(
    device: GPUDevice,
    options: OffscreenTextureOptions,
  ): OffscreenTextureRenderTarget {
    if (options.width < 1 || options.height < 1) {
      throw new RangeError("Offscreen texture dimensions must be positive.");
    }

    const texture = device.createTexture({
      label: options.label ?? "offscreen-render-target",
      size: {
        width: options.width,
        height: options.height,
        depthOrArrayLayers: 1,
      },
      format: options.format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.TEXTURE_BINDING,
    });

    return new OffscreenTextureRenderTarget(texture, options.format);
  }

  acquireView(): GPUTextureView {
    return this.texture.createView();
  }

  destroy(): void {
    this.texture.destroy();
  }
}

export function assertTargetFormat(
  pipelineFormat: GPUTextureFormat,
  target: RenderTarget,
): void {
  if (pipelineFormat !== target.format) {
    throw new Error(
      `Render target format ${target.format} does not match pipeline format ${pipelineFormat}.`,
    );
  }
}
