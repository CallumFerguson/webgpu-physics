import { assertTargetFormat, type RenderTarget } from "./render-target";
import { triangleShader } from "./triangle-shader";

export const TRIANGLE_VERTEX_COUNT = 3;

export class TriangleRenderer {
  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    private readonly format: GPUTextureFormat,
  ) {}

  static create(
    device: GPUDevice,
    format: GPUTextureFormat,
  ): TriangleRenderer {
    const shader = device.createShaderModule({
      label: "triangle-shader",
      code: triangleShader,
    });
    const pipeline = device.createRenderPipeline({
      label: "triangle-pipeline",
      layout: "auto",
      vertex: {
        module: shader,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: shader,
        entryPoint: "fragmentMain",
        targets: [{ format }],
      },
      primitive: {
        topology: "triangle-list",
      },
      multisample: {
        count: 1,
      },
    });

    return new TriangleRenderer(device, pipeline, format);
  }

  async render(target: RenderTarget): Promise<void> {
    assertTargetFormat(this.format, target);

    const encoder = this.device.createCommandEncoder({
      label: `triangle-${target.kind}-command-encoder`,
    });
    const pass = encoder.beginRenderPass({
      label: "triangle-render-pass",
      colorAttachments: [
        {
          view: target.acquireView(),
          clearValue: { r: 0.025, g: 0.055, b: 0.105, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline);
    pass.draw(TRIANGLE_VERTEX_COUNT);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }
}
