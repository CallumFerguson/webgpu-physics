import { sceneShader } from "./scene-shader";
import type { GpuTimestampQueryResolve } from "../simulation/gpu";

export interface SceneRenderInput {
  readonly restPositions: Float32Array;
  readonly vertexColors: Float32Array;
  readonly surfaceTriangles: Uint32Array;
  readonly surfaceEdges: Uint32Array;
  readonly floorHeight: number;
  /** Defaults to true. Force-free fixtures can suppress the decorative floor. */
  readonly showFloor?: boolean;
}

export interface SceneCamera {
  readonly eye: readonly [number, number, number];
  readonly target: readonly [number, number, number];
  readonly floorCenter: readonly [number, number];
  readonly floorScale: number;
}

export interface SceneRenderTimestampWrites {
  readonly querySet: GPUQuerySet;
  readonly startWriteIndex: number;
  readonly endWriteIndex: number;
}

export function encodeSceneTimestampResolve(
  encoder: GPUCommandEncoder,
  resolve: GpuTimestampQueryResolve,
): void {
  if (
    !Number.isSafeInteger(resolve.queryCount) ||
    resolve.queryCount < 1 ||
    resolve.byteLength !==
      resolve.queryCount * BigUint64Array.BYTES_PER_ELEMENT
  ) {
    throw new RangeError(
      "Scene timestamp resolves require a positive query count and exact byte length.",
    );
  }
  encoder.resolveQuerySet(
    resolve.querySet,
    0,
    resolve.queryCount,
    resolve.resolveBuffer,
    0,
  );
  encoder.copyBufferToBuffer(
    resolve.resolveBuffer,
    0,
    resolve.readbackBuffer,
    0,
    resolve.byteLength,
  );
}

function normalize(vector: readonly number[]): [number, number, number] {
  const length = Math.hypot(vector[0] ?? 0, vector[1] ?? 0, vector[2] ?? 0);
  if (length < 1e-12) {
    return [0, 1, 0];
  }
  return [
    (vector[0] ?? 0) / length,
    (vector[1] ?? 0) / length,
    (vector[2] ?? 0) / length,
  ];
}

function cross(
  left: readonly number[],
  right: readonly number[],
): [number, number, number] {
  return [
    (left[1] ?? 0) * (right[2] ?? 0) - (left[2] ?? 0) * (right[1] ?? 0),
    (left[2] ?? 0) * (right[0] ?? 0) - (left[0] ?? 0) * (right[2] ?? 0),
    (left[0] ?? 0) * (right[1] ?? 0) - (left[1] ?? 0) * (right[0] ?? 0),
  ];
}

function multiply4x4(left: Float32Array, right: Float32Array): Float32Array {
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let inner = 0; inner < 4; inner += 1) {
        value +=
          (left[inner * 4 + row] ?? 0) *
          (right[column * 4 + inner] ?? 0);
      }
      result[column * 4 + row] = value;
    }
  }
  return result;
}

function viewMatrix(
  eye: readonly [number, number, number],
  target: readonly [number, number, number],
): Float32Array {
  const forward = normalize([
    eye[0] - target[0],
    eye[1] - target[1],
    eye[2] - target[2],
  ]);
  const right = normalize(cross([0, 1, 0], forward));
  const up = cross(forward, right);
  return new Float32Array([
    right[0], up[0], forward[0], 0,
    right[1], up[1], forward[1], 0,
    right[2], up[2], forward[2], 0,
    -right[0] * eye[0] - right[1] * eye[1] - right[2] * eye[2],
    -up[0] * eye[0] - up[1] * eye[1] - up[2] * eye[2],
    -forward[0] * eye[0] - forward[1] * eye[1] - forward[2] * eye[2],
    1,
  ]);
}

function perspectiveMatrix(
  fieldOfView: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const focal = 1 / Math.tan(fieldOfView / 2);
  const range = far / (near - far);
  return new Float32Array([
    focal / aspect, 0, 0, 0,
    0, focal, 0, 0,
    0, 0, range, -1,
    0, 0, range * near, 0,
  ]);
}

function createBuffer(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const size = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ label, size, usage, mappedAtCreation: true });
  const destination = new Uint8Array(buffer.getMappedRange());
  destination.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

export function inferSceneCamera(input: SceneRenderInput): SceneCamera {
  const vertexCount = input.restPositions.length / 4;
  const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = input.restPositions[vertex * 4 + axis] ?? 0;
      minimum[axis] = Math.min(minimum[axis] ?? value, value);
      maximum[axis] = Math.max(maximum[axis] ?? value, value);
    }
  }
  const center: [number, number, number] = [
    ((minimum[0] ?? 0) + (maximum[0] ?? 0)) / 2,
    ((minimum[1] ?? 0) + (maximum[1] ?? 0)) / 2,
    ((minimum[2] ?? 0) + (maximum[2] ?? 0)) / 2,
  ];
  const extent = Math.max(
    (maximum[0] ?? 1) - (minimum[0] ?? 0),
    (maximum[1] ?? 1) - input.floorHeight,
    (maximum[2] ?? 1) - (minimum[2] ?? 0),
    0.4,
  );
  return {
    eye: [center[0] + extent * 1.65, center[1] + extent * 1.05, center[2] + extent * 2.2],
    target: [center[0], Math.max(input.floorHeight + extent * 0.35, center[1]), center[2]],
    floorCenter: [center[0], center[2]],
    floorScale: extent * 2.4,
  };
}

export class SceneRenderer {
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly renderVertexBuffer: GPUBuffer;
  private readonly triangleIndexBuffer: GPUBuffer;
  private readonly edgeIndexBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly surfacePipeline: GPURenderPipeline;
  private readonly liveEdgePipeline: GPURenderPipeline;
  private readonly restEdgePipeline: GPURenderPipeline;
  private readonly floorPipeline: GPURenderPipeline;
  private readonly positionOffsetVertices: number;
  private depthTexture: GPUTexture;
  private frame = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
    format: GPUTextureFormat,
    private readonly input: SceneRenderInput,
    private camera: SceneCamera,
    positionBuffer: GPUBuffer,
    positionOffsetVertices: number,
  ) {
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("The canvas could not create a WebGPU rendering context.");
    }
    this.context = context;
    this.positionOffsetVertices = positionOffsetVertices;
    context.configure({ device, format, alphaMode: "opaque" });

    const renderVertices = new Float32Array((input.restPositions.length / 4) * 8);
    for (let vertex = 0; vertex < input.restPositions.length / 4; vertex += 1) {
      renderVertices.set(input.restPositions.subarray(vertex * 4, vertex * 4 + 4), vertex * 8);
      renderVertices.set(input.vertexColors.subarray(vertex * 4, vertex * 4 + 4), vertex * 8 + 4);
    }
    this.renderVertexBuffer = createBuffer(
      device,
      "jgs2-render-vertices",
      renderVertices,
      GPUBufferUsage.STORAGE,
    );
    this.triangleIndexBuffer = createBuffer(
      device,
      "jgs2-surface-triangles",
      input.surfaceTriangles,
      GPUBufferUsage.INDEX,
    );
    this.edgeIndexBuffer = createBuffer(
      device,
      "jgs2-surface-edges",
      input.surfaceEdges,
      GPUBufferUsage.INDEX,
    );
    this.uniformBuffer = device.createBuffer({
      label: "jgs2-render-uniforms",
      size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "jgs2-render-bind-group-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });
    this.bindGroup = device.createBindGroup({
      label: "jgs2-render-bind-group",
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: this.renderVertexBuffer } },
      ],
    });
    const module = device.createShaderModule({ label: "jgs2-scene-shader", code: sceneShader });
    const pipelineLayout = device.createPipelineLayout({
      label: "jgs2-render-pipeline-layout",
      bindGroupLayouts: [bindGroupLayout],
    });
    const commonVertex = { module, entryPoint: "liveVertex" } satisfies GPUVertexState;
    const depthStencil = {
      format: "depth24plus" as GPUTextureFormat,
      depthWriteEnabled: true,
      depthCompare: "less" as GPUCompareFunction,
    };
    this.surfacePipeline = device.createRenderPipeline({
      label: "jgs2-surface-pipeline",
      layout: pipelineLayout,
      vertex: commonVertex,
      fragment: { module, entryPoint: "surfaceFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil,
    });
    this.liveEdgePipeline = device.createRenderPipeline({
      label: "jgs2-live-edge-pipeline",
      layout: pipelineLayout,
      vertex: commonVertex,
      fragment: { module, entryPoint: "lineFragment", targets: [{ format }] },
      primitive: { topology: "line-list" },
      depthStencil: { ...depthStencil, depthWriteEnabled: false, depthCompare: "less-equal" },
    });
    const blend: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    this.restEdgePipeline = device.createRenderPipeline({
      label: "jgs2-rest-edge-pipeline",
      layout: pipelineLayout,
      vertex: { module, entryPoint: "restVertex" },
      fragment: { module, entryPoint: "lineFragment", targets: [{ format, blend }] },
      primitive: { topology: "line-list" },
      depthStencil: { ...depthStencil, depthWriteEnabled: false, depthCompare: "always" },
    });
    this.floorPipeline = device.createRenderPipeline({
      label: "jgs2-floor-pipeline",
      layout: pipelineLayout,
      vertex: { module, entryPoint: "floorVertex" },
      fragment: { module, entryPoint: "floorFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil,
    });
    this.depthTexture = this.createDepthTexture();
    this.writeUniforms(positionOffsetVertices);
  }

  static create(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    format: GPUTextureFormat,
    input: SceneRenderInput,
    positionBuffer: GPUBuffer,
    positionOffsetVertices = 0,
    camera: SceneCamera = inferSceneCamera(input),
  ): SceneRenderer {
    return new SceneRenderer(device, canvas, format, input, camera, positionBuffer, positionOffsetVertices);
  }

  private createDepthTexture(): GPUTexture {
    return this.device.createTexture({
      label: "jgs2-depth",
      size: { width: this.canvas.width, height: this.canvas.height },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private writeUniforms(positionOffsetVertices: number): void {
    const projection = perspectiveMatrix(Math.PI / 4.2, this.canvas.width / this.canvas.height, 0.01, 100);
    const viewProjection = multiply4x4(projection, viewMatrix(this.camera.eye, this.camera.target));
    const values = new Float32Array(28);
    values.set(viewProjection, 0);
    values.set([...this.camera.eye, this.frame], 16);
    values.set([
      this.input.floorHeight,
      this.camera.floorCenter[0],
      this.camera.floorCenter[1],
      this.camera.floorScale,
    ], 20);
    const uintValues = new Uint32Array(values.buffer);
    uintValues[24] = positionOffsetVertices;
    uintValues[25] = this.input.restPositions.length / 4;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, values);
  }

  setCamera(camera: SceneCamera): void {
    const values = [
      ...camera.eye,
      ...camera.target,
      ...camera.floorCenter,
      camera.floorScale,
    ];
    if (!values.every(Number.isFinite) || !(camera.floorScale > 0)) {
      throw new RangeError("Scene camera values must be finite with positive floor scale.");
    }
    this.camera = camera;
  }

  render(
    frame = this.frame,
    timestampWrites?: SceneRenderTimestampWrites,
    timestampResolve?: GpuTimestampQueryResolve,
  ): void {
    this.frame = frame;
    this.writeUniforms(this.positionOffsetVertices);
    const encoder = this.device.createCommandEncoder({ label: "jgs2-render-encoder" });
    const pass = encoder.beginRenderPass({
      label: "jgs2-render-pass",
      ...(timestampWrites
        ? {
            timestampWrites: {
              querySet: timestampWrites.querySet,
              beginningOfPassWriteIndex: timestampWrites.startWriteIndex,
              endOfPassWriteIndex: timestampWrites.endWriteIndex,
            },
          }
        : {}),
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.012, g: 0.024, b: 0.045, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setBindGroup(0, this.bindGroup);
    if (this.input.showFloor !== false) {
      pass.setPipeline(this.floorPipeline);
      pass.draw(6);
    }
    pass.setIndexBuffer(this.triangleIndexBuffer, "uint32");
    pass.setPipeline(this.surfacePipeline);
    pass.drawIndexed(this.input.surfaceTriangles.length);
    pass.setIndexBuffer(this.edgeIndexBuffer, "uint32");
    pass.setPipeline(this.liveEdgePipeline);
    pass.drawIndexed(this.input.surfaceEdges.length);
    pass.setPipeline(this.restEdgePipeline);
    pass.drawIndexed(this.input.surfaceEdges.length);
    pass.end();
    if (timestampResolve) {
      if (!timestampWrites || timestampResolve.querySet !== timestampWrites.querySet) {
        throw new Error(
          "A render timestamp resolve must use the same query set as the render pass.",
        );
      }
      encodeSceneTimestampResolve(encoder, timestampResolve);
    }
    this.device.queue.submit([encoder.finish()]);
  }

  async renderAndWait(frame = this.frame): Promise<void> {
    this.render(frame);
    await this.device.queue.onSubmittedWorkDone();
  }

  destroy(): void {
    this.uniformBuffer.destroy();
    this.renderVertexBuffer.destroy();
    this.triangleIndexBuffer.destroy();
    this.edgeIndexBuffer.destroy();
    this.depthTexture.destroy();
  }
}
