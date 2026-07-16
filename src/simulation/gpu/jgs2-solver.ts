import {
  JGS2_UNIFORM_BYTES,
  computeJGS2DynamicOffsets,
  inferJGS2BodyCount,
  jgs2TimestepsMatch,
  normalizeOddIterationCount,
  packJGS2Cubature,
  packJGS2InitialDynamic,
  packJGS2TetStatic,
  packJGS2VertexStatic,
  validateJGS2ContactParameters,
  validateJGS2GpuInput,
  type JGS2DynamicOffsets,
  type JGS2GpuInput,
} from "./layout";
import { JGS2_WORKGROUP_SIZE, jgs2Shader } from "./jgs2-shader";

export interface JGS2StepSettings {
  readonly timestep: number;
  readonly gravity: readonly [number, number, number];
  readonly iterations: number;
  readonly floorHeight: number;
  readonly floorStiffness: number;
  readonly velocityDamping: number;
  readonly regularization: number;
  readonly rotationEpsilon: number;
  readonly maxStep: number;
  /** Exponential x/z damping rate in s^-1 for grounded vertices. */
  readonly contactTangentialDamping: number;
  /** Distance above the floor at which tangential damping becomes active. */
  readonly contactMargin: number;
}

export interface JGS2PositionBufferView {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
  readonly stride: 16;
}

export const DEFAULT_JGS2_STEP_SETTINGS: JGS2StepSettings = {
  timestep: 1 / 60,
  gravity: [0, -9.81, 0],
  iterations: 9,
  floorHeight: 0,
  floorStiffness: 20_000,
  velocityDamping: 0.999,
  regularization: 1e-6,
  rotationEpsilon: 1e-7,
  maxStep: 0.1,
  contactTangentialDamping: 8,
  contactMargin: 0.01,
};

interface JGS2GpuBuffers {
  readonly dynamic: GPUBuffer;
  readonly vertices: GPUBuffer;
  readonly tets: GPUBuffer;
  readonly stiffness: GPUBuffer;
  readonly adjacency: GPUBuffer;
  readonly cubature: GPUBuffer;
}

interface JGS2Pipelines {
  readonly predict: GPUComputePipeline;
  readonly tetPolarRotation: GPUComputePipeline;
  readonly vertexPolarRotation: GPUComputePipeline;
  readonly solve: GPUComputePipeline;
  readonly bodyHorizontalCorrection: GPUComputePipeline;
  readonly applyBodyHorizontalCorrection: GPUComputePipeline;
  readonly finalize: GPUComputePipeline;
}

interface JGS2UniformState {
  readonly base: GPUBuffer;
  readonly fromBToA: GPUBuffer;
  readonly fromAToB: GPUBuffer;
  readonly baseBindGroup: GPUBindGroup;
  readonly fromBToABindGroup: GPUBindGroup;
  readonly fromAToBBindGroup: GPUBindGroup;
}

function alignTo4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

function createInitializedBuffer(
  device: GPUDevice,
  label: string,
  usage: GPUBufferUsageFlags,
  data: ArrayBufferView,
): GPUBuffer {
  const size = Math.max(4, alignTo4(data.byteLength));
  const buffer = device.createBuffer({
    label,
    size,
    usage,
    mappedAtCreation: true,
  });
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function assertStorageBufferFits(
  device: GPUDevice,
  label: string,
  byteLength: number,
): void {
  if (byteLength > device.limits.maxStorageBufferBindingSize) {
    throw new RangeError(
      `${label} needs ${byteLength} bytes, exceeding this adapter's ` +
        `maxStorageBufferBindingSize of ${device.limits.maxStorageBufferBindingSize}.`,
    );
  }
  if (byteLength > device.limits.maxBufferSize) {
    throw new RangeError(
      `${label} needs ${byteLength} bytes, exceeding this adapter's ` +
        `maxBufferSize of ${device.limits.maxBufferSize}.`,
    );
  }
}

function validateStepSettings(settings: JGS2StepSettings): void {
  const finiteValues: ReadonlyArray<readonly [string, number]> = [
    ["timestep", settings.timestep],
    ["iterations", settings.iterations],
    ["floorHeight", settings.floorHeight],
    ["floorStiffness", settings.floorStiffness],
    ["velocityDamping", settings.velocityDamping],
    ["regularization", settings.regularization],
    ["rotationEpsilon", settings.rotationEpsilon],
    ["maxStep", settings.maxStep],
    ["contactTangentialDamping", settings.contactTangentialDamping],
    ["contactMargin", settings.contactMargin],
    ["gravity.x", settings.gravity[0]],
    ["gravity.y", settings.gravity[1]],
    ["gravity.z", settings.gravity[2]],
  ];
  for (const [name, value] of finiteValues) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${name} must be finite; got ${value}.`);
    }
  }
  if (!(settings.timestep > 0)) {
    throw new RangeError("timestep must be positive.");
  }
  if (settings.floorStiffness < 0) {
    throw new RangeError("floorStiffness must be nonnegative.");
  }
  if (settings.velocityDamping < 0 || settings.velocityDamping > 1) {
    throw new RangeError("velocityDamping must be between zero and one.");
  }
  if (!(settings.regularization > 0)) {
    throw new RangeError("regularization must be positive.");
  }
  if (!(settings.rotationEpsilon > 0)) {
    throw new RangeError("rotationEpsilon must be positive.");
  }
  if (settings.maxStep < 0) {
    throw new RangeError("maxStep must be nonnegative.");
  }
  validateJGS2ContactParameters(
    settings.contactTangentialDamping,
    settings.contactMargin,
  );
}

function mergeStepSettings(
  defaults: JGS2StepSettings,
  overrides: Partial<JGS2StepSettings>,
): JGS2StepSettings {
  const merged: JGS2StepSettings = {
    ...defaults,
    ...overrides,
    gravity: overrides.gravity ?? defaults.gravity,
  };
  validateStepSettings(merged);
  return merged;
}

function packUniforms(
  input: Pick<JGS2GpuInput, "vertexCount" | "tetCount" | "cubatureK"> & {
    readonly bodyCount: number;
  },
  offsets: JGS2DynamicOffsets,
  settings: JGS2StepSettings,
  sourcePosition: number,
  targetPosition: number,
): Uint8Array {
  const buffer = new ArrayBuffer(JGS2_UNIFORM_BYTES);
  const integers = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);

  integers.set(
    [input.vertexCount, input.tetCount, input.cubatureK, input.bodyCount],
    0,
  );
  integers.set(
    [offsets.posA, offsets.posB, offsets.predicted, offsets.velocity],
    4,
  );
  integers.set(
    [offsets.old, offsets.vertexRotation, offsets.tetRotation, sourcePosition],
    8,
  );
  integers.set([targetPosition, offsets.bodyCorrection, 0, 0], 12);

  const inverseTimestep = 1 / settings.timestep;
  floats.set(
    [
      settings.timestep,
      inverseTimestep,
      inverseTimestep * inverseTimestep,
      settings.maxStep,
    ],
    16,
  );
  floats.set(
    [
      settings.gravity[0],
      settings.gravity[1],
      settings.gravity[2],
      settings.floorHeight,
    ],
    20,
  );
  floats.set(
    [
      settings.floorStiffness,
      settings.regularization,
      settings.rotationEpsilon,
      settings.velocityDamping,
    ],
    24,
  );
  floats.set(
    [settings.contactTangentialDamping, settings.contactMargin, 0, 0],
    28,
  );

  return new Uint8Array(buffer);
}

function createUniformBuffer(device: GPUDevice, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: JGS2_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function createBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  buffers: JGS2GpuBuffers,
  uniforms: GPUBuffer,
  label: string,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout,
    entries: [
      { binding: 0, resource: { buffer: buffers.dynamic } },
      { binding: 1, resource: { buffer: buffers.vertices } },
      { binding: 2, resource: { buffer: buffers.tets } },
      { binding: 3, resource: { buffer: buffers.stiffness } },
      { binding: 4, resource: { buffer: buffers.adjacency } },
      { binding: 5, resource: { buffer: buffers.cubature } },
      { binding: 6, resource: { buffer: uniforms } },
    ],
  });
}

function encodeDispatch(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  itemCount: number,
  label: string,
): void {
  if (itemCount === 0) {
    return;
  }
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(itemCount / JGS2_WORKGROUP_SIZE));
  pass.end();
}

function formatCompilationErrors(messages: readonly GPUCompilationMessage[]): string {
  return messages
    .filter((message) => message.type === "error")
    .map(
      (message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`,
    )
    .join("\n");
}

export class JGS2GpuSolver {
  readonly vertexCount: number;
  readonly tetCount: number;
  readonly cubatureK: number;
  readonly bodyCount: number;
  readonly dynamicOffsets: JGS2DynamicOffsets;

  private destroyed = false;
  private submittedIterations = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly inputShape: Pick<
      JGS2GpuInput,
      "vertexCount" | "tetCount" | "cubatureK"
    > & { readonly bodyCount: number },
    private readonly buffers: JGS2GpuBuffers,
    private readonly pipelines: JGS2Pipelines,
    private readonly uniforms: JGS2UniformState,
    private readonly defaultSettings: JGS2StepSettings,
    private readonly preprocessingTimestep: number,
    offsets: JGS2DynamicOffsets,
  ) {
    this.vertexCount = inputShape.vertexCount;
    this.tetCount = inputShape.tetCount;
    this.cubatureK = inputShape.cubatureK;
    this.bodyCount = inputShape.bodyCount;
    this.dynamicOffsets = offsets;
  }

  static async create(
    device: GPUDevice,
    input: JGS2GpuInput,
    settings: Partial<JGS2StepSettings> = {},
  ): Promise<JGS2GpuSolver> {
    validateJGS2GpuInput(input);
    if (device.limits.maxStorageBuffersPerShaderStage < 6) {
      throw new Error(
        "JGS2 requires six storage buffers in the compute stage, but this " +
          `adapter supports ${device.limits.maxStorageBuffersPerShaderStage}.`,
      );
    }

    const resolvedSettings = mergeStepSettings(DEFAULT_JGS2_STEP_SETTINGS, settings);
    const bodyCount = inferJGS2BodyCount(input.vertexInfo, input.vertexCount);
    const offsets = computeJGS2DynamicOffsets(
      input.vertexCount,
      input.tetCount,
      bodyCount,
    );
    const dynamic = packJGS2InitialDynamic(input, offsets);
    const vertices = packJGS2VertexStatic(input);
    const tets = packJGS2TetStatic(input);
    const cubature = packJGS2Cubature(input);

    const storageSizes: ReadonlyArray<readonly [string, number]> = [
      ["JGS2 dynamic buffer", dynamic.byteLength],
      ["JGS2 vertex buffer", vertices.byteLength],
      ["JGS2 tetrahedron buffer", tets.byteLength],
      ["JGS2 stiffness buffer", input.tetRestStiffness.byteLength],
      ["JGS2 adjacency buffer", Math.max(4, input.adjacency.byteLength)],
      ["JGS2 cubature buffer", Math.max(4, cubature.byteLength)],
    ];
    for (const [label, byteLength] of storageSizes) {
      assertStorageBufferFits(device, label, byteLength);
    }

    const buffers: JGS2GpuBuffers = {
      dynamic: createInitializedBuffer(
        device,
        "jgs2-dynamic",
        GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.VERTEX,
        dynamic,
      ),
      vertices: createInitializedBuffer(
        device,
        "jgs2-vertices",
        GPUBufferUsage.STORAGE,
        vertices,
      ),
      tets: createInitializedBuffer(
        device,
        "jgs2-tetrahedra",
        GPUBufferUsage.STORAGE,
        tets,
      ),
      stiffness: createInitializedBuffer(
        device,
        "jgs2-rest-stiffness",
        GPUBufferUsage.STORAGE,
        input.tetRestStiffness,
      ),
      adjacency: createInitializedBuffer(
        device,
        "jgs2-adjacency",
        GPUBufferUsage.STORAGE,
        input.adjacency,
      ),
      cubature: createInitializedBuffer(
        device,
        "jgs2-cubature",
        GPUBufferUsage.STORAGE,
        cubature,
      ),
    };

    const bindGroupLayout = device.createBindGroupLayout({
      label: "jgs2-bind-group-layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        ...[1, 2, 3, 4, 5].map<GPUBindGroupLayoutEntry>((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        })),
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", minBindingSize: JGS2_UNIFORM_BYTES },
        },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      label: "jgs2-pipeline-layout",
      bindGroupLayouts: [bindGroupLayout],
    });
    const shader = device.createShaderModule({
      label: "jgs2-compute-shader",
      code: jgs2Shader,
    });
    const compilation = await shader.getCompilationInfo();
    const compilationErrors = formatCompilationErrors(compilation.messages);
    if (compilationErrors) {
      for (const buffer of Object.values(buffers)) {
        buffer.destroy();
      }
      throw new Error(`JGS2 WGSL failed to compile:\n${compilationErrors}`);
    }

    const createPipeline = (entryPoint: string, label: string) =>
      device.createComputePipelineAsync({
        label,
        layout: pipelineLayout,
        compute: { module: shader, entryPoint },
      });
    const [
      predict,
      tetPolarRotation,
      vertexPolarRotation,
      solve,
      bodyHorizontalCorrection,
      applyBodyHorizontalCorrection,
      finalize,
    ] =
      await Promise.all([
        createPipeline("predict", "jgs2-predict-pipeline"),
        createPipeline("tetPolarRotation", "jgs2-tet-polar-pipeline"),
        createPipeline("vertexPolarRotation", "jgs2-vertex-polar-pipeline"),
        createPipeline("jgs2Solve", "jgs2-solve-pipeline"),
        createPipeline(
          "bodyHorizontalCorrection",
          "jgs2-body-horizontal-correction-pipeline",
        ),
        createPipeline(
          "applyBodyHorizontalCorrection",
          "jgs2-apply-body-horizontal-correction-pipeline",
        ),
        createPipeline("finalize", "jgs2-finalize-pipeline"),
      ]);
    const pipelines: JGS2Pipelines = {
      predict,
      tetPolarRotation,
      vertexPolarRotation,
      solve,
      bodyHorizontalCorrection,
      applyBodyHorizontalCorrection,
      finalize,
    };

    const base = createUniformBuffer(device, "jgs2-uniform-base");
    const fromBToA = createUniformBuffer(device, "jgs2-uniform-b-to-a");
    const fromAToB = createUniformBuffer(device, "jgs2-uniform-a-to-b");
    const uniforms: JGS2UniformState = {
      base,
      fromBToA,
      fromAToB,
      baseBindGroup: createBindGroup(
        device,
        bindGroupLayout,
        buffers,
        base,
        "jgs2-base-bind-group",
      ),
      fromBToABindGroup: createBindGroup(
        device,
        bindGroupLayout,
        buffers,
        fromBToA,
        "jgs2-b-to-a-bind-group",
      ),
      fromAToBBindGroup: createBindGroup(
        device,
        bindGroupLayout,
        buffers,
        fromAToB,
        "jgs2-a-to-b-bind-group",
      ),
    };

    return new JGS2GpuSolver(
      device,
      {
        vertexCount: input.vertexCount,
        tetCount: input.tetCount,
        cubatureK: input.cubatureK,
        bodyCount,
      },
      buffers,
      pipelines,
      uniforms,
      resolvedSettings,
      resolvedSettings.timestep,
      offsets,
    );
  }

  get currentPositionBuffer(): GPUBuffer {
    this.assertUsable();
    return this.buffers.dynamic;
  }

  get currentPositionByteOffset(): number {
    return this.dynamicOffsets.posA * 16;
  }

  get currentPositionByteLength(): number {
    return this.vertexCount * 16;
  }

  get velocityByteOffset(): number {
    return this.dynamicOffsets.velocity * 16;
  }

  get velocityByteLength(): number {
    return this.vertexCount * 16;
  }

  get currentPositionView(): JGS2PositionBufferView {
    return {
      buffer: this.currentPositionBuffer,
      offset: this.currentPositionByteOffset,
      size: this.currentPositionByteLength,
      stride: 16,
    };
  }

  get lastSubmittedIterationCount(): number {
    return this.submittedIterations;
  }

  step(overrides: Partial<JGS2StepSettings> = {}): void {
    this.assertUsable();
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but step() requested ${overrides.timestep}. Regenerate the ` +
          "precomputation before changing timestep.",
      );
    }
    const settings = mergeStepSettings(this.defaultSettings, overrides);
    const iterations = normalizeOddIterationCount(settings.iterations);
    this.submittedIterations = iterations;

    this.device.queue.writeBuffer(
      this.uniforms.base,
      0,
      packUniforms(
        this.inputShape,
        this.dynamicOffsets,
        settings,
        this.dynamicOffsets.posA,
        this.dynamicOffsets.posA,
      ),
    );
    this.device.queue.writeBuffer(
      this.uniforms.fromBToA,
      0,
      packUniforms(
        this.inputShape,
        this.dynamicOffsets,
        settings,
        this.dynamicOffsets.posB,
        this.dynamicOffsets.posA,
      ),
    );
    this.device.queue.writeBuffer(
      this.uniforms.fromAToB,
      0,
      packUniforms(
        this.inputShape,
        this.dynamicOffsets,
        settings,
        this.dynamicOffsets.posA,
        this.dynamicOffsets.posB,
      ),
    );

    const encoder = this.device.createCommandEncoder({
      label: "jgs2-step-command-encoder",
    });
    encodeDispatch(
      encoder,
      this.pipelines.predict,
      this.uniforms.baseBindGroup,
      this.vertexCount,
      "jgs2-predict-pass",
    );

    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const bindGroup =
        iteration % 2 === 0
          ? this.uniforms.fromBToABindGroup
          : this.uniforms.fromAToBBindGroup;
      encodeDispatch(
        encoder,
        this.pipelines.tetPolarRotation,
        bindGroup,
        this.tetCount,
        `jgs2-tet-polar-pass-${iteration}`,
      );
      encodeDispatch(
        encoder,
        this.pipelines.vertexPolarRotation,
        bindGroup,
        this.vertexCount,
        `jgs2-vertex-polar-pass-${iteration}`,
      );
      encodeDispatch(
        encoder,
        this.pipelines.solve,
        bindGroup,
        this.vertexCount,
        `jgs2-solve-pass-${iteration}`,
      );
    }

    encodeDispatch(
      encoder,
      this.pipelines.bodyHorizontalCorrection,
      this.uniforms.baseBindGroup,
      this.bodyCount,
      "jgs2-body-horizontal-correction-pass",
    );
    encodeDispatch(
      encoder,
      this.pipelines.applyBodyHorizontalCorrection,
      this.uniforms.baseBindGroup,
      this.vertexCount,
      "jgs2-apply-body-horizontal-correction-pass",
    );

    encodeDispatch(
      encoder,
      this.pipelines.finalize,
      this.uniforms.baseBindGroup,
      this.vertexCount,
      "jgs2-finalize-pass",
    );
    this.device.queue.submit([encoder.finish()]);
  }

  async awaitIdle(): Promise<void> {
    this.assertUsable();
    await this.device.queue.onSubmittedWorkDone();
  }

  async readPositions(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.currentPositionByteOffset,
      this.currentPositionByteLength,
      "jgs2-position-readback",
    );
  }

  async readVelocities(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.velocityByteOffset,
      this.velocityByteLength,
      "jgs2-velocity-readback",
    );
  }

  private async readVec4Region(
    sourceOffset: number,
    byteLength: number,
    label: string,
  ): Promise<Float32Array> {
    const readback = this.device.createBuffer({
      label,
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: `${label}-encoder`,
    });
    encoder.copyBufferToBuffer(
      this.buffers.dynamic,
      sourceOffset,
      readback,
      0,
      byteLength,
    );
    this.device.queue.submit([encoder.finish()]);

    try {
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      return result;
    } finally {
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    for (const buffer of Object.values(this.buffers)) {
      buffer.destroy();
    }
    this.uniforms.base.destroy();
    this.uniforms.fromBToA.destroy();
    this.uniforms.fromAToB.destroy();
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("JGS2GpuSolver has been destroyed.");
    }
  }
}
