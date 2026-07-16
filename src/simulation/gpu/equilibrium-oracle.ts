const EQUILIBRIUM_ORACLE_WORKGROUP_SIZE = 64;
const EQUILIBRIUM_ORACLE_OUTPUT_FLOATS = 4;

export interface DenseGpuEquilibriumOracleInput {
  /** Dense active-coordinate gradient. Its length defines the dimension. */
  readonly gradient: Float32Array;
  /** Dense row-major active-coordinate Hessian. */
  readonly hessian: Float32Array;
  /**
   * Consecutive dense row-major dimension-by-three bases. Each basis maps a
   * local three-vector into the full active-coordinate space.
   */
  readonly bases: Float32Array;
}

export interface PackedDenseGpuEquilibriumOracleInput
  extends DenseGpuEquilibriumOracleInput {
  readonly dimension: number;
  readonly basisCount: number;
}

function assertFiniteArray(name: string, values: Float32Array): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${name}[${index}] must be finite.`);
    }
  }
}

/** Validate and snapshot the dense inputs before they are submitted to WebGPU. */
export function packDenseGpuEquilibriumOracleInput(
  input: DenseGpuEquilibriumOracleInput,
): PackedDenseGpuEquilibriumOracleInput {
  const dimension = input.gradient.length;
  if (!Number.isSafeInteger(dimension) || dimension < 3) {
    throw new RangeError(
      `The equilibrium-oracle dimension must be an integer >= 3; got ${dimension}.`,
    );
  }
  if (input.hessian.length !== dimension * dimension) {
    throw new RangeError(
      `The equilibrium-oracle Hessian must contain ${dimension * dimension} ` +
        `values; got ${input.hessian.length}.`,
    );
  }
  const basisFloatCount = dimension * 3;
  if (
    input.bases.length < basisFloatCount ||
    input.bases.length % basisFloatCount !== 0
  ) {
    throw new RangeError(
      `Equilibrium bases must contain one or more ${dimension} by 3 matrices.`,
    );
  }

  assertFiniteArray("gradient", input.gradient);
  assertFiniteArray("hessian", input.hessian);
  assertFiniteArray("bases", input.bases);

  return {
    dimension,
    basisCount: input.bases.length / basisFloatCount,
    gradient: input.gradient.slice(),
    hessian: input.hessian.slice(),
    bases: input.bases.slice(),
  };
}

/** Remove the storage-buffer vec4 padding from GPU oracle results. */
export function decodeDenseGpuEquilibriumOracleSteps(
  paddedSteps: Float32Array,
  basisCount: number,
): Float32Array {
  if (!Number.isSafeInteger(basisCount) || basisCount < 1) {
    throw new RangeError("The equilibrium-oracle basis count must be positive.");
  }
  if (paddedSteps.length !== basisCount * EQUILIBRIUM_ORACLE_OUTPUT_FLOATS) {
    throw new RangeError(
      `The padded equilibrium result must contain ` +
        `${basisCount * EQUILIBRIUM_ORACLE_OUTPUT_FLOATS} values.`,
    );
  }
  const result = new Float32Array(basisCount * 3);
  for (let basis = 0; basis < basisCount; basis += 1) {
    const record = basis * EQUILIBRIUM_ORACLE_OUTPUT_FLOATS;
    if (
      !Number.isFinite(paddedSteps[record + 3]) ||
      paddedSteps[record + 3]! < 0.5
    ) {
      throw new Error(
        `GPU equilibrium basis ${basis} was not finite and strictly ` +
          "positive definite without pivot regularization.",
      );
    }
    result.set(
      paddedSteps.subarray(
        record,
        record + 3,
      ),
      basis * 3,
    );
  }
  return result;
}

const EQUILIBRIUM_ORACLE_SHADER = /* wgsl */ `
struct OracleParameters {
  dimension: u32,
  basis_count: u32,
  padding0: u32,
  padding1: u32,
}

@group(0) @binding(0) var<uniform> parameters: OracleParameters;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read> hessian: array<f32>;
@group(0) @binding(3) var<storage, read> bases: array<f32>;
@group(0) @binding(4) var<storage, read_write> local_steps: array<vec4<f32>>;

fn basis_value(basis: u32, row: u32, column: u32) -> f32 {
  return bases[(basis * parameters.dimension + row) * 3u + column];
}

@compute @workgroup_size(${EQUILIBRIUM_ORACLE_WORKGROUP_SIZE})
fn solve_equilibrium_block(@builtin(global_invocation_id) id: vec3<u32>) {
  let basis = id.x;
  if (basis >= parameters.basis_count) {
    return;
  }

  // Equation 19's reduced gradient: B^T g.
  var reduced_g0 = 0.0;
  var reduced_g1 = 0.0;
  var reduced_g2 = 0.0;
  for (var row = 0u; row < parameters.dimension; row += 1u) {
    let g = gradient[row];
    reduced_g0 += basis_value(basis, row, 0u) * g;
    reduced_g1 += basis_value(basis, row, 1u) * g;
    reduced_g2 += basis_value(basis, row, 2u) * g;
  }

  // Equation 19's exact dense reduced Hessian: B^T H B.
  var a00 = 0.0;
  var a01 = 0.0;
  var a02 = 0.0;
  var a10 = 0.0;
  var a11 = 0.0;
  var a12 = 0.0;
  var a20 = 0.0;
  var a21 = 0.0;
  var a22 = 0.0;
  for (var hessian_row = 0u; hessian_row < parameters.dimension; hessian_row += 1u) {
    var hb0 = 0.0;
    var hb1 = 0.0;
    var hb2 = 0.0;
    for (var inner = 0u; inner < parameters.dimension; inner += 1u) {
      let h = hessian[hessian_row * parameters.dimension + inner];
      hb0 += h * basis_value(basis, inner, 0u);
      hb1 += h * basis_value(basis, inner, 1u);
      hb2 += h * basis_value(basis, inner, 2u);
    }
    let b0 = basis_value(basis, hessian_row, 0u);
    let b1 = basis_value(basis, hessian_row, 1u);
    let b2 = basis_value(basis, hessian_row, 2u);
    a00 += b0 * hb0;
    a01 += b0 * hb1;
    a02 += b0 * hb2;
    a10 += b1 * hb0;
    a11 += b1 * hb1;
    a12 += b1 * hb2;
    a20 += b2 * hb0;
    a21 += b2 * hb1;
    a22 += b2 * hb2;
  }

  // Roundoff can make the two accumulated triangles differ slightly.
  let s01 = 0.5 * (a01 + a10);
  let s02 = 0.5 * (a02 + a20);
  let s12 = 0.5 * (a12 + a21);

  // Guarded 3x3 Cholesky. The guard prevents undefined arithmetic, but lane w
  // reports whether every unmodified pivot was strictly above the guard. The
  // CPU decoder rejects any guarded result so this oracle never silently
  // substitutes a regularized solve for the exact SPD solve.
  let diagonal_scale = max(max(abs(a00), abs(a11)), max(abs(a22), 1.0));
  let pivot_floor = max(diagonal_scale * 1.0e-7, 1.0e-12);
  let pivot0 = a00;
  let l00 = sqrt(max(pivot0, pivot_floor));
  let l10 = s01 / l00;
  let l20 = s02 / l00;
  let pivot1 = a11 - l10 * l10;
  let l11 = sqrt(max(pivot1, pivot_floor));
  let l21 = (s12 - l20 * l10) / l11;
  let pivot2 = a22 - l20 * l20 - l21 * l21;
  let l22 = sqrt(max(pivot2, pivot_floor));

  // L y = -B^T g, followed by L^T delta = y.
  let y0 = -reduced_g0 / l00;
  let y1 = (-reduced_g1 - l10 * y0) / l11;
  let y2 = (-reduced_g2 - l20 * y0 - l21 * y1) / l22;
  let delta2 = y2 / l22;
  let delta1 = (y1 - l21 * delta2) / l11;
  let delta0 = (y0 - l10 * delta1 - l20 * delta2) / l00;

  let finite_limit = 3.0e38;
  let exact = pivot0 > pivot_floor && pivot1 > pivot_floor &&
    pivot2 > pivot_floor && delta0 == delta0 && delta1 == delta1 &&
    delta2 == delta2 && abs(delta0) < finite_limit &&
    abs(delta1) < finite_limit && abs(delta2) < finite_limit;
  local_steps[basis] = vec4<f32>(
    delta0,
    delta1,
    delta2,
    select(0.0, 1.0, exact),
  );
}
`;

function createStorageBuffer(
  device: GPUDevice,
  label: string,
  data: Float32Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function assertStorageBufferFits(
  device: GPUDevice,
  name: string,
  byteLength: number,
): void {
  if (byteLength > device.limits.maxStorageBufferBindingSize) {
    throw new RangeError(
      `${name} requires ${byteLength} bytes, exceeding the device's ` +
        `${device.limits.maxStorageBufferBindingSize}-byte storage binding limit.`,
    );
  }
}

/**
 * Test-only WebGPU reference for the paper's exact equilibrium-coordinate
 * local update. Production JGS2 dispatches intentionally do not call this
 * dense O(numberOfBases * dimension^2) implementation.
 */
export async function solveDenseEquilibriumBasisOnGpu(
  device: GPUDevice,
  input: DenseGpuEquilibriumOracleInput,
): Promise<Float32Array> {
  const packed = packDenseGpuEquilibriumOracleInput(input);
  assertStorageBufferFits(device, "gradient", packed.gradient.byteLength);
  assertStorageBufferFits(device, "hessian", packed.hessian.byteLength);
  assertStorageBufferFits(device, "bases", packed.bases.byteLength);

  const outputByteLength =
    packed.basisCount * EQUILIBRIUM_ORACLE_OUTPUT_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  assertStorageBufferFits(device, "local steps", outputByteLength);

  const buffers: GPUBuffer[] = [];
  let validationError: GPUError | null = null;
  let paddedSteps: Float32Array | undefined;
  device.pushErrorScope("validation");
  try {
    const shaderModule = device.createShaderModule({
      label: "Dense equilibrium-basis oracle shader",
      code: EQUILIBRIUM_ORACLE_SHADER,
    });
    const compilation = await shaderModule.getCompilationInfo();
    const compilationErrors = compilation.messages.filter(
      (message) => message.type === "error",
    );
    if (compilationErrors.length > 0) {
      throw new Error(
        `Dense equilibrium-oracle WGSL compilation failed:\n${compilationErrors
          .map((message) => message.message)
          .join("\n")}`,
      );
    }

    const pipeline = await device.createComputePipelineAsync({
      label: "Dense equilibrium-basis oracle pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "solve_equilibrium_block",
      },
    });

    const parameterBuffer = device.createBuffer({
      label: "Dense equilibrium-oracle parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      parameterBuffer,
      0,
      new Uint32Array([packed.dimension, packed.basisCount, 0, 0]),
    );
    buffers.push(parameterBuffer);

    const gradientBuffer = createStorageBuffer(
      device,
      "Dense equilibrium-oracle gradient",
      packed.gradient,
    );
    const hessianBuffer = createStorageBuffer(
      device,
      "Dense equilibrium-oracle Hessian",
      packed.hessian,
    );
    const basisBuffer = createStorageBuffer(
      device,
      "Dense equilibrium-oracle bases",
      packed.bases,
    );
    const outputBuffer = device.createBuffer({
      label: "Dense equilibrium-oracle output",
      size: outputByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      label: "Dense equilibrium-oracle readback",
      size: outputByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    buffers.push(
      gradientBuffer,
      hessianBuffer,
      basisBuffer,
      outputBuffer,
      readbackBuffer,
    );

    const bindGroup = device.createBindGroup({
      label: "Dense equilibrium-oracle bind group",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: parameterBuffer } },
        { binding: 1, resource: { buffer: gradientBuffer } },
        { binding: 2, resource: { buffer: hessianBuffer } },
        { binding: 3, resource: { buffer: basisBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({
      label: "Dense equilibrium-oracle command encoder",
    });
    const pass = encoder.beginComputePass({
      label: "Dense equilibrium-oracle compute pass",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(packed.basisCount / EQUILIBRIUM_ORACLE_WORKGROUP_SIZE),
    );
    pass.end();
    encoder.copyBufferToBuffer(
      outputBuffer,
      0,
      readbackBuffer,
      0,
      outputByteLength,
    );
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    paddedSteps = new Float32Array(
      readbackBuffer.getMappedRange().slice(0),
    );
    readbackBuffer.unmap();
  } finally {
    validationError = await device.popErrorScope();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }

  if (validationError) {
    throw new Error(
      `Dense equilibrium-oracle WebGPU validation failed: ${validationError.message}`,
    );
  }
  if (!paddedSteps) {
    throw new Error("Dense equilibrium-oracle readback did not complete.");
  }
  return decodeDenseGpuEquilibriumOracleSteps(
    paddedSteps,
    packed.basisCount,
  );
}
