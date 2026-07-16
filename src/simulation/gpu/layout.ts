export const JGS2_VERTEX_STATIC_WORDS = 12;
export const JGS2_TET_STATIC_WORDS = 20;
export const JGS2_REST_STIFFNESS_FLOATS = 12 * 12;
export const JGS2_CUBATURE_BASIS_FLOATS = 4 * 3 * 3;
export const JGS2_CUBATURE_RECORD_WORDS =
  2 + JGS2_CUBATURE_BASIS_FLOATS;
export const JGS2_UNIFORM_BYTES = 8 * 16;
export const JGS2_MATERIAL_COROTATED_LINEAR = 0;
export const JGS2_MATERIAL_STABLE_NEO_HOOKEAN = 1;

export interface JGS2GpuInput {
  readonly vertexCount: number;
  readonly tetCount: number;
  readonly cubatureK: number;

  /** Four floats per vertex. xyz are positions; w is normalized to one. */
  readonly positions: Float32Array;
  /** Optional four floats per vertex. xyz are velocities. */
  readonly velocities?: Float32Array;

  /** vec4 per vertex: rest xyz and mass in w. */
  readonly vertexRest: Float32Array;
  /** vec4 per vertex, retained beside the simulation data for rendering. */
  readonly vertexColors: Float32Array;
  /** vec4u per vertex: adjacency start/count, pinned flag, body id. */
  readonly vertexInfo: Uint32Array;

  /** Four vertex indices per tetrahedron. */
  readonly tetIndices: Uint32Array;
  /** Three padded vec4 columns of inverse Dm per tetrahedron. */
  readonly tetInverseDm: Float32Array;
  /**
   * vec4 per tetrahedron: rest volume, material lambda, material mu, and
   * JGS2_MATERIAL_* model tag. Stable Neo-Hookean uses the paper's adjusted
   * lambda/mu; co-rotated regression scenes retain conventional Lamé values.
   */
  readonly tetMeta: Float32Array;
  /**
   * Row-major 12x12 corotated rest stiffness per tetrahedron. Cubature
   * training should distribute each vertex's inertia equally among its
   * incident tetrahedra; the shader uses the same m / adjacencyCount rule.
   */
  readonly tetRestStiffness: Float32Array;

  /** Flat CSR payload containing incident tetrahedron ids. */
  readonly adjacency: Uint32Array;

  /** vertexCount * cubatureK tetrahedron ids; 0xffffffff is an empty slot. */
  readonly cubatureTetIds: Uint32Array;
  /** vertexCount * cubatureK nonnegative weights. */
  readonly cubatureWeights: Float32Array;
  /** Four row-major 3x3 Ubar blocks per cubature record. */
  readonly cubatureBasis: Float32Array;
}

export interface JGS2DynamicOffsets {
  /** Offsets are expressed in vec4 elements, not bytes. */
  readonly posA: number;
  readonly posB: number;
  readonly predicted: number;
  readonly velocity: number;
  readonly old: number;
  readonly vertexRotation: number;
  readonly tetRotation: number;
  readonly bodyCorrection: number;
  /** Per-vertex final nonlinear update magnitude and validity flag. */
  readonly finalUpdate: number;
  readonly vec4Count: number;
}

function assertInteger(name: string, value: number, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be an integer >= ${minimum}; got ${value}.`);
  }
}

function assertLength(
  name: string,
  value: ArrayBufferView,
  expected: number,
): void {
  const actual = "length" in value ? Number(value.length) : value.byteLength;
  if (actual !== expected) {
    throw new RangeError(`${name} must contain ${expected} values; got ${actual}.`);
  }
}

function assertFiniteArray(name: string, values: Float32Array): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${name}[${index}] must be finite.`);
    }
  }
}

export function validateJGS2GpuInput(input: JGS2GpuInput): void {
  const { vertexCount, tetCount, cubatureK } = input;
  assertInteger("vertexCount", vertexCount, 1);
  assertInteger("tetCount", tetCount, 1);
  assertInteger("cubatureK", cubatureK, 0);

  assertLength("positions", input.positions, vertexCount * 4);
  if (input.velocities) {
    assertLength("velocities", input.velocities, vertexCount * 4);
  }
  assertLength("vertexRest", input.vertexRest, vertexCount * 4);
  assertLength("vertexColors", input.vertexColors, vertexCount * 4);
  assertLength("vertexInfo", input.vertexInfo, vertexCount * 4);
  inferJGS2BodyCount(input.vertexInfo, vertexCount);
  assertLength("tetIndices", input.tetIndices, tetCount * 4);
  assertLength("tetInverseDm", input.tetInverseDm, tetCount * 12);
  assertLength("tetMeta", input.tetMeta, tetCount * 4);
  assertLength(
    "tetRestStiffness",
    input.tetRestStiffness,
    tetCount * JGS2_REST_STIFFNESS_FLOATS,
  );

  const recordCount = vertexCount * cubatureK;
  assertLength("cubatureTetIds", input.cubatureTetIds, recordCount);
  assertLength("cubatureWeights", input.cubatureWeights, recordCount);
  assertLength(
    "cubatureBasis",
    input.cubatureBasis,
    recordCount * JGS2_CUBATURE_BASIS_FLOATS,
  );

  assertFiniteArray("positions", input.positions);
  if (input.velocities) {
    assertFiniteArray("velocities", input.velocities);
  }
  assertFiniteArray("vertexRest", input.vertexRest);
  assertFiniteArray("vertexColors", input.vertexColors);
  assertFiniteArray("tetInverseDm", input.tetInverseDm);
  assertFiniteArray("tetMeta", input.tetMeta);
  assertFiniteArray("tetRestStiffness", input.tetRestStiffness);
  assertFiniteArray("cubatureWeights", input.cubatureWeights);
  assertFiniteArray("cubatureBasis", input.cubatureBasis);

  for (let tet = 0; tet < input.tetIndices.length; tet += 1) {
    if (input.tetIndices[tet] >= vertexCount) {
      throw new RangeError(
        `tetIndices[${tet}] references vertex ${input.tetIndices[tet]}, ` +
          `but vertexCount is ${vertexCount}.`,
      );
    }
  }

  for (let tet = 0; tet < tetCount; tet += 1) {
    const base = tet * 4;
    const volume = input.tetMeta[base]!;
    const lambda = input.tetMeta[base + 1]!;
    const mu = input.tetMeta[base + 2]!;
    const model = input.tetMeta[base + 3]!;
    if (!(volume > 0) || !(mu > 0)) {
      throw new RangeError(
        `tetMeta for tetrahedron ${tet} requires positive volume and mu.`,
      );
    }
    if (
      model !== JGS2_MATERIAL_COROTATED_LINEAR &&
      model !== JGS2_MATERIAL_STABLE_NEO_HOOKEAN
    ) {
      throw new RangeError(
        `tetMeta for tetrahedron ${tet} has unknown material model ${model}.`,
      );
    }
    if (
      model === JGS2_MATERIAL_STABLE_NEO_HOOKEAN &&
      !(lambda > 0)
    ) {
      throw new RangeError(
        `Stable Neo-Hookean tetMeta for tetrahedron ${tet} requires positive lambda.`,
      );
    }
    if (
      model === JGS2_MATERIAL_COROTATED_LINEAR &&
      !(3 * lambda + 2 * mu > 0)
    ) {
      throw new RangeError(
        `Co-rotated tetMeta for tetrahedron ${tet} requires positive bulk modulus.`,
      );
    }
  }

  for (let adjacencyIndex = 0; adjacencyIndex < input.adjacency.length; adjacencyIndex += 1) {
    if (input.adjacency[adjacencyIndex] >= tetCount) {
      throw new RangeError(
        `adjacency[${adjacencyIndex}] references tetrahedron ` +
          `${input.adjacency[adjacencyIndex]}, but tetCount is ${tetCount}.`,
      );
    }
  }

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const info = vertex * 4;
    const start = input.vertexInfo[info];
    const count = input.vertexInfo[info + 1];
    if (start + count > input.adjacency.length) {
      throw new RangeError(
        `vertexInfo for vertex ${vertex} addresses adjacency range ` +
          `[${start}, ${start + count}), beyond ${input.adjacency.length}.`,
      );
    }
    const mass = input.vertexRest[vertex * 4 + 3];
    const pinned = input.vertexInfo[info + 2] !== 0;
    if (!pinned && !(mass > 0)) {
      throw new RangeError(`Unpinned vertex ${vertex} must have positive mass.`);
    }
  }

  for (let record = 0; record < recordCount; record += 1) {
    const weight = input.cubatureWeights[record];
    const tet = input.cubatureTetIds[record];
    if (weight < 0) {
      throw new RangeError(`cubatureWeights[${record}] must be nonnegative.`);
    }
    if (weight > 0 && tet >= tetCount) {
      throw new RangeError(
        `Positive cubature record ${record} references tetrahedron ${tet}, ` +
          `but tetCount is ${tetCount}.`,
      );
    }
  }
}

export function computeJGS2DynamicOffsets(
  vertexCount: number,
  tetCount: number,
  bodyCount = 0,
): JGS2DynamicOffsets {
  assertInteger("vertexCount", vertexCount, 1);
  assertInteger("tetCount", tetCount, 1);
  assertInteger("bodyCount", bodyCount, 0);

  const posA = 0;
  const posB = posA + vertexCount;
  const predicted = posB + vertexCount;
  const velocity = predicted + vertexCount;
  const old = velocity + vertexCount;
  const vertexRotation = old + vertexCount;
  const tetRotation = vertexRotation + vertexCount * 3;
  const bodyCorrection = tetRotation + tetCount * 3;
  // Keep the existing position/rotation/body ABI stable and append diagnostics.
  const finalUpdate = bodyCorrection + bodyCount;

  return {
    posA,
    posB,
    predicted,
    velocity,
    old,
    vertexRotation,
    tetRotation,
    bodyCorrection,
    finalUpdate,
    vec4Count: finalUpdate + vertexCount,
  };
}

/**
 * Body ids occupy vertexInfo.w and are expected to be compact enough that one
 * correction slot per id is no larger than the vertex array itself.
 */
export function inferJGS2BodyCount(
  vertexInfo: Uint32Array,
  vertexCount: number,
): number {
  assertInteger("vertexCount", vertexCount, 1);
  assertLength("vertexInfo", vertexInfo, vertexCount * 4);
  let maximumBody = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    maximumBody = Math.max(maximumBody, vertexInfo[vertex * 4 + 3]!);
  }
  const bodyCount = maximumBody + 1;
  if (!Number.isSafeInteger(bodyCount) || bodyCount > vertexCount) {
    throw new RangeError(
      "vertexInfo body ids must be compact nonnegative values whose maximum " +
        `is smaller than vertexCount (${vertexCount}); got ${maximumBody}.`,
    );
  }
  return bodyCount;
}

function writeIdentityRotation(
  target: Float32Array,
  vec4Offset: number,
  index: number,
): void {
  const base = (vec4Offset + index * 3) * 4;
  target[base] = 1;
  target[base + 5] = 1;
  target[base + 10] = 1;
}

export function packJGS2InitialDynamic(
  input: JGS2GpuInput,
  offsets = computeJGS2DynamicOffsets(input.vertexCount, input.tetCount),
): Float32Array {
  const packed = new Float32Array(offsets.vec4Count * 4);
  const positionRegions = [offsets.posA, offsets.posB, offsets.predicted, offsets.old];

  for (const region of positionRegions) {
    packed.set(input.positions, region * 4);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      packed[(region + vertex) * 4 + 3] = 1;
    }
  }
  if (input.velocities) {
    packed.set(input.velocities, offsets.velocity * 4);
  }
  for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
    packed[(offsets.velocity + vertex) * 4 + 3] = 0;
    writeIdentityRotation(packed, offsets.vertexRotation, vertex);
  }
  for (let tet = 0; tet < input.tetCount; tet += 1) {
    writeIdentityRotation(packed, offsets.tetRotation, tet);
  }

  return packed;
}

export function packJGS2VertexStatic(input: JGS2GpuInput): Uint8Array {
  const buffer = new ArrayBuffer(input.vertexCount * JGS2_VERTEX_STATIC_WORDS * 4);
  const floats = new Float32Array(buffer);
  const integers = new Uint32Array(buffer);

  for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
    const destination = vertex * JGS2_VERTEX_STATIC_WORDS;
    const source = vertex * 4;
    floats.set(input.vertexRest.subarray(source, source + 4), destination);
    floats.set(input.vertexColors.subarray(source, source + 4), destination + 4);
    integers.set(input.vertexInfo.subarray(source, source + 4), destination + 8);
  }

  return new Uint8Array(buffer);
}

export function packJGS2TetStatic(input: JGS2GpuInput): Uint8Array {
  const buffer = new ArrayBuffer(input.tetCount * JGS2_TET_STATIC_WORDS * 4);
  const floats = new Float32Array(buffer);
  const integers = new Uint32Array(buffer);

  for (let tet = 0; tet < input.tetCount; tet += 1) {
    const destination = tet * JGS2_TET_STATIC_WORDS;
    const vec4Source = tet * 4;
    const matrixSource = tet * 12;
    integers.set(input.tetIndices.subarray(vec4Source, vec4Source + 4), destination);
    floats.set(
      input.tetInverseDm.subarray(matrixSource, matrixSource + 12),
      destination + 4,
    );
    floats.set(input.tetMeta.subarray(vec4Source, vec4Source + 4), destination + 16);
  }

  return new Uint8Array(buffer);
}

export function packJGS2Cubature(input: JGS2GpuInput): Uint32Array {
  const recordCount = input.vertexCount * input.cubatureK;
  const buffer = new ArrayBuffer(recordCount * JGS2_CUBATURE_RECORD_WORDS * 4);
  const integers = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);

  for (let record = 0; record < recordCount; record += 1) {
    const destination = record * JGS2_CUBATURE_RECORD_WORDS;
    const basisSource = record * JGS2_CUBATURE_BASIS_FLOATS;
    integers[destination] = input.cubatureTetIds[record];
    floats[destination + 1] = input.cubatureWeights[record];
    floats.set(
      input.cubatureBasis.subarray(
        basisSource,
        basisSource + JGS2_CUBATURE_BASIS_FLOATS,
      ),
      destination + 2,
    );
  }

  return integers;
}

export function normalizeOddIterationCount(iterations: number): number {
  if (!Number.isFinite(iterations)) {
    throw new RangeError(`iterations must be finite; got ${iterations}.`);
  }
  const integral = Math.max(1, Math.floor(iterations));
  return integral % 2 === 0 ? integral + 1 : integral;
}

/**
 * JGS2 bases and Cubature weights include M / h^2. Runtime h is compatible
 * only when it rounds to the exact same f32 value uploaded to the GPU.
 */
export function jgs2TimestepsMatch(
  preprocessingTimestep: number,
  runtimeTimestep: number,
): boolean {
  return (
    Number.isFinite(preprocessingTimestep) &&
    Number.isFinite(runtimeTimestep) &&
    preprocessingTimestep > 0 &&
    runtimeTimestep > 0 &&
    Math.fround(preprocessingTimestep) === Math.fround(runtimeTimestep)
  );
}

/** Validate the viscous tangential floor-contact controls used by finalize. */
export function validateJGS2ContactParameters(
  contactTangentialDamping: number,
  contactMargin: number,
): void {
  if (
    !Number.isFinite(contactTangentialDamping) ||
    contactTangentialDamping < 0
  ) {
    throw new RangeError(
      "contactTangentialDamping must be finite and nonnegative.",
    );
  }
  if (!Number.isFinite(contactMargin) || contactMargin < 0) {
    throw new RangeError("contactMargin must be finite and nonnegative.");
  }
}
