export const JGS2_CLOTH_WORD_BYTES = 4;
export const JGS2_CLOTH_VEC4_WORDS = 4;
export const JGS2_CLOTH_GLOBAL_VEC4S = 2;
export const JGS2_CLOTH_TRIANGLE_VEC4S = 4;
export const JGS2_CLOTH_HINGE_VEC4S = 2;
export const JGS2_CLOTH_GLOBAL_WORDS =
  JGS2_CLOTH_GLOBAL_VEC4S * JGS2_CLOTH_VEC4_WORDS;
export const JGS2_CLOTH_TRIANGLE_WORDS =
  JGS2_CLOTH_TRIANGLE_VEC4S * JGS2_CLOTH_VEC4_WORDS;
export const JGS2_CLOTH_HINGE_WORDS =
  JGS2_CLOTH_HINGE_VEC4S * JGS2_CLOTH_VEC4_WORDS;
export const JGS2_CLOTH_GLOBAL_BYTES =
  JGS2_CLOTH_GLOBAL_WORDS * JGS2_CLOTH_WORD_BYTES;
export const JGS2_CLOTH_TRIANGLE_BYTES =
  JGS2_CLOTH_TRIANGLE_WORDS * JGS2_CLOTH_WORD_BYTES;
export const JGS2_CLOTH_HINGE_BYTES =
  JGS2_CLOTH_HINGE_WORDS * JGS2_CLOTH_WORD_BYTES;

/** "CSR2" in little-endian ASCII. */
export const JGS2_CLOTH_INCIDENCE_MAGIC = 0x3252_5343;
export const JGS2_CLOTH_INCIDENCE_HEADER_WORDS = 8;

/** u32 fields in the validated incidence suffix header. */
export const JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS = {
  magic: 0,
  vertexCount: 1,
  triangleIncidenceCount: 2,
  hingeIncidenceCount: 3,
  triangleRows: 4,
  triangleElements: 5,
  hingeRows: 6,
  wordCount: 7,
} as const;

/** Mixed u32/f32 fields in the first global vec4. */
export const JGS2_CLOTH_CONTROL_WORD_OFFSETS = {
  triangleCount: 0,
  hingeCount: 1,
  reserved: 2,
  safeAlpha: 3,
} as const;

/** f32 fields in the second global vec4. */
export const JGS2_CLOTH_MATERIAL_WORD_OFFSETS = {
  planeStressLambda: 0,
  mu: 1,
  thickness: 2,
  bendingStiffness: 3,
} as const;

/** Vec4 offsets relative to one triangle record. */
export const JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS = {
  indices: 0,
  inverseRestBasis: 1,
  areaAttributes: 2,
  scratch: 3,
} as const;

/** f32 fields in a triangle's area/attributes vec4. */
export const JGS2_CLOTH_TRIANGLE_AREA_WORD_OFFSETS = {
  restArea: 0,
  reserved0: 1,
  reserved1: 2,
  reserved2: 3,
} as const;

/** Vec4 offsets relative to one hinge record. */
export const JGS2_CLOTH_HINGE_VEC4_OFFSETS = {
  indices: 0,
  rest: 1,
} as const;

/** f32 fields in a hinge's rest-data vec4. */
export const JGS2_CLOTH_HINGE_REST_WORD_OFFSETS = {
  angle: 0,
  edgeLength: 1,
  reserved0: 2,
  reserved1: 3,
} as const;

export interface JGS2GpuClothInput {
  /** Bounds every triangle and hinge index. */
  readonly vertexCount: number;
  /** Three vertex indices per triangle. */
  readonly triangleIndices: Uint32Array;
  /** Four row-major inverse-rest-basis values per triangle. */
  readonly triangleInverseRestBases: Float32Array;
  readonly triangleRestAreas: Float32Array;
  /** Four indices per oriented hinge: triangles (0,1,2) and (1,0,3). */
  readonly hingeIndices: Uint32Array;
  readonly hingeRestAngles: Float32Array;
  readonly hingeRestEdgeLengths: Float32Array;
  readonly youngModulus: number;
  readonly poissonRatio: number;
  readonly thickness: number;
  readonly bendingStiffness: number;
}

export interface PackedJGS2GpuClothArena {
  readonly buffer: ArrayBuffer;
  /** u32 interpretation for counts and indices. */
  readonly integers: Uint32Array;
  /** f32 interpretation for material, rest data, and scratch. */
  readonly floats: Float32Array;
  readonly byteLength: number;
  readonly triangleCount: number;
  readonly hingeCount: number;
  /** Word offset of the validated vertex-to-element CSR suffix. */
  readonly incidenceWordOffset: number;
  readonly incidenceWordCount: number;
  readonly planeStressLambda: number;
  readonly mu: number;
}

const MAX_U32 = 0xffff_ffff;

function requireCount(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_U32) {
    throw new RangeError(`${label} must be an unsigned 32-bit count.`);
  }
}

function requireTypedArray(
  value: ArrayBufferView,
  constructor: Uint32ArrayConstructor | Float32ArrayConstructor,
  label: string,
): void {
  if (!(value instanceof constructor)) {
    throw new TypeError(`${label} must use ${constructor.name}.`);
  }
}

function requireFiniteArray(values: Float32Array, label: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${label}[${index}] must be finite.`);
    }
  }
}

function requireLength(
  values: ArrayLike<number>,
  expected: number,
  label: string,
): void {
  if (values.length !== expected) {
    throw new RangeError(`${label} must contain ${expected} values; got ${values.length}.`);
  }
}

function validateElementIndices(
  values: Uint32Array,
  width: number,
  vertexCount: number,
  label: string,
): void {
  for (let element = 0; element < values.length / width; element += 1) {
    const distinct = new Set<number>();
    for (let lane = 0; lane < width; lane += 1) {
      const index = values[element * width + lane]!;
      if (index >= vertexCount) {
        throw new RangeError(
          `${label} ${element} index ${index} is outside vertexCount ${vertexCount}.`,
        );
      }
      distinct.add(index);
    }
    if (distinct.size !== width) {
      throw new RangeError(`${label} ${element} must reference ${width} distinct vertices.`);
    }
  }
}

function alignTo4Words(wordCount: number): number {
  return Math.ceil(wordCount / JGS2_CLOTH_VEC4_WORDS) * JGS2_CLOTH_VEC4_WORDS;
}

interface IncidentElementCsr {
  readonly rows: Uint32Array;
  readonly elements: Uint32Array;
}

function buildIncidentElementCsr(
  vertexCount: number,
  elementIndices: Uint32Array,
  elementWidth: number,
): IncidentElementCsr {
  const elementCount = elementIndices.length / elementWidth;
  const rows = new Uint32Array(vertexCount + 1);
  for (let index = 0; index < elementIndices.length; index += 1) {
    rows[elementIndices[index]! + 1] += 1;
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    rows[vertex + 1] += rows[vertex]!;
  }

  const elements = new Uint32Array(elementIndices.length);
  const cursors = rows.slice(0, vertexCount);
  for (let element = 0; element < elementCount; element += 1) {
    for (let lane = 0; lane < elementWidth; lane += 1) {
      const vertex = elementIndices[element * elementWidth + lane]!;
      elements[cursors[vertex]!] = element;
      cursors[vertex] += 1;
    }
  }
  return { rows, elements };
}

/** Validate the static cloth topology/rest data before crossing the GPU ABI. */
export function validateJGS2GpuClothInput(input: JGS2GpuClothInput): void {
  requireCount(input.vertexCount, "Cloth vertexCount");
  requireTypedArray(input.triangleIndices, Uint32Array, "triangleIndices");
  requireTypedArray(
    input.triangleInverseRestBases,
    Float32Array,
    "triangleInverseRestBases",
  );
  requireTypedArray(input.triangleRestAreas, Float32Array, "triangleRestAreas");
  requireTypedArray(input.hingeIndices, Uint32Array, "hingeIndices");
  requireTypedArray(input.hingeRestAngles, Float32Array, "hingeRestAngles");
  requireTypedArray(
    input.hingeRestEdgeLengths,
    Float32Array,
    "hingeRestEdgeLengths",
  );

  if (input.triangleIndices.length % 3 !== 0) {
    throw new RangeError("triangleIndices must contain three indices per triangle.");
  }
  if (input.hingeIndices.length % 4 !== 0) {
    throw new RangeError("hingeIndices must contain four indices per hinge.");
  }
  const triangleCount = input.triangleIndices.length / 3;
  const hingeCount = input.hingeIndices.length / 4;
  requireCount(triangleCount, "Triangle count");
  requireCount(hingeCount, "Hinge count");
  requireLength(
    input.triangleInverseRestBases,
    triangleCount * 4,
    "triangleInverseRestBases",
  );
  requireLength(input.triangleRestAreas, triangleCount, "triangleRestAreas");
  requireLength(input.hingeRestAngles, hingeCount, "hingeRestAngles");
  requireLength(
    input.hingeRestEdgeLengths,
    hingeCount,
    "hingeRestEdgeLengths",
  );
  validateElementIndices(
    input.triangleIndices,
    3,
    input.vertexCount,
    "Triangle",
  );
  validateElementIndices(input.hingeIndices, 4, input.vertexCount, "Hinge");

  requireFiniteArray(input.triangleInverseRestBases, "triangleInverseRestBases");
  requireFiniteArray(input.triangleRestAreas, "triangleRestAreas");
  requireFiniteArray(input.hingeRestAngles, "hingeRestAngles");
  requireFiniteArray(input.hingeRestEdgeLengths, "hingeRestEdgeLengths");
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (!(input.triangleRestAreas[triangle]! > 0)) {
      throw new RangeError(`triangleRestAreas[${triangle}] must be positive.`);
    }
    const base = triangle * 4;
    const determinant =
      input.triangleInverseRestBases[base]! *
        input.triangleInverseRestBases[base + 3]! -
      input.triangleInverseRestBases[base + 1]! *
        input.triangleInverseRestBases[base + 2]!;
    if (determinant === 0 || !Number.isFinite(determinant)) {
      throw new RangeError(
        `triangleInverseRestBases for triangle ${triangle} must be invertible.`,
      );
    }
  }
  for (let hinge = 0; hinge < hingeCount; hinge += 1) {
    if (!(input.hingeRestEdgeLengths[hinge]! > 0)) {
      throw new RangeError(`hingeRestEdgeLengths[${hinge}] must be positive.`);
    }
  }

  if (!(input.youngModulus > 0) || !Number.isFinite(input.youngModulus)) {
    throw new RangeError("Cloth Young's modulus must be finite and positive.");
  }
  if (
    !(input.poissonRatio >= 0 && input.poissonRatio < 0.5) ||
    !Number.isFinite(input.poissonRatio)
  ) {
    throw new RangeError("Cloth Poisson ratio must be finite and in [0, 0.5).");
  }
  if (!(input.thickness > 0) || !Number.isFinite(input.thickness)) {
    throw new RangeError("Cloth thickness must be finite and positive.");
  }
  if (!(input.bendingStiffness >= 0) || !Number.isFinite(input.bendingStiffness)) {
    throw new RangeError("Cloth bending stiffness must be finite and nonnegative.");
  }
}

/**
 * Pack two global vec4s, then four vec4s per triangle, two per hinge, and a
 * validated vertex-to-triangle/hinge CSR suffix. Integer indices and
 * floating-point rest data share one storage-buffer arena.
 */
export function packJGS2GpuClothArena(
  input: JGS2GpuClothInput,
): PackedJGS2GpuClothArena {
  validateJGS2GpuClothInput(input);
  const triangleCount = input.triangleIndices.length / 3;
  const hingeCount = input.hingeIndices.length / 4;
  const triangleCsr = buildIncidentElementCsr(
    input.vertexCount,
    input.triangleIndices,
    3,
  );
  const hingeCsr = buildIncidentElementCsr(
    input.vertexCount,
    input.hingeIndices,
    4,
  );
  const incidenceWordOffset =
    JGS2_CLOTH_GLOBAL_WORDS +
    triangleCount * JGS2_CLOTH_TRIANGLE_WORDS +
    hingeCount * JGS2_CLOTH_HINGE_WORDS;
  const triangleRowsOffset = JGS2_CLOTH_INCIDENCE_HEADER_WORDS;
  const triangleElementsOffset = triangleRowsOffset + triangleCsr.rows.length;
  const hingeRowsOffset = triangleElementsOffset + triangleCsr.elements.length;
  const hingeElementsOffset = hingeRowsOffset + hingeCsr.rows.length;
  const incidenceWordCount = alignTo4Words(
    hingeElementsOffset + hingeCsr.elements.length,
  );
  if (!Number.isSafeInteger(incidenceWordCount) || incidenceWordCount > MAX_U32) {
    throw new RangeError("Cloth incidence suffix exceeds unsigned 32-bit storage.");
  }
  const byteLength =
    (incidenceWordOffset + incidenceWordCount) * JGS2_CLOTH_WORD_BYTES;
  const buffer = new ArrayBuffer(byteLength);
  const integers = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  const planeStressLambda =
    (input.youngModulus * input.poissonRatio) /
    (1 - input.poissonRatio * input.poissonRatio);
  const mu = input.youngModulus / (2 * (1 + input.poissonRatio));

  integers[JGS2_CLOTH_CONTROL_WORD_OFFSETS.triangleCount] = triangleCount;
  integers[JGS2_CLOTH_CONTROL_WORD_OFFSETS.hingeCount] = hingeCount;
  floats[JGS2_CLOTH_CONTROL_WORD_OFFSETS.safeAlpha] = 1;
  const materialBase = JGS2_CLOTH_VEC4_WORDS;
  floats[
    materialBase + JGS2_CLOTH_MATERIAL_WORD_OFFSETS.planeStressLambda
  ] = planeStressLambda;
  floats[materialBase + JGS2_CLOTH_MATERIAL_WORD_OFFSETS.mu] = mu;
  floats[materialBase + JGS2_CLOTH_MATERIAL_WORD_OFFSETS.thickness] =
    input.thickness;
  floats[materialBase + JGS2_CLOTH_MATERIAL_WORD_OFFSETS.bendingStiffness] =
    input.bendingStiffness;

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const recordBase =
      JGS2_CLOTH_GLOBAL_WORDS + triangle * JGS2_CLOTH_TRIANGLE_WORDS;
    const indicesBase =
      recordBase +
      JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS.indices * JGS2_CLOTH_VEC4_WORDS;
    integers.set(
      input.triangleIndices.subarray(triangle * 3, triangle * 3 + 3),
      indicesBase,
    );
    const inverseBase =
      recordBase +
      JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS.inverseRestBasis *
        JGS2_CLOTH_VEC4_WORDS;
    floats.set(
      input.triangleInverseRestBases.subarray(
        triangle * 4,
        triangle * 4 + 4,
      ),
      inverseBase,
    );
    const areaBase =
      recordBase +
      JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS.areaAttributes *
        JGS2_CLOTH_VEC4_WORDS;
    floats[areaBase + JGS2_CLOTH_TRIANGLE_AREA_WORD_OFFSETS.restArea] =
      input.triangleRestAreas[triangle]!;
  }

  const hingeArenaBase =
    JGS2_CLOTH_GLOBAL_WORDS + triangleCount * JGS2_CLOTH_TRIANGLE_WORDS;
  for (let hinge = 0; hinge < hingeCount; hinge += 1) {
    const recordBase = hingeArenaBase + hinge * JGS2_CLOTH_HINGE_WORDS;
    integers.set(
      input.hingeIndices.subarray(hinge * 4, hinge * 4 + 4),
      recordBase +
        JGS2_CLOTH_HINGE_VEC4_OFFSETS.indices * JGS2_CLOTH_VEC4_WORDS,
    );
    const restBase =
      recordBase +
      JGS2_CLOTH_HINGE_VEC4_OFFSETS.rest * JGS2_CLOTH_VEC4_WORDS;
    floats[restBase + JGS2_CLOTH_HINGE_REST_WORD_OFFSETS.angle] =
      input.hingeRestAngles[hinge]!;
    floats[restBase + JGS2_CLOTH_HINGE_REST_WORD_OFFSETS.edgeLength] =
      input.hingeRestEdgeLengths[hinge]!;
  }

  const incidenceHeader = incidenceWordOffset;
  integers[
    incidenceHeader + JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.magic
  ] = JGS2_CLOTH_INCIDENCE_MAGIC;
  integers[
    incidenceHeader + JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.vertexCount
  ] = input.vertexCount;
  integers[
    incidenceHeader +
      JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.triangleIncidenceCount
  ] = triangleCsr.elements.length;
  integers[
    incidenceHeader +
      JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.hingeIncidenceCount
  ] = hingeCsr.elements.length;
  integers[
    incidenceHeader + JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.triangleRows
  ] = triangleRowsOffset;
  integers[
    incidenceHeader +
      JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.triangleElements
  ] = triangleElementsOffset;
  integers[
    incidenceHeader + JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.hingeRows
  ] = hingeRowsOffset;
  integers[
    incidenceHeader + JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.wordCount
  ] = incidenceWordCount;
  integers.set(triangleCsr.rows, incidenceHeader + triangleRowsOffset);
  integers.set(triangleCsr.elements, incidenceHeader + triangleElementsOffset);
  integers.set(hingeCsr.rows, incidenceHeader + hingeRowsOffset);
  integers.set(hingeCsr.elements, incidenceHeader + hingeElementsOffset);

  return {
    buffer,
    integers,
    floats,
    byteLength,
    triangleCount,
    hingeCount,
    incidenceWordOffset,
    incidenceWordCount,
    planeStressLambda: floats[
      materialBase + JGS2_CLOTH_MATERIAL_WORD_OFFSETS.planeStressLambda
    ]!,
    mu: floats[materialBase + JGS2_CLOTH_MATERIAL_WORD_OFFSETS.mu]!,
  };
}
