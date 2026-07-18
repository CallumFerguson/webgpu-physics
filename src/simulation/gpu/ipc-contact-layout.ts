import type { StaticIpcContactCandidates } from "../cpu/ipc-contact";

export const IPC_CONTACT_WORD_BYTES = 4;
export const IPC_CONTACT_VEC4_WORDS = 4;
/** Counts/control followed by runtime contact parameters. */
export const IPC_CONTACT_GLOBAL_VEC4S = 2;
export const IPC_CONTACT_CANDIDATE_VEC4S = 9;
export const IPC_CONTACT_GLOBAL_WORDS =
  IPC_CONTACT_GLOBAL_VEC4S * IPC_CONTACT_VEC4_WORDS;
export const IPC_CONTACT_CANDIDATE_WORDS =
  IPC_CONTACT_CANDIDATE_VEC4S * IPC_CONTACT_VEC4_WORDS;
export const IPC_CONTACT_GLOBAL_BYTES =
  IPC_CONTACT_GLOBAL_WORDS * IPC_CONTACT_WORD_BYTES;
export const IPC_CONTACT_CANDIDATE_BYTES =
  IPC_CONTACT_CANDIDATE_WORDS * IPC_CONTACT_WORD_BYTES;

/** u32 fields in the global control vec4. */
export const IPC_CONTACT_GLOBAL_WORD_OFFSETS = {
  candidateCount: 0,
  vertexTriangleCount: 1,
  edgeEdgeCount: 2,
  safeAlpha: 3,
} as const;

/** Vec4 offset of (minimum distance, friction, smoothing, step safety). */
export const IPC_CONTACT_PARAMETERS_VEC4_OFFSET = 1;

/** Vec4 offsets relative to the start of one candidate record. */
export const IPC_CONTACT_CANDIDATE_VEC4_OFFSETS = {
  indices: 0,
  meta: 1,
  laggedNormal: 2,
  laggedWeights: 3,
  scratch: 4,
  sourceContact: 5,
  sourceWeights: 6,
  targetContact: 7,
  targetWeights: 8,
} as const;

/** u32 fields in the candidate meta vec4. */
export const IPC_CONTACT_META_WORD_OFFSETS = {
  type: 0,
  enabled: 1,
  reserved0: 2,
  reserved1: 3,
} as const;

/** f32 fields in the candidate lagged-normal vec4. */
export const IPC_CONTACT_LAGGED_NORMAL_WORD_OFFSETS = {
  x: 0,
  y: 1,
  z: 2,
  normalForce: 3,
} as const;

/** f32 fields in the per-iteration scratch vec4. */
export const IPC_CONTACT_SCRATCH_WORD_OFFSETS = {
  safeAlpha: 0,
  sourceDistance: 1,
  candidateDistance: 2,
  valid: 3,
} as const;

export const IPC_CONTACT_TYPE_VERTEX_TRIANGLE = 0;
export const IPC_CONTACT_TYPE_EDGE_EDGE = 1;
export const IPC_CONTACT_CANDIDATE_ENABLED = 1;

export const IPC_CONTACT_INCIDENCE_MAGIC = 0x4950_4331;
export const IPC_CONTACT_INCIDENCE_HEADER_WORDS = 4;
export const IPC_CONTACT_INCIDENCE_HEADER_OFFSETS = {
  magic: 0,
  vertexCount: 1,
  candidateCount: 2,
  incidenceCount: 3,
} as const;

export interface PackedIpcContactBuffer {
  readonly buffer: ArrayBuffer;
  /** u32 interpretation for indices, counts, and candidate metadata. */
  readonly integers: Uint32Array;
  /** f32 interpretation for lagged state and per-iteration scratch. */
  readonly floats: Float32Array;
  readonly byteLength: number;
  readonly candidateCount: number;
  readonly vertexTriangleCount: number;
  readonly edgeEdgeCount: number;
}

export interface PackedIpcIncidentCandidateAdjacency {
  /** Versioned CSR suffix appended to the tetrahedron adjacency buffer. */
  readonly words: Uint32Array;
  readonly vertexCount: number;
  readonly candidateCount: number;
  readonly incidenceCount: number;
}

const MAX_U32 = 0xffff_ffff;

function requireU32(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 0 || value > MAX_U32) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer.`);
  }
}

function validateCandidateIndices(
  values: readonly number[],
  label: string,
): void {
  if (values.length !== IPC_CONTACT_VEC4_WORDS) {
    throw new RangeError(`${label} must contain exactly four vertex indices.`);
  }
  for (let lane = 0; lane < values.length; lane += 1) {
    requireU32(values[lane]!, `${label}[${lane}]`);
  }
  if (new Set(values).size !== IPC_CONTACT_VEC4_WORDS) {
    throw new RangeError(`${label} must reference four distinct vertices.`);
  }
}

/**
 * Validates the redundant CPU candidate representation before it crosses the
 * GPU ABI. In addition to count/range checks, this guarantees that the packed
 * index stream preserves the documented VT-then-EE tuple order.
 */
export function validateIpcContactCandidatesForGpu(
  candidates: StaticIpcContactCandidates,
): void {
  requireU32(candidates.vertexTriangleCount, "Vertex-triangle candidate count");
  requireU32(candidates.edgeEdgeCount, "Edge-edge candidate count");

  if (
    candidates.vertexTriangleCount !==
    candidates.vertexTriangleCandidates.length
  ) {
    throw new RangeError(
      "Vertex-triangle candidate count does not match its candidate array.",
    );
  }
  if (candidates.edgeEdgeCount !== candidates.edgeEdgeCandidates.length) {
    throw new RangeError(
      "Edge-edge candidate count does not match its candidate array.",
    );
  }

  const candidateCount =
    candidates.vertexTriangleCount + candidates.edgeEdgeCount;
  requireU32(candidateCount, "Total IPC candidate count");
  const expectedPackedWords = candidateCount * IPC_CONTACT_VEC4_WORDS;
  if (
    !(candidates.packedIndices instanceof Uint32Array) ||
    candidates.packedIndices.length !== expectedPackedWords
  ) {
    throw new RangeError(
      `Packed IPC indices must be a Uint32Array with ${expectedPackedWords} words.`,
    );
  }

  const orderedCandidates: readonly (readonly number[])[] = [
    ...candidates.vertexTriangleCandidates,
    ...candidates.edgeEdgeCandidates,
  ];
  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex += 1) {
    const tuple = orderedCandidates[candidateIndex]!;
    const type =
      candidateIndex < candidates.vertexTriangleCount ? "VT" : "EE";
    validateCandidateIndices(tuple, `${type} candidate ${candidateIndex}`);
    const packedBase = candidateIndex * IPC_CONTACT_VEC4_WORDS;
    for (let lane = 0; lane < IPC_CONTACT_VEC4_WORDS; lane += 1) {
      if (candidates.packedIndices[packedBase + lane] !== tuple[lane]) {
        throw new RangeError(
          `Packed IPC indices disagree with ${type} candidate ${candidateIndex} at lane ${lane}.`,
        );
      }
    }
  }
}

/**
 * Packs a deterministic vertex-to-contact-candidate CSR suffix.
 *
 * The existing tetrahedron adjacency remains an unchanged prefix. This
 * versioned suffix lets vertex-local GPU work visit only incident candidates
 * while candidate-parallel IPC passes continue to consume the original list.
 */
export function packIpcIncidentCandidateAdjacency(
  vertexCount: number,
  candidates: StaticIpcContactCandidates,
): PackedIpcIncidentCandidateAdjacency {
  requireU32(vertexCount, "IPC vertex count");
  validateIpcContactCandidatesForGpu(candidates);

  const candidateCount =
    candidates.vertexTriangleCount + candidates.edgeEdgeCount;
  const incidenceCount = candidateCount * IPC_CONTACT_VEC4_WORDS;
  requireU32(incidenceCount, "IPC vertex-candidate incidence count");

  if (candidateCount === 0) {
    return {
      words: new Uint32Array(0),
      vertexCount,
      candidateCount,
      incidenceCount,
    };
  }

  const rowOffsetWords = vertexCount + 1;
  requireU32(rowOffsetWords, "IPC vertex-candidate row-offset count");
  // The mutable tail mirrors the static row capacities: one active count per
  // vertex followed by up to one active id for every static incidence.
  const activeScratchWords = vertexCount + incidenceCount;
  requireU32(activeScratchWords, "IPC active-incidence scratch word count");
  const packedWordCount =
    IPC_CONTACT_INCIDENCE_HEADER_WORDS +
    rowOffsetWords +
    incidenceCount +
    activeScratchWords;
  requireU32(packedWordCount, "Packed IPC incidence word count");

  const words = new Uint32Array(packedWordCount);
  words[IPC_CONTACT_INCIDENCE_HEADER_OFFSETS.magic] =
    IPC_CONTACT_INCIDENCE_MAGIC;
  words[IPC_CONTACT_INCIDENCE_HEADER_OFFSETS.vertexCount] = vertexCount;
  words[IPC_CONTACT_INCIDENCE_HEADER_OFFSETS.candidateCount] = candidateCount;
  words[IPC_CONTACT_INCIDENCE_HEADER_OFFSETS.incidenceCount] = incidenceCount;

  const rowOffsetsBase = IPC_CONTACT_INCIDENCE_HEADER_WORDS;
  const candidateIdsBase = rowOffsetsBase + rowOffsetWords;
  for (const vertex of candidates.packedIndices) {
    if (vertex >= vertexCount) {
      throw new RangeError(
        `IPC candidate references vertex ${vertex}, but vertexCount is ${vertexCount}.`,
      );
    }
    words[rowOffsetsBase + vertex + 1] += 1;
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    words[rowOffsetsBase + vertex + 1] += words[rowOffsetsBase + vertex]!;
  }

  const cursors = words.slice(
    rowOffsetsBase,
    rowOffsetsBase + vertexCount,
  );
  for (let candidate = 0; candidate < candidateCount; candidate += 1) {
    const packedBase = candidate * IPC_CONTACT_VEC4_WORDS;
    for (let lane = 0; lane < IPC_CONTACT_VEC4_WORDS; lane += 1) {
      const vertex = candidates.packedIndices[packedBase + lane]!;
      words[candidateIdsBase + cursors[vertex]!] = candidate;
      cursors[vertex] += 1;
    }
  }

  return { words, vertexCount, candidateCount, incidenceCount };
}

/**
 * Packs static IPC candidates into a storage-buffer-ready mixed u32/f32 ABI.
 *
 * Word layout:
 * - global u32 vec4: total count, VT count, EE count, safe-step alpha;
 * - global f32 vec4: minimum distance, friction, smoothing, step safety;
 * - nine vec4s per candidate: indices, metadata, lagged normal/force,
 *   lagged weights, per-iteration scratch, and compact source/target contact
 *   caches (normal/distance plus closest-feature weights).
 *
 * Lagged records start at zero. Scratch safe-alpha starts at the neutral value
 * one while its valid flag remains zero; GPU contact passes replace the scratch
 * vec4 before consuming it.
 */
export function packIpcContactBuffer(
  candidates: StaticIpcContactCandidates,
): PackedIpcContactBuffer {
  validateIpcContactCandidatesForGpu(candidates);

  const vertexTriangleCount = candidates.vertexTriangleCount;
  const edgeEdgeCount = candidates.edgeEdgeCount;
  const candidateCount = vertexTriangleCount + edgeEdgeCount;
  const byteLength =
    IPC_CONTACT_GLOBAL_BYTES + candidateCount * IPC_CONTACT_CANDIDATE_BYTES;
  const buffer = new ArrayBuffer(byteLength);
  const integers = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);

  integers[IPC_CONTACT_GLOBAL_WORD_OFFSETS.candidateCount] = candidateCount;
  integers[IPC_CONTACT_GLOBAL_WORD_OFFSETS.vertexTriangleCount] =
    vertexTriangleCount;
  integers[IPC_CONTACT_GLOBAL_WORD_OFFSETS.edgeEdgeCount] = edgeEdgeCount;
  floats[IPC_CONTACT_GLOBAL_WORD_OFFSETS.safeAlpha] = 1;

  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex += 1) {
    const recordBase =
      IPC_CONTACT_GLOBAL_WORDS + candidateIndex * IPC_CONTACT_CANDIDATE_WORDS;
    const indicesBase =
      recordBase +
      IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.indices * IPC_CONTACT_VEC4_WORDS;
    integers.set(
      candidates.packedIndices.subarray(
        candidateIndex * IPC_CONTACT_VEC4_WORDS,
        (candidateIndex + 1) * IPC_CONTACT_VEC4_WORDS,
      ),
      indicesBase,
    );

    const metaBase =
      recordBase +
      IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.meta * IPC_CONTACT_VEC4_WORDS;
    integers[metaBase + IPC_CONTACT_META_WORD_OFFSETS.type] =
      candidateIndex < vertexTriangleCount
        ? IPC_CONTACT_TYPE_VERTEX_TRIANGLE
        : IPC_CONTACT_TYPE_EDGE_EDGE;
    integers[metaBase + IPC_CONTACT_META_WORD_OFFSETS.enabled] =
      IPC_CONTACT_CANDIDATE_ENABLED;

    const scratchBase =
      recordBase +
      IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.scratch * IPC_CONTACT_VEC4_WORDS;
    floats[scratchBase + IPC_CONTACT_SCRATCH_WORD_OFFSETS.safeAlpha] = 1;
    floats[
      recordBase +
        IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.sourceContact *
          IPC_CONTACT_VEC4_WORDS +
        3
    ] = -1;
    floats[
      recordBase +
        IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.targetContact *
          IPC_CONTACT_VEC4_WORDS +
        3
    ] = -1;
  }

  return {
    buffer,
    integers,
    floats,
    byteLength,
    candidateCount,
    vertexTriangleCount,
    edgeEdgeCount,
  };
}
