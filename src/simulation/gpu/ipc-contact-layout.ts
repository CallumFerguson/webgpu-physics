import type { StaticIpcContactCandidates } from "../cpu/ipc-contact";

export const IPC_CONTACT_WORD_BYTES = 4;
export const IPC_CONTACT_VEC4_WORDS = 4;
/** Counts/control followed by runtime contact parameters. */
export const IPC_CONTACT_GLOBAL_VEC4S = 2;
export const IPC_CONTACT_CANDIDATE_VEC4S = 5;
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
 * Packs static IPC candidates into a storage-buffer-ready mixed u32/f32 ABI.
 *
 * Word layout:
 * - global u32 vec4: total count, VT count, EE count, safe-step alpha;
 * - global f32 vec4: minimum distance, friction, smoothing, step safety;
 * - five vec4s per candidate: indices, metadata, lagged normal/force,
 *   lagged weights, and per-iteration scratch.
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
