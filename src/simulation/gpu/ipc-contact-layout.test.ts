import { describe, expect, it } from "vitest";

import type { StaticIpcContactCandidates } from "../cpu/ipc-contact";
import {
  IPC_CONTACT_CANDIDATE_BYTES,
  IPC_CONTACT_CANDIDATE_VEC4_OFFSETS,
  IPC_CONTACT_CANDIDATE_WORDS,
  IPC_CONTACT_GLOBAL_BYTES,
  IPC_CONTACT_GLOBAL_WORDS,
  IPC_CONTACT_INCIDENCE_HEADER_WORDS,
  IPC_CONTACT_INCIDENCE_MAGIC,
  IPC_CONTACT_META_WORD_OFFSETS,
  IPC_CONTACT_SCRATCH_WORD_OFFSETS,
  IPC_CONTACT_TYPE_EDGE_EDGE,
  IPC_CONTACT_TYPE_VERTEX_TRIANGLE,
  IPC_CONTACT_VEC4_WORDS,
  packIpcIncidentCandidateAdjacency,
  packIpcContactBuffer,
  validateIpcContactCandidatesForGpu,
} from "./ipc-contact-layout";

function mixedCandidates(): StaticIpcContactCandidates {
  return {
    vertexTriangleCandidates: [[7, 2, 3, 4]],
    edgeEdgeCandidates: [[8, 9, 12, 13]],
    packedIndices: new Uint32Array([7, 2, 3, 4, 8, 9, 12, 13]),
    vertexTriangleCount: 1,
    edgeEdgeCount: 1,
  };
}

function candidateWordBase(candidateIndex: number): number {
  return IPC_CONTACT_GLOBAL_WORDS + candidateIndex * IPC_CONTACT_CANDIDATE_WORDS;
}

describe("IPC GPU contact buffer layout", () => {
  it("packs exact mixed u32/f32 records in VT-then-EE order", () => {
    const packed = packIpcContactBuffer(mixedCandidates());

    expect(packed.buffer).toBe(packed.integers.buffer);
    expect(packed.buffer).toBe(packed.floats.buffer);
    expect(packed.byteLength).toBe(
      IPC_CONTACT_GLOBAL_BYTES + 2 * IPC_CONTACT_CANDIDATE_BYTES,
    );
    expect(packed.candidateCount).toBe(2);
    expect(packed.vertexTriangleCount).toBe(1);
    expect(packed.edgeEdgeCount).toBe(1);
    expect([...packed.integers.subarray(0, IPC_CONTACT_GLOBAL_WORDS)]).toEqual([
      2, 1, 1, 0x3f80_0000,
      0, 0, 0, 0,
    ]);

    const vtBase = candidateWordBase(0);
    expect([...packed.integers.subarray(vtBase, vtBase + 8)]).toEqual([
      7, 2, 3, 4,
      IPC_CONTACT_TYPE_VERTEX_TRIANGLE, 1, 0, 0,
    ]);
    const eeBase = candidateWordBase(1);
    expect([...packed.integers.subarray(eeBase, eeBase + 8)]).toEqual([
      8, 9, 12, 13,
      IPC_CONTACT_TYPE_EDGE_EDGE, 1, 0, 0,
    ]);

    for (const base of [vtBase, eeBase]) {
      const laggedBase =
        base +
        IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.laggedNormal *
          IPC_CONTACT_VEC4_WORDS;
      expect([...packed.floats.subarray(laggedBase, laggedBase + 8)]).toEqual([
        0, 0, 0, 0,
        0, 0, 0, 0,
      ]);
      const scratchBase =
        base +
        IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.scratch * IPC_CONTACT_VEC4_WORDS;
      expect([...packed.floats.subarray(scratchBase, scratchBase + 4)]).toEqual([
        1, 0, 0, 0,
      ]);
      expect(
        packed.integers[
          base +
            IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.meta *
              IPC_CONTACT_VEC4_WORDS +
            IPC_CONTACT_META_WORD_OFFSETS.enabled
        ],
      ).toBe(1);
      expect(
        packed.floats[
          scratchBase + IPC_CONTACT_SCRATCH_WORD_OFFSETS.valid
        ],
      ).toBe(0);
      for (const offset of [
        IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.sourceContact,
        IPC_CONTACT_CANDIDATE_VEC4_OFFSETS.targetContact,
      ]) {
        expect(
          packed.floats[base + offset * IPC_CONTACT_VEC4_WORDS + 3],
        ).toBe(-1);
      }
    }
  });

  it("packs deterministic static and mutable vertex-candidate CSR rows", () => {
    const packed = packIpcIncidentCandidateAdjacency(14, mixedCandidates());
    const rowsBase = IPC_CONTACT_INCIDENCE_HEADER_WORDS;
    const rowOffsets = [0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 8];
    const candidateIdsBase = rowsBase + rowOffsets.length;
    const activeCountsBase = candidateIdsBase + packed.incidenceCount;
    const activeCandidateIdsBase = activeCountsBase + packed.vertexCount;

    expect([...packed.words.subarray(0, rowsBase)]).toEqual([
      IPC_CONTACT_INCIDENCE_MAGIC,
      14,
      2,
      8,
    ]);
    expect([...packed.words.subarray(rowsBase, candidateIdsBase)]).toEqual(
      rowOffsets,
    );
    expect(
      [...packed.words.subarray(candidateIdsBase, activeCountsBase)],
    ).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(
      [...packed.words.subarray(activeCountsBase, activeCandidateIdsBase)],
    ).toEqual(new Array(14).fill(0));
    expect([...packed.words.subarray(activeCandidateIdsBase)]).toEqual(
      new Array(8).fill(0),
    );
    expect(packed.words).toHaveLength(49);
  });

  it("omits an empty suffix and rejects candidate vertices outside the mesh", () => {
    const emptyCandidates: StaticIpcContactCandidates = {
      vertexTriangleCandidates: [],
      edgeEdgeCandidates: [],
      packedIndices: new Uint32Array(),
      vertexTriangleCount: 0,
      edgeEdgeCount: 0,
    };
    expect(packIpcIncidentCandidateAdjacency(14, emptyCandidates)).toEqual({
      words: new Uint32Array(),
      vertexCount: 14,
      candidateCount: 0,
      incidenceCount: 0,
    });
    expect(() =>
      packIpcIncidentCandidateAdjacency(13, mixedCandidates()),
    ).toThrow(/references vertex 13.*vertexCount is 13/i);
  });

  it("always emits the 32-byte global records for an empty candidate set", () => {
    const packed = packIpcContactBuffer({
      vertexTriangleCandidates: [],
      edgeEdgeCandidates: [],
      packedIndices: new Uint32Array(),
      vertexTriangleCount: 0,
      edgeEdgeCount: 0,
    });

    expect(packed.byteLength).toBe(32);
    expect(packed.candidateCount).toBe(0);
    expect([...packed.integers]).toEqual([
      0, 0, 0, 0x3f80_0000,
      0, 0, 0, 0,
    ]);
    expect(packed.integers.buffer).toBe(packed.floats.buffer);
  });

  it("rejects inconsistent counts, packed order, and invalid indices", () => {
    const valid = mixedCandidates();
    expect(() =>
      validateIpcContactCandidatesForGpu({
        ...valid,
        vertexTriangleCount: 2,
      }),
    ).toThrow(/count does not match/i);

    expect(() =>
      validateIpcContactCandidatesForGpu({
        ...valid,
        packedIndices: new Uint32Array([7, 2, 4, 3, 8, 9, 12, 13]),
      }),
    ).toThrow(/disagree/i);

    expect(() =>
      validateIpcContactCandidatesForGpu({
        vertexTriangleCandidates: [[7, 2, 2, 4]],
        edgeEdgeCandidates: [],
        packedIndices: new Uint32Array([7, 2, 2, 4]),
        vertexTriangleCount: 1,
        edgeEdgeCount: 0,
      }),
    ).toThrow(/distinct/i);

    expect(() =>
      validateIpcContactCandidatesForGpu({
        vertexTriangleCandidates: [[-1, 2, 3, 4]],
        edgeEdgeCandidates: [],
        packedIndices: new Uint32Array([0xffff_ffff, 2, 3, 4]),
        vertexTriangleCount: 1,
        edgeEdgeCount: 0,
      }),
    ).toThrow(/unsigned 32-bit/i);
  });
});
