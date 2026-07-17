import { describe, expect, it } from "vitest";

import {
  JGS2_MATERIAL_COROTATED_LINEAR,
  JGS2_MATERIAL_STABLE_NEO_HOOKEAN,
  JGS2_VERTEX_OBJECTIVE_BYTES,
  JGS2_VERTEX_OBJECTIVE_WORDS,
  computeJGS2DynamicOffsets,
  inferJGS2MaterialMode,
  inferJGS2BodyCount,
  jgs2TimestepsMatch,
  minimumJGS2InputDeformationDeterminant,
  packJGS2VertexObjectives,
  validateJGS2ContactParameters,
  validateJGS2GpuInput,
  type JGS2GpuInput,
} from "./layout";

function twoTetrahedronGpuInput(): JGS2GpuInput {
  return {
    vertexCount: 5,
    tetCount: 2,
    cubatureK: 0,
    positions: new Float32Array([
      0, 0, 0, 1,
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
      1, 1, 1, 1,
    ]),
    vertexRest: new Float32Array([
      0, 0, 0, 1,
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
      1, 1, 1, 1,
    ]),
    vertexColors: new Float32Array(20),
    vertexInfo: new Uint32Array([
      0, 1, 0, 0,
      1, 2, 0, 0,
      3, 2, 0, 0,
      5, 2, 0, 0,
      7, 1, 0, 0,
    ]),
    tetIndices: new Uint32Array([
      0, 1, 2, 3,
      1, 2, 3, 4,
    ]),
    tetInverseDm: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
    ]),
    tetMeta: new Float32Array([
      1 / 6, 1, 1, JGS2_MATERIAL_COROTATED_LINEAR,
      1 / 6, 1, 1, JGS2_MATERIAL_COROTATED_LINEAR,
    ]),
    tetRestStiffness: new Float32Array(2 * 12 * 12),
    adjacency: new Uint32Array([
      0,
      0, 1,
      0, 1,
      0, 1,
      1,
    ]),
    cubatureTetIds: new Uint32Array(0),
    cubatureWeights: new Float32Array(0),
    cubatureBasis: new Float32Array(0),
  };
}

describe("JGS2 canonical adjacency validation", () => {
  it("accepts one sorted, unique incident-tetrahedron slice per vertex", () => {
    expect(() => validateJGS2GpuInput(twoTetrahedronGpuInput())).not.toThrow();
  });

  it("rejects a duplicate tetrahedron in a vertex slice", () => {
    const input = twoTetrahedronGpuInput();
    input.adjacency[2] = 0;

    expect(() => validateJGS2GpuInput(input)).toThrow(
      /vertex 1 must equal incident tetrahedron 1; got 0/,
    );
  });

  it("rejects a valid tetrahedron id that is not incident to the vertex", () => {
    const input = twoTetrahedronGpuInput();
    input.adjacency[0] = 1;

    expect(() => validateJGS2GpuInput(input)).toThrow(
      /vertex 0 must equal incident tetrahedron 0; got 1/,
    );
  });

  it("rejects a vertex slice that omits an incident tetrahedron", () => {
    const input: JGS2GpuInput = {
      ...twoTetrahedronGpuInput(),
      vertexInfo: new Uint32Array([
        0, 1, 0, 0,
        1, 1, 0, 0,
        2, 2, 0, 0,
        4, 2, 0, 0,
        6, 1, 0, 0,
      ]),
      adjacency: new Uint32Array([0, 0, 0, 1, 0, 1, 1]),
    };

    expect(() => validateJGS2GpuInput(input)).toThrow(
      /vertex 1 must list exactly 2 unique incident tetrahedra; got 1/,
    );
  });

  it("rejects adjacency entries not owned by any canonical vertex slice", () => {
    const source = twoTetrahedronGpuInput();
    const input: JGS2GpuInput = {
      ...source,
      adjacency: new Uint32Array([...source.adjacency, 0]),
    };

    expect(() => validateJGS2GpuInput(input)).toThrow(/1 unused entry/);
  });
});

describe("JGS2 per-vertex objective ABI", () => {
  it("packs zero forces and inactive current-position targets by default", () => {
    const input = twoTetrahedronGpuInput();
    const packed = packJGS2VertexObjectives(input);

    expect(JGS2_VERTEX_OBJECTIVE_WORDS).toBe(8);
    expect(JGS2_VERTEX_OBJECTIVE_BYTES).toBe(32);
    expect(packed.byteLength).toBe(input.vertexCount * 32);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      const base = vertex * 8;
      const position = vertex * 4;
      expect(Array.from(packed.subarray(base, base + 4))).toEqual([
        0, 0, 0, 0,
      ]);
      expect(Array.from(packed.subarray(base + 4, base + 8))).toEqual([
        input.positions[position],
        input.positions[position + 1],
        input.positions[position + 2],
        0,
      ]);
    }
  });

  it("packs explicit xyz forces and isotropic targets into two vec4s", () => {
    const input: JGS2GpuInput = {
      ...twoTetrahedronGpuInput(),
      objectives: {
        externalForces: new Float32Array([
          1, 2, 3,
          0, 0, 0,
          -1, -2, -3,
          0, 4, 0,
          5, 0, 6,
        ]),
        targetPositions: new Float32Array([
          0.1, 0.2, 0.3,
          1.1, 0.2, 0.3,
          0.1, 1.2, 0.3,
          0.1, 0.2, 1.3,
          1.1, 1.2, 1.3,
        ]),
        targetStiffnesses: new Float32Array([7, 0, 8, 9, 10]),
      },
    };

    const packed = packJGS2VertexObjectives(input);

    expect(Array.from(packed.subarray(0, 8))).toEqual([
      1, 2, 3, 0, 0.1, 0.2, 0.3, 7,
    ].map(Math.fround));
    expect(Array.from(packed.subarray(4 * 8, 5 * 8))).toEqual([
      5, 0, 6, 0, 1.1, 1.2, 1.3, 10,
    ].map(Math.fround));
  });

  it("requires paired target arrays with exact per-vertex lengths", () => {
    const input = twoTetrahedronGpuInput();
    expect(() =>
      validateJGS2GpuInput({
        ...input,
        objectives: {
          targetPositions: new Float32Array(input.vertexCount * 3),
        },
      }),
    ).toThrow(/provided together/);
    expect(() =>
      validateJGS2GpuInput({
        ...input,
        objectives: {
          externalForces: new Float32Array(input.vertexCount * 3 - 1),
        },
      }),
    ).toThrow(/must contain 15 values/);
  });

  it("rejects nonfinite, negative, or active pinned objective values", () => {
    const input = twoTetrahedronGpuInput();
    const targetPositions = new Float32Array(input.vertexCount * 3);
    const targetStiffnesses = new Float32Array(input.vertexCount);
    targetStiffnesses[1] = -1;
    expect(() =>
      validateJGS2GpuInput({
        ...input,
        objectives: { targetPositions, targetStiffnesses },
      }),
    ).toThrow(/must be nonnegative/);

    const externalForces = new Float32Array(input.vertexCount * 3);
    externalForces[0] = Number.POSITIVE_INFINITY;
    expect(() =>
      validateJGS2GpuInput({ ...input, objectives: { externalForces } }),
    ).toThrow(/must be finite/);

    const pinnedInfo = input.vertexInfo.slice();
    pinnedInfo[2] = 1;
    externalForces[0] = 1;
    expect(() =>
      validateJGS2GpuInput({
        ...input,
        vertexInfo: pinnedInfo,
        objectives: { externalForces },
      }),
    ).toThrow(/Pinned vertex 0/);
    externalForces[0] = 0;
    targetStiffnesses[0] = 2;
    targetStiffnesses[1] = 0;
    expect(() =>
      validateJGS2GpuInput({
        ...input,
        vertexInfo: pinnedInfo,
        objectives: { targetPositions, targetStiffnesses },
      }),
    ).toThrow(/Pinned vertex 0/);
  });
});

describe("JGS2 timestep compatibility", () => {
  it("accepts values represented by the same GPU f32", () => {
    const timestep = 1 / 60;
    expect(jgs2TimestepsMatch(timestep, timestep)).toBe(true);
    expect(jgs2TimestepsMatch(timestep, Math.fround(timestep))).toBe(true);
  });

  it("rejects a changed or invalid timestep", () => {
    expect(jgs2TimestepsMatch(1 / 60, 1 / 50)).toBe(false);
    expect(jgs2TimestepsMatch(1 / 60, Number.NaN)).toBe(false);
    expect(jgs2TimestepsMatch(1 / 60, 0)).toBe(false);
  });
});

describe("JGS2 per-body bookkeeping", () => {
  it("allocates body corrections after all per-tetrahedron rotations", () => {
    const offsets = computeJGS2DynamicOffsets(2, 1, 3);

    expect(offsets).toEqual({
      posA: 0,
      posB: 2,
      predicted: 4,
      velocity: 6,
      old: 8,
      vertexRotation: 10,
      tetRotation: 16,
      bodyCorrection: 19,
      finalUpdate: 22,
      localGlobalization: 24,
      tetGlobalization: 32,
      assembledVertexEnergy: 34,
      convergenceGradient: 36,
      globalizationControl: 46,
      globalizationHistory: 50,
      vec4Count: 562,
    });
  });

  it("does not allocate stable-globalization scratch for legacy scenes", () => {
    const offsets = computeJGS2DynamicOffsets(2, 1, 3, false);

    expect(offsets.finalUpdate).toBe(22);
    expect(offsets.localGlobalization).toBe(24);
    expect(offsets.tetGlobalization).toBe(24);
    expect(offsets.assembledVertexEnergy).toBe(24);
    expect(offsets.convergenceGradient).toBe(24);
    expect(offsets.globalizationControl).toBe(24);
    expect(offsets.globalizationHistory).toBe(24);
    expect(offsets.vec4Count).toBe(24);
  });

  it("infers body count from packed vertexInfo body ids", () => {
    const vertexInfo = new Uint32Array([
      0, 0, 0, 0,
      0, 0, 0, 1,
      0, 0, 0, 1,
    ]);

    expect(inferJGS2BodyCount(vertexInfo, 3)).toBe(2);
  });

  it("rejects body ids that would allocate more bodies than vertices", () => {
    expect(() =>
      inferJGS2BodyCount(new Uint32Array([0, 0, 0, 1]), 1),
    ).toThrow(/maximum is smaller than vertexCount/);
  });
});

describe("JGS2 contact parameters", () => {
  it("accepts a nonnegative damping rate and contact margin", () => {
    expect(() => validateJGS2ContactParameters(8, 0.01)).not.toThrow();
    expect(() => validateJGS2ContactParameters(0, 0)).not.toThrow();
  });

  it("rejects invalid damping rates or margins", () => {
    expect(() => validateJGS2ContactParameters(-0.01, 0)).toThrow();
    expect(() => validateJGS2ContactParameters(Number.NaN, 0)).toThrow();
    expect(() =>
      validateJGS2ContactParameters(Number.POSITIVE_INFINITY, 0),
    ).toThrow();
    expect(() => validateJGS2ContactParameters(8, -0.01)).toThrow();
  });
});

describe("JGS2 production material mode", () => {
  const tetrahedronInput = {
    vertexCount: 4,
    tetCount: 1,
    positions: new Float32Array([
      0, 0, 0, 1,
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
    ]),
    tetIndices: new Uint32Array([0, 1, 2, 3]),
    tetInverseDm: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
    ]),
    tetMeta: new Float32Array([
      1 / 6,
      1,
      1,
      JGS2_MATERIAL_STABLE_NEO_HOOKEAN,
    ]),
  } as JGS2GpuInput;

  it("classifies homogeneous materials and rejects an unsafe mixed solve", () => {
    expect(inferJGS2MaterialMode(tetrahedronInput)).toBe(
      "stable-neo-hookean",
    );
    expect(
      inferJGS2MaterialMode({
        ...tetrahedronInput,
        tetMeta: new Float32Array([
          1 / 6,
          1,
          1,
          JGS2_MATERIAL_COROTATED_LINEAR,
        ]),
      }),
    ).toBe("corotated-linear");
    expect(() =>
      inferJGS2MaterialMode({
        ...tetrahedronInput,
        tetCount: 2,
        tetMeta: new Float32Array([
          1 / 6, 1, 1, JGS2_MATERIAL_COROTATED_LINEAR,
          1 / 6, 1, 1, JGS2_MATERIAL_STABLE_NEO_HOOKEAN,
        ]),
      }),
    ).toThrow(/cannot mix/i);
  });

  it("computes the uploaded source determinant without padded-column leakage", () => {
    expect(minimumJGS2InputDeformationDeterminant(tetrahedronInput)).toBe(1);
    const compressed = {
      ...tetrahedronInput,
      positions: tetrahedronInput.positions.slice(),
    };
    compressed.positions[4] = 0.25;
    expect(minimumJGS2InputDeformationDeterminant(compressed)).toBe(0.25);
  });
});
