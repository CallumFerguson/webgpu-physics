import { describe, expect, it } from "vitest";
import {
  JGS2_CLOTH_GLOBAL_BYTES,
  JGS2_CLOTH_GLOBAL_WORDS,
  JGS2_CLOTH_HINGE_BYTES,
  JGS2_CLOTH_HINGE_REST_WORD_OFFSETS,
  JGS2_CLOTH_HINGE_VEC4_OFFSETS,
  JGS2_CLOTH_HINGE_WORDS,
  JGS2_CLOTH_INCIDENCE_HEADER_WORDS,
  JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS,
  JGS2_CLOTH_INCIDENCE_MAGIC,
  JGS2_CLOTH_MATERIAL_WORD_OFFSETS,
  JGS2_CLOTH_TRIANGLE_AREA_WORD_OFFSETS,
  JGS2_CLOTH_TRIANGLE_BYTES,
  JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS,
  JGS2_CLOTH_TRIANGLE_WORDS,
  JGS2_CLOTH_VEC4_WORDS,
  packJGS2GpuClothArena,
  validateJGS2GpuClothInput,
  type JGS2GpuClothInput,
} from "./cloth-layout";

function floatBits(...values: number[]): number[] {
  return [...new Uint32Array(new Float32Array(values).buffer)];
}

function fixture(): JGS2GpuClothInput {
  return {
    vertexCount: 5,
    triangleIndices: new Uint32Array([0, 1, 2, 1, 0, 3]),
    triangleInverseRestBases: new Float32Array([
      1, -0.25, 0, 1.25,
      -1, 0.5, 0.25, 2,
    ]),
    triangleRestAreas: new Float32Array([0.4, 0.75]),
    hingeIndices: new Uint32Array([0, 1, 2, 3]),
    hingeRestAngles: new Float32Array([-0.125]),
    hingeRestEdgeLengths: new Float32Array([1.5]),
    youngModulus: 1_200,
    poissonRatio: 0.25,
    thickness: 0.125,
    bendingStiffness: 3.5,
  };
}

describe("GPU cloth arena layout", () => {
  it("packs the exact mixed u32/f32 global, triangle, and hinge ABI", () => {
    const packed = packJGS2GpuClothArena(fixture());
    expect(packed.buffer).toBe(packed.integers.buffer);
    expect(packed.buffer).toBe(packed.floats.buffer);
    expect(packed.triangleCount).toBe(2);
    expect(packed.hingeCount).toBe(1);
    expect(packed.byteLength).toBe(
      JGS2_CLOTH_GLOBAL_BYTES +
        2 * JGS2_CLOTH_TRIANGLE_BYTES +
        JGS2_CLOTH_HINGE_BYTES +
        32 * 4,
    );
    expect([...packed.integers.subarray(0, JGS2_CLOTH_GLOBAL_WORDS)]).toEqual([
      2, 1, 0, ...floatBits(1, 320, 480, 0.125, 3.5),
    ]);
    expect(packed.planeStressLambda).toBe(320);
    expect(packed.mu).toBe(480);

    const triangle0 = JGS2_CLOTH_GLOBAL_WORDS;
    expect([...packed.integers.subarray(triangle0, triangle0 + 4)]).toEqual([
      0, 1, 2, 0,
    ]);
    expect(
      [...packed.floats.subarray(triangle0 + 4, triangle0 + 16)],
    ).toEqual([
      1, -0.25, 0, 1.25,
      Math.fround(0.4), 0, 0, 0,
      0, 0, 0, 0,
    ]);

    const triangle1 = triangle0 + JGS2_CLOTH_TRIANGLE_WORDS;
    expect([...packed.integers.subarray(triangle1, triangle1 + 4)]).toEqual([
      1, 0, 3, 0,
    ]);
    expect(
      packed.floats[
        triangle1 +
          JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS.areaAttributes *
            JGS2_CLOTH_VEC4_WORDS +
          JGS2_CLOTH_TRIANGLE_AREA_WORD_OFFSETS.restArea
      ],
    ).toBe(0.75);
    expect(
      [...packed.floats.subarray(
        triangle1 +
          JGS2_CLOTH_TRIANGLE_VEC4_OFFSETS.scratch * JGS2_CLOTH_VEC4_WORDS,
        triangle1 + JGS2_CLOTH_TRIANGLE_WORDS,
      )],
    ).toEqual([0, 0, 0, 0]);

    const hinge = triangle0 + 2 * JGS2_CLOTH_TRIANGLE_WORDS;
    expect([...packed.integers.subarray(hinge, hinge + 4)]).toEqual([
      0, 1, 2, 3,
    ]);
    const hingeRest =
      hinge +
      JGS2_CLOTH_HINGE_VEC4_OFFSETS.rest * JGS2_CLOTH_VEC4_WORDS;
    expect([
      packed.floats[
        hingeRest + JGS2_CLOTH_HINGE_REST_WORD_OFFSETS.angle
      ],
      packed.floats[
        hingeRest + JGS2_CLOTH_HINGE_REST_WORD_OFFSETS.edgeLength
      ],
    ]).toEqual([-0.125, 1.5]);
    expect(hinge + JGS2_CLOTH_HINGE_WORDS).toBe(
      packed.incidenceWordOffset,
    );
    expect(
      packed.floats[
        JGS2_CLOTH_VEC4_WORDS +
          JGS2_CLOTH_MATERIAL_WORD_OFFSETS.bendingStiffness
      ],
    ).toBe(3.5);

    const incidence = packed.incidenceWordOffset;
    expect(packed.incidenceWordCount).toBe(32);
    expect(
      [...packed.integers.subarray(
        incidence,
        incidence + JGS2_CLOTH_INCIDENCE_HEADER_WORDS,
      )],
    ).toEqual([
      JGS2_CLOTH_INCIDENCE_MAGIC,
      5,
      6,
      4,
      8,
      14,
      20,
      32,
    ]);
    expect(
      [...packed.integers.subarray(incidence + 8, incidence + 14)],
    ).toEqual([0, 2, 4, 5, 6, 6]);
    expect(
      [...packed.integers.subarray(incidence + 14, incidence + 20)],
    ).toEqual([0, 1, 0, 1, 0, 1]);
    expect(
      [...packed.integers.subarray(incidence + 20, incidence + 26)],
    ).toEqual([0, 1, 2, 3, 4, 4]);
    expect(
      [...packed.integers.subarray(incidence + 26, incidence + 30)],
    ).toEqual([0, 0, 0, 0]);
    expect(
      [...packed.integers.subarray(incidence + 30, incidence + 32)],
    ).toEqual([0, 0]);
    expect(incidence + packed.incidenceWordCount).toBe(
      packed.integers.length,
    );
  });

  it("emits empty CSR rows and aligned padding for empty topology", () => {
    const input = fixture();
    const packed = packJGS2GpuClothArena({
      ...input,
      vertexCount: 0,
      triangleIndices: new Uint32Array(),
      triangleInverseRestBases: new Float32Array(),
      triangleRestAreas: new Float32Array(),
      hingeIndices: new Uint32Array(),
      hingeRestAngles: new Float32Array(),
      hingeRestEdgeLengths: new Float32Array(),
    });
    expect(packed.byteLength).toBe(80);
    expect(packed.triangleCount).toBe(0);
    expect(packed.hingeCount).toBe(0);
    expect([...packed.integers.subarray(0, 4)]).toEqual([
      0, 0, 0, floatBits(1)[0],
    ]);
    expect(packed.incidenceWordOffset).toBe(JGS2_CLOTH_GLOBAL_WORDS);
    expect(packed.incidenceWordCount).toBe(12);
    const incidence = packed.incidenceWordOffset;
    expect(
      [...packed.integers.subarray(
        incidence,
        incidence + JGS2_CLOTH_INCIDENCE_HEADER_WORDS,
      )],
    ).toEqual([
      JGS2_CLOTH_INCIDENCE_MAGIC,
      0,
      0,
      0,
      JGS2_CLOTH_INCIDENCE_HEADER_WORDS,
      JGS2_CLOTH_INCIDENCE_HEADER_WORDS + 1,
      JGS2_CLOTH_INCIDENCE_HEADER_WORDS + 1,
      12,
    ]);
    expect(
      packed.integers[
        incidence + JGS2_CLOTH_INCIDENCE_HEADER_WORD_OFFSETS.wordCount
      ],
    ).toBe(12);
    expect(
      [...packed.integers.subarray(
        incidence + JGS2_CLOTH_INCIDENCE_HEADER_WORDS,
      )],
    ).toEqual([0, 0, 0, 0]);
  });

  it("rejects count, index, rest-data, and material violations", () => {
    const valid = fixture();
    expect(() =>
      validateJGS2GpuClothInput({
        ...valid,
        triangleInverseRestBases: new Float32Array(4),
      }),
    ).toThrow(/8 values/i);
    expect(() =>
      validateJGS2GpuClothInput({
        ...valid,
        hingeIndices: new Uint32Array([0, 1, 2, 2]),
      }),
    ).toThrow(/distinct/i);
    expect(() =>
      validateJGS2GpuClothInput({
        ...valid,
        triangleIndices: new Uint32Array([0, 1, 5]),
        triangleInverseRestBases: new Float32Array([1, 0, 0, 1]),
        triangleRestAreas: new Float32Array([0.5]),
      }),
    ).toThrow(/outside vertexCount/i);
    expect(() =>
      validateJGS2GpuClothInput({
        ...valid,
        triangleRestAreas: new Float32Array([0.4, 0]),
      }),
    ).toThrow(/positive/i);
    expect(() =>
      validateJGS2GpuClothInput({
        ...valid,
        triangleInverseRestBases: new Float32Array([
          1, 2, 2, 4,
          -1, 0.5, 0.25, 2,
        ]),
      }),
    ).toThrow(/invertible/i);
    expect(() =>
      validateJGS2GpuClothInput({ ...valid, poissonRatio: 0.5 }),
    ).toThrow(/Poisson/i);
    expect(() =>
      validateJGS2GpuClothInput({ ...valid, bendingStiffness: -1 }),
    ).toThrow(/nonnegative/i);
  });
});
