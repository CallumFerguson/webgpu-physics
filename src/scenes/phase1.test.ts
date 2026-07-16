import { describe, expect, it } from "vitest";

import {
  buildPrecomputedScene,
  computeStableNeoHookeanParameters,
} from "../simulation/cpu";
import {
  JGS2_MATERIAL_COROTATED_LINEAR,
  JGS2_MATERIAL_STABLE_NEO_HOOKEAN,
  validateJGS2GpuInput,
} from "../simulation/gpu";
import { toJGS2GpuInput } from "./index";
import {
  PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED,
  PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID,
  PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
  PHASE1_STABLE_NEO_HOOKEAN_MAXIMUM_SEEDED_DETERMINANT,
  PHASE1_STABLE_NEO_HOOKEAN_MINIMUM_SEEDED_DETERMINANT,
  PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS,
  PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX,
  buildPhase1StableNeoHookeanOracleDefinition,
  generatePhase1StableNeoHookeanPoseCorpus,
} from "./phase1";

function multiplyTranspose(
  matrix: ArrayLike<number>,
): Float64Array {
  const result = new Float64Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let inner = 0; inner < 3; inner += 1) {
        result[row * 3 + column] +=
          matrix[row * 3 + inner]! * matrix[column * 3 + inner]!;
      }
    }
  }
  return result;
}

describe("frozen Phase 1 stable Neo-Hookean oracle corpus", () => {
  it("P1-MANIFEST-001 builds the private explicit single-tetrahedron fixture", () => {
    const definition = buildPhase1StableNeoHookeanOracleDefinition();

    expect(definition.id).toBe(PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID);
    expect(definition.mesh.positions).toEqual(
      PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS,
    );
    expect([...definition.mesh.tetrahedra]).toEqual([0, 1, 2, 3]);
    expect([...definition.mesh.fixed]).toEqual([1, 0, 0, 0]);
    expect(definition.materials).toHaveLength(1);
    expect(definition.materials[0]).toMatchObject({
      name: "phase1-stable-neo-hookean-reference",
      model: "stable-neo-hookean",
      density: 1_000,
      youngModulus: 80_000,
      poissonRatio: 0.3,
    });
    expect(definition.settings).toMatchObject({
      timestep: 1 / 60,
      gravity: [0, 0, 0],
      solverIterations: 1,
      cubatureSamples: 4,
    });
  });

  it("P1-MANIFEST-002 deterministically covers rigid, compressed, sheared, stretched, and seeded states", () => {
    const first = generatePhase1StableNeoHookeanPoseCorpus();
    const second = generatePhase1StableNeoHookeanPoseCorpus();

    expect(PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED).toBe(1_048_583);
    expect(first).toEqual(second);
    expect(first).toHaveLength(PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT);
    expect(new Set(first.map((state) => state.id))).toHaveLength(
      PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT,
    );
    expect(first.map((state) => state.kind).slice(0, 9)).toEqual([
      "rest",
      "rigid",
      "rigid",
      "rigid",
      "compression",
      "compression",
      "compression",
      "shear",
      "stretch",
    ]);
    expect(first.filter((state) => state.kind === "seeded")).toHaveLength(
      PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT -
        PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX,
    );
    expect(first.slice(4, 7).map((state) => state.determinant)).toEqual(
      PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS,
    );

    for (const state of first.slice(1, 4)) {
      const product = multiplyTranspose(state.deformationGradient);
      for (const [index, value] of product.entries()) {
        expect(value).toBeCloseTo(index % 4 === 0 ? 1 : 0, 12);
      }
      expect(state.determinant).toBeCloseTo(1, 12);
    }
    for (const state of first.slice(PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX)) {
      expect(state.kind).toBe("seeded");
      expect(state.determinant).toBeGreaterThanOrEqual(
        PHASE1_STABLE_NEO_HOOKEAN_MINIMUM_SEEDED_DETERMINANT,
      );
      expect(state.determinant).toBeLessThanOrEqual(
        PHASE1_STABLE_NEO_HOOKEAN_MAXIMUM_SEEDED_DETERMINANT,
      );
    }
    for (const [index, state] of first.entries()) {
      expect(state.index).toBe(index);
      expect(state.id).toBe(
        `${PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID}/${index
          .toString()
          .padStart(2, "0")}`,
      );
      expect(state.determinant).toBeGreaterThan(0);
      expect([...state.positions].every(Number.isFinite)).toBe(true);
      expect([...state.deformationGradient].every(Number.isFinite)).toBe(true);
      expect([...state.positions.slice(0, 3)]).toEqual([0, 0, 0]);
    }
  });

  it("P1-MATERIAL-ABI-001 packs and validates the stable material tag and converted constants", () => {
    const definition = buildPhase1StableNeoHookeanOracleDefinition();
    const scene = buildPrecomputedScene(definition);
    const input = toJGS2GpuInput(scene);
    const parameters = computeStableNeoHookeanParameters(
      PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
    );

    expect(input.tetMeta).toEqual(
      new Float32Array([
        1 / 6,
        parameters.lambda,
        parameters.mu,
        JGS2_MATERIAL_STABLE_NEO_HOOKEAN,
      ]),
    );
    expect(() => validateJGS2GpuInput(input)).not.toThrow();

    const invalid = {
      ...input,
      tetMeta: input.tetMeta.slice(),
    };
    invalid.tetMeta[3] = 17;
    expect(() => validateJGS2GpuInput(invalid)).toThrow(
      /unknown material model/,
    );

    const zeroLambdaCorotated = {
      ...input,
      tetMeta: new Float32Array([
        input.tetMeta[0]!,
        0,
        1,
        JGS2_MATERIAL_COROTATED_LINEAR,
      ]),
    };
    expect(() => validateJGS2GpuInput(zeroLambdaCorotated)).not.toThrow();
    zeroLambdaCorotated.tetMeta[1] = -1;
    expect(() => validateJGS2GpuInput(zeroLambdaCorotated)).toThrow(
      /positive bulk modulus/,
    );

    const invalidDefinition = {
      ...definition,
      materials: [
        {
          ...PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
          model: "stable-neo-hooken",
        },
      ],
    };
    expect(() => buildPrecomputedScene(invalidDefinition as never)).toThrow(
      /invalid model/,
    );
    const invalidPrecomputedScene = {
      ...scene,
      materials: invalidDefinition.materials,
    };
    expect(() => toJGS2GpuInput(invalidPrecomputedScene as never)).toThrow(
      /unknown model stable-neo-hooken/,
    );
  });
});
