import { describe, expect, it } from "vitest";
import {
  JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
  JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR,
  PHASE1_ACCEPTED_DETERMINANT_FLOOR,
  computeJGS2LocalDescentDirection,
  globalizeExactStableNeoHookeanJGS2Local,
  lineSearchRestrictedJGS2LocalDirection,
  minimumSymmetricEigenvalue3,
} from "./nonlinear-globalization";
import {
  assembleRestLinearSystem,
  computeLumpedMasses,
  computeRestTetraData,
} from "./fem";
import { createExactStableNeoHookeanJGS2LocalModel } from "./jgs2-local";
import { generateRegularCuboidMesh } from "./mesh";
import { createStableNeoHookeanImplicitEulerOracle } from "./nonlinear-oracle";
import {
  activeCoordinatesFromFullPositions,
  fullPositionsFromActiveCoordinates,
} from "./oracle";
import { computeExactVertexBasis, exactBasisToActiveMatrix } from "./precompute";
import { minimumTetrahedralDeformationDeterminant } from "./stable-neo-hookean";
import type { LinearMaterial } from "./types";

const TIMESTEP = 1 / 60;
const MATERIAL: LinearMaterial = {
  name: "globalization-reference-solid",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 80_000,
  poissonRatio: 0.3,
  color: [0.2, 0.7, 0.9, 1],
};

describe("Phase 1 nonlinear JGS2 globalization reference", () => {
  it("computes the minimum eigenvalue of diagonal and coupled symmetric systems", () => {
    expect(
      minimumSymmetricEigenvalue3(new Float64Array([3, 0, 0, 0, -2, 0, 0, 0, 5])),
    ).toBe(-2);

    // Q diag(-0.0005, 2, 4) Q^T for a 45-degree xy rotation.
    const minimum = -5e-4;
    const maximum = 2;
    const mean = 0.5 * (minimum + maximum);
    const offDiagonal = 0.5 * (minimum - maximum);
    const coupled = new Float64Array([
      mean, offDiagonal, 0,
      offDiagonal, mean, 0,
      0, 0, 4,
    ]);
    expect(minimumSymmetricEigenvalue3(coupled)).toBeCloseTo(minimum, 12);

    const asymmetric = computeJGS2LocalDescentDirection(
      new Float64Array([
        2, 0.2, 0,
        0.1, 3, 0,
        0, 0, 4,
      ]),
      new Float64Array([1, 0, 0]),
      { inertiaScale: 1 },
    );
    expect(asymmetric.maximumRelativeAsymmetry).toBeCloseTo(0.025, 12);
  });

  it("provides CPU reference coverage for capped shifting and descent", () => {
    const result = computeJGS2LocalDescentDirection(
      new Float64Array([
        -5e-4, 0, 0,
        0, 1, 0,
        0, 0, 0.75,
      ]),
      new Float64Array([1, -2, 0.5]),
      { inertiaScale: 1 },
    );

    expect(result.accepted).toBe(true);
    expect(result.minimumEigenvalue).toBeCloseTo(-5e-4, 12);
    expect(result.diagonalShift).toBeCloseTo(
      5e-4 + JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR,
      12,
    );
    expect(result.normalizedShift).toBeLessThanOrEqual(
      JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
    );
    expect(result.gradientDotDirection).toBeLessThan(0);
    expect(result.linearResidual).toBeLessThan(1e-10);
  });

  it("rejects rather than silently raising the normalized shift cap", () => {
    const result = computeJGS2LocalDescentDirection(
      new Float64Array([
        -2e-3, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]),
      new Float64Array([1, 0, 0]),
      { inertiaScale: 1 },
    );

    expect(result.status).toBe("shift-limit-exceeded");
    expect(result.accepted).toBe(false);
    expect(result.normalizedShift).toBeGreaterThan(
      JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
    );
    expect([...result.direction]).toEqual([0, 0, 0]);

    const exactCap = computeJGS2LocalDescentDirection(
      new Float64Array([
        JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR -
          JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
        0, 0,
        0, 1, 0,
        0, 0, 1,
      ]),
      new Float64Array([1, 0, 0]),
      { inertiaScale: 1 },
    );
    expect(exactCap.accepted).toBe(true);
    expect(exactCap.normalizedShift).toBeCloseTo(
      JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
      14,
    );
  });

  it("keeps the prescribed shift and solve scale invariant", () => {
    let referenceDirection: Float64Array | undefined;
    let referenceNormalizedShift: number | undefined;
    for (const scale of [1e-12, 1, 1e12]) {
      const result = computeJGS2LocalDescentDirection(
        Float64Array.from([
          -5e-4, 0, 0,
          0, 1, 0,
          0, 0, 0.75,
        ], (value) => scale * value),
        Float64Array.from([1, -2, 0.5], (value) => scale * value),
        { inertiaScale: scale },
      );
      expect(result.accepted).toBe(true);
      expect(result.linearResidual).toBeLessThan(1e-9);
      if (!referenceDirection) {
        referenceDirection = result.direction;
        referenceNormalizedShift = result.normalizedShift;
      } else {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          expect(result.direction[coordinate]).toBeCloseTo(
            referenceDirection[coordinate]!,
            8,
          );
        }
        expect(result.normalizedShift).toBeCloseTo(
          referenceNormalizedShift!,
          12,
        );
      }
    }
  });

  it("accepts a feasible Armijo reference step and records its bound", () => {
    const result = lineSearchRestrictedJGS2LocalDirection({
      initialEnergy: 0,
      gradient: new Float64Array([-1, 0, 0]),
      direction: new Float64Array([1, 0, 0]),
      minimumDeformationDeterminant: () => 0.5,
      energy: (alpha) => -alpha + 0.25 * alpha * alpha,
    });

    expect(result.status).toBe("accepted");
    expect(result.alpha).toBe(1);
    expect(result.acceptedEnergy).toBeLessThanOrEqual(result.armijoBound);
    expect(result.minimumDeformationDeterminant).toBeGreaterThan(
      PHASE1_ACCEPTED_DETERMINANT_FLOOR,
    );
    expect([...result.step]).toEqual([1, 0, 0]);
  });

  it("backtracks past an infeasible reference trial before energy evaluation", () => {
    let energyEvaluationCount = 0;
    const result = lineSearchRestrictedJGS2LocalDirection({
      initialEnergy: 0,
      gradient: new Float64Array([-1, 0, 0]),
      direction: new Float64Array([1, 0, 0]),
      minimumDeformationDeterminant: (alpha) =>
        alpha > 0.5 ? PHASE1_ACCEPTED_DETERMINANT_FLOOR : 0.25,
      energy: (alpha) => {
        energyEvaluationCount += 1;
        return -0.5 * alpha;
      },
    });

    expect(result.accepted).toBe(true);
    expect(result.alpha).toBe(0.5);
    expect(result.backtrackCount).toBe(1);
    expect(result.infeasibleTrialCount).toBe(1);
    expect(result.energyEvaluationCount).toBe(1);
    expect(energyEvaluationCount).toBe(1);
    expect(result.minimumDeformationDeterminant).toBe(0.25);
  });

  it("performs multiple Armijo backtracks when feasible energy requires them", () => {
    const result = lineSearchRestrictedJGS2LocalDirection({
      initialEnergy: 0,
      gradient: new Float64Array([-1, 0, 0]),
      direction: new Float64Array([1, 0, 0]),
      minimumDeformationDeterminant: () => 1,
      energy: (alpha) => (alpha > 0.25 ? alpha : -0.1),
    });
    expect(result.accepted).toBe(true);
    expect(result.alpha).toBe(0.25);
    expect(result.backtrackCount).toBe(2);
    expect(result.energyEvaluationCount).toBe(3);
  });

  it("returns a zero update when no finite feasible Armijo trial exists", () => {
    const result = lineSearchRestrictedJGS2LocalDirection({
      initialEnergy: 2,
      gradient: new Float64Array([-1, 0, 0]),
      direction: new Float64Array([1, 0, 0]),
      minimumDeformationDeterminant: (alpha) => (alpha === 0.5 ? 0 : 0.5),
      energy: (alpha) => (alpha === 1 ? Number.NaN : 3),
    });

    expect(result.status).toBe("no-acceptable-step");
    expect(result.accepted).toBe(false);
    expect(result.evaluatedTrialCount).toBe(13);
    expect(result.nonFiniteTrialCount).toBe(1);
    expect(result.infeasibleTrialCount).toBe(1);
    expect(result.minimumDeformationDeterminantValid).toBe(true);
    expect(result.acceptedEnergy).toBe(result.initialEnergy);
    expect([...result.step]).toEqual([0, 0, 0]);
  });

  it("rejects zero and non-descent directions without evaluating a trial", () => {
    let determinantEvaluations = 0;
    let energyEvaluations = 0;
    const minimumDeformationDeterminant = () => {
      determinantEvaluations += 1;
      return 1;
    };
    const energy = () => {
      energyEvaluations += 1;
      return 0;
    };
    expect(
      lineSearchRestrictedJGS2LocalDirection({
        initialEnergy: 0,
        gradient: new Float64Array([1, 0, 0]),
        direction: new Float64Array([0, 0, 0]),
        minimumDeformationDeterminant,
        energy,
      }).status,
    ).toBe("zero-direction");
    expect(
      lineSearchRestrictedJGS2LocalDirection({
        initialEnergy: 0,
        gradient: new Float64Array([1, 0, 0]),
        direction: new Float64Array([1, 0, 0]),
        minimumDeformationDeterminant,
        energy,
      }).status,
    ).toBe("non-descent-direction");
    expect(determinantEvaluations).toBe(0);
    expect(energyEvaluations).toBe(0);
  });

  it("validates fixed algorithm inputs instead of silently weakening them", () => {
    expect(() =>
      computeJGS2LocalDescentDirection(
        new Float64Array(9),
        new Float64Array(3),
        { inertiaScale: 0 },
      ),
    ).toThrow(/inertia scale must be positive/i);
    expect(() =>
      computeJGS2LocalDescentDirection(
        new Float64Array([Number.POSITIVE_INFINITY, 0, 0, 0, 1, 0, 0, 0, 1]),
        new Float64Array(3),
        { inertiaScale: 1 },
      ),
    ).toThrow(/local hessian must be finite/i);
  });

  it("globalizes the exact stable Neo-Hookean local oracle with feasible Armijo energy", () => {
    const mesh = generateRegularCuboidMesh({
      cells: [1, 1, 1],
      fixed: (_position, [x]) => x === 0,
    });
    const restData = computeRestTetraData(mesh, [MATERIAL]);
    const lumpedMasses = computeLumpedMasses(mesh, [MATERIAL], restData);
    const restSystem = assembleRestLinearSystem(
      mesh,
      restData,
      lumpedMasses,
      TIMESTEP,
    );
    const coordinates = activeCoordinatesFromFullPositions(
      mesh.positions,
      restSystem,
    );
    for (let coordinate = 0; coordinate < coordinates.length; coordinate += 1) {
      coordinates[coordinate] += 0.015 * Math.sin(0.3 + coordinate * 0.7);
    }
    const oracle = createStableNeoHookeanImplicitEulerOracle({
      mesh,
      restData,
      materials: [MATERIAL],
      lumpedMasses,
      restSystem,
      timestep: TIMESTEP,
      predictedPositions: mesh.positions,
    });
    const vertex = restSystem.activeVertices.at(-1)!;
    const equilibriumBasis = exactBasisToActiveMatrix(
      computeExactVertexBasis(mesh, restSystem, vertex).basis,
      restSystem,
    );
    const localModel = createExactStableNeoHookeanJGS2LocalModel({
      oracle,
      restSystem,
      vertex,
      coordinates,
      equilibriumBasis,
    });

    const minimumDeformationDeterminant = (local: Float64Array) => {
      const active = coordinates.slice();
      for (let row = 0; row < active.length; row += 1) {
        for (let column = 0; column < 3; column += 1) {
          active[row] += equilibriumBasis[row * 3 + column]! * local[column]!;
        }
      }
      return minimumTetrahedralDeformationDeterminant(
        mesh,
        restData,
        fullPositionsFromActiveCoordinates(active, mesh, restSystem),
      );
    };

    const result = globalizeExactStableNeoHookeanJGS2Local({
      model: localModel,
      inertiaScale: lumpedMasses[vertex]! / (TIMESTEP * TIMESTEP),
      minimumDeformationDeterminant,
    });

    expect(result.direction.accepted).toBe(true);
    expect(result.direction.gradientDotDirection).toBeLessThan(0);
    expect(result.direction.normalizedShift).toBeLessThanOrEqual(
      JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
    );
    expect(result.direction.linearResidual).toBeLessThan(1e-10);
    expect(result.lineSearch?.accepted).toBe(true);
    expect(result.lineSearch!.acceptedEnergy).toBeLessThanOrEqual(
      result.lineSearch!.armijoBound,
    );
    expect(result.lineSearch!.minimumDeformationDeterminant).toBeGreaterThan(
      PHASE1_ACCEPTED_DETERMINANT_FLOOR,
    );

    for (const sourceMinimum of [
      PHASE1_ACCEPTED_DETERMINANT_FLOOR,
      0,
      -0.1,
      Number.NaN,
    ]) {
      expect(() =>
        globalizeExactStableNeoHookeanJGS2Local({
          model: localModel,
          inertiaScale: lumpedMasses[vertex]! / (TIMESTEP * TIMESTEP),
          minimumDeformationDeterminant: () => sourceMinimum,
        }),
      ).toThrow(/source pose must have minimum deformation determinant/i);
    }
  });
});
