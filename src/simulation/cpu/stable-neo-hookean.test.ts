import { describe, expect, it } from "vitest";

import {
  centralDifferenceGradient,
  centralDifferenceHessian,
  relativeError,
} from "./finite-difference";
import { multiply3 } from "./math";
import {
  buildPhase0OracleFixture,
  generatePhase0OraclePoseCorpus,
} from "./phase0-canonical";
import {
  applyStableNeoHookeanTangent,
  assertStableNeoHookeanFeasible,
  computeStableNeoHookeanParameters,
  createStableNeoHookeanMaterialTerm,
  evaluateStableNeoHookeanDensity,
  evaluateStableNeoHookeanMesh,
  evaluateStableNeoHookeanTetrahedron,
} from "./stable-neo-hookean";

const MATERIAL = {
  name: "reference stable solid",
  density: 1_000,
  youngModulus: 80_000,
  poissonRatio: 0.3,
  color: [0.2, 0.7, 0.9, 1] as const,
};

function transformedPositions(
  restPositions: Float64Array,
  transform: Float64Array,
  translation: readonly [number, number, number],
): Float64Array {
  const result = new Float64Array(restPositions.length);
  for (let vertex = 0; vertex < restPositions.length / 3; vertex += 1) {
    const x = restPositions[vertex * 3]!;
    const y = restPositions[vertex * 3 + 1]!;
    const z = restPositions[vertex * 3 + 2]!;
    for (let row = 0; row < 3; row += 1) {
      result[vertex * 3 + row] =
        transform[row * 3]! * x +
        transform[row * 3 + 1]! * y +
        transform[row * 3 + 2]! * z +
        translation[row]!;
    }
  }
  return result;
}

describe("stable Neo-Hookean density", () => {
  it("reparameterizes Lamé constants and reproduces linear Hooke response", () => {
    const parameters = computeStableNeoHookeanParameters(MATERIAL);
    const expectedLameMu =
      MATERIAL.youngModulus / (2 * (1 + MATERIAL.poissonRatio));
    const expectedLameLambda =
      (MATERIAL.youngModulus * MATERIAL.poissonRatio) /
      ((1 + MATERIAL.poissonRatio) * (1 - 2 * MATERIAL.poissonRatio));
    expect(parameters.lameMu).toBeCloseTo(expectedLameMu, 12);
    expect(parameters.lameLambda).toBeCloseTo(expectedLameLambda, 12);
    expect(parameters.mu).toBeCloseTo((4 / 3) * expectedLameMu, 12);
    expect(parameters.lambda).toBeCloseTo(
      expectedLameLambda + (5 / 6) * expectedLameMu,
      12,
    );
    expect(parameters.alpha).toBeCloseTo(
      1 + parameters.mu / parameters.lambda - parameters.mu / (4 * parameters.lambda),
      14,
    );

    const identity = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const direction = new Float64Array([
      0.2, -0.3, 0.1,
      0.4, -0.1, 0.25,
      -0.2, 0.15, 0.35,
    ]);
    const response = applyStableNeoHookeanTangent(
      identity,
      direction,
      parameters,
    );
    const trace = direction[0]! + direction[4]! + direction[8]!;
    const expected = new Float64Array(9);
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        expected[row * 3 + column] =
          expectedLameMu *
            (direction[row * 3 + column]! + direction[column * 3 + row]!) +
          (row === column ? expectedLameLambda * trace : 0);
      }
    }
    expect(relativeError(response, expected)).toBeLessThan(1e-12);
  });

  it("has zero rest response and is objective under rigid rotation", () => {
    const parameters = computeStableNeoHookeanParameters(MATERIAL);
    const identity = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const angle = 0.63;
    const rotation = new Float64Array([
      Math.cos(angle), -Math.sin(angle), 0,
      Math.sin(angle), Math.cos(angle), 0,
      0, 0, 1,
    ]);
    for (const [label, deformation] of [
      ["rest", identity],
      ["rotation", rotation],
    ] as const) {
      const evaluation = evaluateStableNeoHookeanDensity(
        deformation,
        parameters,
      );
      expect(Math.abs(evaluation.energyDensity), `${label} energy`).toBeLessThan(
        1e-10,
      );
      expect(
        Math.hypot(...evaluation.firstPiola),
        `${label} first Piola`,
      ).toBeLessThan(1e-10);
      expect(evaluation.deformationDeterminant).toBeCloseTo(1, 14);
    }
  });

  it("remains finite through collapse/inversion while feasibility rejects them", () => {
    const parameters = computeStableNeoHookeanParameters(MATERIAL);
    for (const determinant of [0.5, 0.1, 0.01]) {
      const evaluation = evaluateStableNeoHookeanDensity(
        new Float64Array([determinant, 0, 0, 0, 1, 0, 0, 0, 1]),
        parameters,
      );
      expect(evaluation.deformationDeterminant).toBeCloseTo(determinant, 14);
      expect(Number.isFinite(evaluation.energyDensity)).toBe(true);
      expect([...evaluation.firstPiola].every(Number.isFinite)).toBe(true);
      expect([...evaluation.tangent].every(Number.isFinite)).toBe(true);
    }
    for (const determinant of [0, -0.1, -1]) {
      const deformation = new Float64Array([
        determinant, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]);
      const evaluation = evaluateStableNeoHookeanDensity(
        deformation,
        parameters,
      );
      expect(Number.isFinite(evaluation.energyDensity)).toBe(true);
      expect([...evaluation.firstPiola].every(Number.isFinite)).toBe(true);
      expect([...evaluation.tangent].every(Number.isFinite)).toBe(true);
      expect(() => assertStableNeoHookeanFeasible(deformation)).toThrow(
        /feasibility floor/,
      );
    }
    expect(
      assertStableNeoHookeanFeasible(
        new Float64Array([0.01, 0, 0, 0, 1, 0, 0, 0, 1]),
        1e-4,
      ),
    ).toBeCloseTo(0.01, 14);
  });

  it("returns a symmetric density tangent", () => {
    const evaluation = evaluateStableNeoHookeanDensity(
      new Float64Array([
        1.08, 0.17, -0.04,
        0.03, 0.84, 0.12,
        -0.09, 0.05, 1.14,
      ]),
      computeStableNeoHookeanParameters(MATERIAL),
    );
    let maximumAsymmetry = 0;
    for (let row = 0; row < 9; row += 1) {
      for (let column = 0; column < 9; column += 1) {
        maximumAsymmetry = Math.max(
          maximumAsymmetry,
          Math.abs(
            evaluation.tangent[row * 9 + column]! -
              evaluation.tangent[column * 9 + row]!,
          ),
        );
      }
    }
    expect(maximumAsymmetry).toBeLessThan(1e-10);
  });

  it("rotates energy, stress, and tangent actions objectively", () => {
    const parameters = computeStableNeoHookeanParameters(MATERIAL);
    const deformation = new Float64Array([
      1.08, 0.17, -0.04,
      0.03, 0.84, 0.12,
      -0.09, 0.05, 1.14,
    ]);
    const direction = new Float64Array([
      0.07, -0.02, 0.11,
      -0.04, 0.03, 0.08,
      0.05, -0.09, -0.01,
    ]);
    const angle = -0.52;
    const rotation = new Float64Array([
      1, 0, 0,
      0, Math.cos(angle), -Math.sin(angle),
      0, Math.sin(angle), Math.cos(angle),
    ]);
    const rotatedDeformation = multiply3(rotation, deformation);
    const rotatedDirection = multiply3(rotation, direction);
    const base = evaluateStableNeoHookeanDensity(deformation, parameters);
    const rotated = evaluateStableNeoHookeanDensity(
      rotatedDeformation,
      parameters,
    );
    expect(Math.abs(base.energyDensity - rotated.energyDensity)).toBeLessThan(
      1e-10,
    );
    expect(
      relativeError(rotated.firstPiola, multiply3(rotation, base.firstPiola)),
    ).toBeLessThan(1e-12);
    const baseAction = applyStableNeoHookeanTangent(
      deformation,
      direction,
      parameters,
    );
    const rotatedAction = applyStableNeoHookeanTangent(
      rotatedDeformation,
      rotatedDirection,
      parameters,
    );
    expect(
      relativeError(rotatedAction, multiply3(rotation, baseAction)),
    ).toBeLessThan(1e-12);
  });
});

describe("stable Neo-Hookean tetrahedral reference", () => {
  it("matches the existing linear rest stiffness exactly", () => {
    const fixture = buildPhase0OracleFixture();
    const evaluation = evaluateStableNeoHookeanTetrahedron(
      fixture.mesh,
      fixture.restData,
      fixture.definition.materials,
      0,
      fixture.mesh.positions,
    );
    expect(Math.abs(evaluation.energy)).toBeLessThan(1e-10);
    expect(Math.hypot(...evaluation.gradient)).toBeLessThan(1e-10);
    expect(
      relativeError(
        evaluation.hessian,
        fixture.restData.stiffnessMatrices.subarray(0, 144),
      ),
    ).toBeLessThan(1e-12);
  });

  it("is objective for a translated rigid tetrahedron", () => {
    const fixture = buildPhase0OracleFixture();
    const angle = 0.47;
    const rotation = new Float64Array([
      Math.cos(angle), 0, Math.sin(angle),
      0, 1, 0,
      -Math.sin(angle), 0, Math.cos(angle),
    ]);
    const positions = transformedPositions(
      fixture.mesh.positions,
      rotation,
      [1.7, -0.4, 0.8],
    );
    const evaluation = evaluateStableNeoHookeanMesh(
      fixture.mesh,
      fixture.restData,
      fixture.definition.materials,
      positions,
    );
    expect(Math.abs(evaluation.energy)).toBeLessThan(1e-10);
    expect(Math.hypot(...evaluation.gradient)).toBeLessThan(1e-9);
    expect(evaluation.deformationDeterminants[0]).toBeCloseTo(1, 13);
  });

  it("matches finite differences on every deformed canonical pose", () => {
    const fixture = buildPhase0OracleFixture();
    const poses = generatePhase0OraclePoseCorpus().filter(
      (pose) => pose.kind === "deformed",
    );
    let worstGradient = 0;
    let worstHessian = 0;
    for (const pose of poses) {
      const energy = (positions: Float64Array): number =>
        evaluateStableNeoHookeanMesh(
          fixture.mesh,
          fixture.restData,
          fixture.definition.materials,
          positions,
        ).energy;
      const gradient = (positions: Float64Array): Float64Array =>
        evaluateStableNeoHookeanMesh(
          fixture.mesh,
          fixture.restData,
          fixture.definition.materials,
          positions,
        ).gradient;
      const analytic = evaluateStableNeoHookeanMesh(
        fixture.mesh,
        fixture.restData,
        fixture.definition.materials,
        pose.positions,
      );
      const gradientError = relativeError(
        analytic.gradient,
        centralDifferenceGradient(energy, pose.positions),
      );
      const hessianError = relativeError(
        analytic.hessian,
        centralDifferenceHessian(gradient, pose.positions),
      );
      worstGradient = Math.max(worstGradient, gradientError);
      worstHessian = Math.max(worstHessian, hessianError);
      expect(gradientError, `${pose.id} gradient`).toBeLessThanOrEqual(1e-5);
      expect(hessianError, `${pose.id} Hessian`).toBeLessThanOrEqual(1e-4);
    }
    console.log(
      `Stable Neo-Hookean CPU corpus (${poses.length} poses): ` +
        `gradient=${worstGradient.toExponential(3)}, ` +
        `Hessian=${worstHessian.toExponential(3)}`,
    );
  }, 30_000);

  it("preserves internal force, torque, and translational Hessian null modes", () => {
    const fixture = buildPhase0OracleFixture();
    const pose = generatePhase0OraclePoseCorpus().find(
      (candidate) => candidate.kind === "deformed",
    )!;
    const evaluation = evaluateStableNeoHookeanMesh(
      fixture.mesh,
      fixture.restData,
      fixture.definition.materials,
      pose.positions,
    );
    const forceSum = new Float64Array(3);
    const torqueSum = new Float64Array(3);
    for (let vertex = 0; vertex < pose.positions.length / 3; vertex += 1) {
      const x = pose.positions[vertex * 3]!;
      const y = pose.positions[vertex * 3 + 1]!;
      const z = pose.positions[vertex * 3 + 2]!;
      const fx = evaluation.gradient[vertex * 3]!;
      const fy = evaluation.gradient[vertex * 3 + 1]!;
      const fz = evaluation.gradient[vertex * 3 + 2]!;
      forceSum[0] += fx;
      forceSum[1] += fy;
      forceSum[2] += fz;
      torqueSum[0] += y * fz - z * fy;
      torqueSum[1] += z * fx - x * fz;
      torqueSum[2] += x * fy - y * fx;
    }
    const gradientScale = Math.max(1, Math.hypot(...evaluation.gradient));
    expect(Math.hypot(...forceSum) / gradientScale).toBeLessThan(1e-12);
    expect(Math.hypot(...torqueSum) / gradientScale).toBeLessThan(1e-12);

    const dimension = evaluation.gradient.length;
    const hessianScale = Math.max(1, Math.hypot(...evaluation.hessian));
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const response = new Float64Array(dimension);
      for (let row = 0; row < dimension; row += 1) {
        for (let vertex = 0; vertex < dimension / 3; vertex += 1) {
          response[row] +=
            evaluation.hessian[row * dimension + vertex * 3 + coordinate]!;
        }
      }
      expect(Math.hypot(...response) / hessianScale).toBeLessThan(1e-12);
    }
  });

  it("implements the shared material-energy contract", () => {
    const fixture = buildPhase0OracleFixture();
    const pose = generatePhase0OraclePoseCorpus().find(
      (candidate) => candidate.kind === "deformed",
    )!;
    const term = createStableNeoHookeanMaterialTerm({
      id: "material.stable-neo-hookean",
      mesh: fixture.mesh,
      restData: fixture.restData,
      materials: fixture.definition.materials,
    });
    const direct = evaluateStableNeoHookeanMesh(
      fixture.mesh,
      fixture.restData,
      fixture.definition.materials,
      pose.positions,
    );
    const throughContract = term.evaluate(pose.positions);
    expect(term.kind).toBe("material");
    expect(term.dimension).toBe(12);
    expect(throughContract.energy).toBe(direct.energy);
    expect(throughContract.gradient).toEqual(direct.gradient);
    expect(throughContract.hessian).toEqual(direct.hessian);
  });
});
