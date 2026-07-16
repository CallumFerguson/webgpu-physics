import { describe, expect, it } from "vitest";

import {
  centralDifferenceGradient,
  centralDifferenceHessian,
  relativeError,
} from "./finite-difference";
import {
  assembleRestLinearSystem,
  computeLumpedMasses,
  computeRestTetraData,
} from "./fem";
import { determinant3 } from "./math";
import { generateRegularCuboidMesh } from "./mesh";
import {
  createStableNeoHookeanImplicitEulerOracle,
  type StableNeoHookeanImplicitEulerOptions,
} from "./nonlinear-oracle";
import {
  activeCoordinatesFromFullPositions,
  fullPositionsFromActiveCoordinates,
} from "./oracle";
import { evaluateStableNeoHookeanMesh } from "./stable-neo-hookean";
import type {
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

const TIMESTEP = 1 / 60;
const MATERIAL: LinearMaterial = {
  name: "nonlinear-oracle-solid",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 80_000,
  poissonRatio: 0.3,
  color: [0.2, 0.7, 0.9, 1],
};

interface Fixture {
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
}

function buildFixture(mesh: TetrahedralMesh): Fixture {
  const restData = computeRestTetraData(mesh, [MATERIAL]);
  const lumpedMasses = computeLumpedMasses(mesh, [MATERIAL], restData);
  const restSystem = assembleRestLinearSystem(
    mesh,
    restData,
    lumpedMasses,
    TIMESTEP,
  );
  return { mesh, restData, lumpedMasses, restSystem };
}

function singleTetrahedronFixture(): Fixture {
  return buildFixture({
    positions: new Float64Array([
      0, 0, 0,
      1, 0, 0,
      0.15, 0.9, 0,
      0.1, 0.2, 0.85,
    ]),
    tetrahedra: new Uint32Array([0, 1, 2, 3]),
    materialIds: new Uint16Array([0]),
    fixed: new Uint8Array(4),
    bodyIds: new Uint16Array(4),
  });
}

function createOracle(
  fixture: Fixture,
  overrides: Partial<StableNeoHookeanImplicitEulerOptions> = {},
) {
  return createStableNeoHookeanImplicitEulerOracle({
    mesh: fixture.mesh,
    restData: fixture.restData,
    materials: [MATERIAL],
    lumpedMasses: fixture.lumpedMasses,
    restSystem: fixture.restSystem,
    timestep: TIMESTEP,
    predictedPositions: fixture.mesh.positions,
    ...overrides,
  });
}

function transformPositions(
  positions: Float64Array,
  transform: Float64Array,
  translation: readonly [number, number, number],
): Float64Array {
  const result = new Float64Array(positions.length);
  for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
    for (let row = 0; row < 3; row += 1) {
      result[vertex * 3 + row] =
        transform[row * 3]! * positions[vertex * 3]! +
        transform[row * 3 + 1]! * positions[vertex * 3 + 1]! +
        transform[row * 3 + 2]! * positions[vertex * 3 + 2]! +
        translation[row]!;
    }
  }
  return result;
}

describe("stable Neo-Hookean implicit-Euler oracle", () => {
  it("has exact zero rest inertia and rest-normalized material response", () => {
    const fixture = singleTetrahedronFixture();
    const oracle = createOracle(fixture);
    const coordinates = activeCoordinatesFromFullPositions(
      fixture.mesh.positions,
      fixture.restSystem,
    );
    const evaluation = oracle.evaluate(coordinates);

    expect(Math.abs(evaluation.energy)).toBeLessThan(1e-10);
    expect(Math.hypot(...evaluation.gradient)).toBeLessThan(1e-9);
    expect(evaluation.components.inertia).toBe(0);
    expect(Math.abs(evaluation.components.material)).toBeLessThan(1e-10);
    expect(evaluation.components.externalForce).toBe(0);
    expect(evaluation.components.quadraticTargets).toBe(0);
    expect(evaluation.minimumDeformationDeterminant).toBeCloseTo(1, 14);
  });

  it("preserves a translated rigid pose when that pose is the inertial target", () => {
    const fixture = singleTetrahedronFixture();
    const angle = 0.61;
    const rotation = new Float64Array([
      Math.cos(angle), -Math.sin(angle), 0,
      Math.sin(angle), Math.cos(angle), 0,
      0, 0, 1,
    ]);
    const rigidPositions = transformPositions(
      fixture.mesh.positions,
      rotation,
      [1.4, -0.7, 0.35],
    );
    const oracle = createOracle(fixture, {
      predictedPositions: rigidPositions,
    });
    const coordinates = activeCoordinatesFromFullPositions(
      rigidPositions,
      fixture.restSystem,
    );
    const evaluation = oracle.evaluate(coordinates);

    expect(Math.abs(evaluation.energy)).toBeLessThan(1e-9);
    expect(Math.hypot(...evaluation.gradient)).toBeLessThan(1e-8);
    expect(evaluation.components.inertia).toBe(0);
    expect(Math.abs(evaluation.components.material)).toBeLessThan(1e-9);
    expect(evaluation.deformationDeterminants[0]).toBeCloseTo(1, 13);
  });

  it("matches central differences with inertia, force, and multiple targets", () => {
    const fixture = buildFixture(
      generateRegularCuboidMesh({
        cells: [1, 1, 1],
        fixed: (_position, gridCoordinate) => gridCoordinate[0] === 0,
      }),
    );
    const coordinates = activeCoordinatesFromFullPositions(
      fixture.mesh.positions,
      fixture.restSystem,
    );
    for (let index = 0; index < coordinates.length; index += 1) {
      coordinates[index] += 0.025 * Math.sin(index * 1.7 + 0.3);
    }
    const externalForce = Float64Array.from(
      coordinates,
      (_value, index) => 0.7 * Math.cos(index * 0.8),
    );
    const targetA = coordinates.slice();
    const targetB = coordinates.slice();
    for (let index = 0; index < coordinates.length; index += 1) {
      targetA[index] += 0.01 * Math.cos(index + 0.2);
      targetB[index] -= 0.015 * Math.sin(index + 0.4);
    }
    const oracle = createOracle(fixture, {
      predictedPositions: fixture.mesh.positions.map(
        (value, index) => value + 0.005 * Math.sin(index * 0.4),
      ),
      externalForce,
      quadraticTargets: [
        {
          id: "handle.primary",
          target: targetA,
          stiffness: Float64Array.from(
            coordinates,
            (_value, index) => 20 + index,
          ),
        },
        {
          id: "handle.secondary",
          target: targetB,
          stiffness: Float64Array.from(
            coordinates,
            (_value, index) => (index % 3 === 1 ? 35 : 0),
          ),
        },
      ],
    });
    const analytic = oracle.evaluate(coordinates);
    const finiteGradient = centralDifferenceGradient(oracle.energy, coordinates);
    const finiteHessian = centralDifferenceHessian(oracle.gradient, coordinates);

    expect(relativeError(analytic.gradient, finiteGradient)).toBeLessThan(1e-7);
    expect(relativeError(analytic.hessian, finiteHessian)).toBeLessThan(1e-6);
  });

  it("reports exact all-element material data and additive components", () => {
    const fixture = buildFixture(
      generateRegularCuboidMesh({
        cells: [1, 1, 1],
        fixed: (_position, gridCoordinate) => gridCoordinate[1] === 0,
      }),
    );
    const coordinates = activeCoordinatesFromFullPositions(
      fixture.mesh.positions,
      fixture.restSystem,
    );
    for (let index = 0; index < coordinates.length; index += 1) {
      coordinates[index] += 0.04 * Math.sin(0.6 + index * 0.9);
    }
    const predictedPositions = fixture.mesh.positions.slice();
    const externalForce = Float64Array.from(
      coordinates,
      (_value, index) => index - 2.5,
    );
    const target = coordinates.map((value, index) => value + 0.02 * (index + 1));
    const stiffness = Float64Array.from(
      coordinates,
      (_value, index) => 3 + index,
    );
    const oracle = createOracle(fixture, {
      predictedPositions,
      externalForce,
      quadraticTargets: [{ id: "constraint.scripted", target, stiffness }],
    });
    const evaluation = oracle.evaluate(coordinates);
    const positions = fullPositionsFromActiveCoordinates(
      coordinates,
      fixture.mesh,
      fixture.restSystem,
    );
    const material = evaluateStableNeoHookeanMesh(
      fixture.mesh,
      fixture.restData,
      [MATERIAL],
      positions,
    );
    let expectedInertia = 0;
    for (const vertex of fixture.restSystem.activeVertices) {
      const inertia = fixture.lumpedMasses[vertex]! / (TIMESTEP * TIMESTEP);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const difference =
          positions[vertex * 3 + coordinate]! -
          predictedPositions[vertex * 3 + coordinate]!;
        expectedInertia += 0.5 * inertia * difference * difference;
      }
    }
    let expectedExternal = 0;
    let expectedTarget = 0;
    for (let coordinate = 0; coordinate < coordinates.length; coordinate += 1) {
      expectedExternal -= externalForce[coordinate]! * coordinates[coordinate]!;
      const difference = coordinates[coordinate]! - target[coordinate]!;
      expectedTarget += 0.5 * stiffness[coordinate]! * difference * difference;
    }

    expect(evaluation.components.inertia).toBeCloseTo(expectedInertia, 10);
    expect(evaluation.components.material).toBe(material.energy);
    expect(evaluation.components.externalForce).toBeCloseTo(expectedExternal, 12);
    expect(evaluation.components.quadraticTargets).toBeCloseTo(expectedTarget, 12);
    expect(evaluation.energy).toBeCloseTo(
      expectedInertia + material.energy + expectedExternal + expectedTarget,
      10,
    );
    expect(evaluation.quadraticTargetEnergies).toEqual([
      { id: "constraint.scripted", energy: evaluation.components.quadraticTargets },
    ]);
    expect(evaluation.tetrahedronEnergies).toEqual(material.tetrahedronEnergies);
    expect(evaluation.deformationDeterminants).toEqual(
      material.deformationDeterminants,
    );
    expect(evaluation.deformationDeterminants).toHaveLength(6);
    expect(evaluation.minimumDeformationDeterminant).toBe(
      Math.min(...material.deformationDeterminants),
    );
  });

  it("reports the exact determinant for an affine deformation", () => {
    const fixture = singleTetrahedronFixture();
    const deformation = new Float64Array([
      1.1, 0.12, -0.03,
      0.04, 0.83, 0.09,
      -0.02, 0.06, 1.17,
    ]);
    const positions = transformPositions(
      fixture.mesh.positions,
      deformation,
      [0.2, -0.1, 0.3],
    );
    const oracle = createOracle(fixture, { predictedPositions: positions });
    const evaluation = oracle.evaluate(
      activeCoordinatesFromFullPositions(positions, fixture.restSystem),
    );
    const expectedDeterminant = determinant3(deformation);

    expect(evaluation.deformationDeterminants[0]).toBeCloseTo(
      expectedDeterminant,
      13,
    );
    expect(evaluation.minimumDeformationDeterminant).toBeCloseTo(
      expectedDeterminant,
      13,
    );
  });

  it("rejects a material that does not explicitly select the stable model", () => {
    const fixture = singleTetrahedronFixture();
    expect(() =>
      createOracle(fixture, {
        materials: [{ ...MATERIAL, model: undefined }],
      }),
    ).toThrow(/must declare model stable-neo-hookean/);
  });
});
