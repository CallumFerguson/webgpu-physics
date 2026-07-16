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
import {
  createExactStableNeoHookeanJGS2LocalModel,
  evaluateExactStableNeoHookeanJGS2Local,
  projectDenseJGS2LocalTerm,
} from "./jgs2-local";
import { generateRegularCuboidMesh } from "./mesh";
import { createStableNeoHookeanImplicitEulerOracle } from "./nonlinear-oracle";
import { activeCoordinatesFromFullPositions } from "./oracle";
import {
  computeExactVertexBasis,
  exactBasisToActiveMatrix,
} from "./precompute";
import type { LinearMaterial } from "./types";

const TIMESTEP = 1 / 60;
const MATERIAL: LinearMaterial = {
  name: "jgs2-local-solid",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 80_000,
  poissonRatio: 0.3,
  color: [0.2, 0.7, 0.9, 1],
};

function independentProjection(
  gradient: Float64Array,
  hessian: Float64Array,
  basis: Float64Array,
): { gradient: Float64Array; hessian: Float64Array } {
  const dimension = gradient.length;
  const reducedGradient = new Float64Array(3);
  const reducedHessian = new Float64Array(9);
  for (let reducedRow = 0; reducedRow < 3; reducedRow += 1) {
    for (let row = 0; row < dimension; row += 1) {
      reducedGradient[reducedRow] +=
        basis[row * 3 + reducedRow]! * gradient[row]!;
    }
    for (let reducedColumn = 0; reducedColumn < 3; reducedColumn += 1) {
      let value = 0;
      for (let row = 0; row < dimension; row += 1) {
        for (let column = 0; column < dimension; column += 1) {
          value +=
            basis[row * 3 + reducedRow]! *
            hessian[row * dimension + column]! *
            basis[column * 3 + reducedColumn]!;
        }
      }
      reducedHessian[reducedRow * 3 + reducedColumn] = value;
    }
  }
  return { gradient: reducedGradient, hessian: reducedHessian };
}

function add(left: Float64Array, right: Float64Array): Float64Array {
  return Float64Array.from(left, (value, index) => value + right[index]!);
}

describe("exact stable Neo-Hookean JGS2 local reference", () => {
  it("matches an independent dense projection and Newton residual", () => {
    const dimension = 6;
    const localBase = 3;
    const basis = new Float64Array([
      0.35, -0.2, 0.1,
      -0.15, 0.4, 0.25,
      0.2, 0.05, -0.3,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const factor = new Float64Array([
      1.2, 0, 0, 0, 0, 0,
      -0.2, 1.1, 0, 0, 0, 0,
      0.1, 0.3, 1.3, 0, 0, 0,
      0.4, -0.1, 0.2, 1.5, 0, 0,
      -0.3, 0.2, 0.1, 0.25, 1.4, 0,
      0.15, 0.05, -0.2, 0.1, 0.3, 1.6,
    ]);
    const hessian = new Float64Array(dimension * dimension);
    for (let row = 0; row < dimension; row += 1) {
      for (let column = 0; column < dimension; column += 1) {
        for (let inner = 0; inner < dimension; inner += 1) {
          hessian[row * dimension + column] +=
            factor[inner * dimension + row]! *
            factor[inner * dimension + column]!;
        }
      }
    }
    const gradient = new Float64Array([0.7, -1.1, 0.35, 1.4, -0.6, 0.8]);

    const projection = projectDenseJGS2LocalTerm(
      { gradient, hessian },
      basis,
      localBase,
    );
    const independent = independentProjection(gradient, hessian, basis);

    expect(relativeError(projection.gradient, independent.gradient)).toBeLessThan(
      1e-14,
    );
    expect(relativeError(projection.hessian, independent.hessian)).toBeLessThan(
      1e-14,
    );
    expect(projection.sourceGradient).toEqual(gradient.slice(3, 6));
    expect(
      relativeError(
        add(projection.sourceGradient, projection.remainderGradient),
        projection.gradient,
      ),
    ).toBeLessThan(1e-14);
    expect(
      relativeError(
        add(projection.sourceHessian, projection.remainderHessian),
        projection.hessian,
      ),
    ).toBeLessThan(1e-14);

    const zeroEnergyOracle = {
      dimension,
      evaluate: (_coordinates: Float64Array) => ({
        energy: 0,
        gradient: add(gradient, new Float64Array(dimension)),
        hessian: hessian.slice(),
        components: {
          inertia: 0,
          material: 0,
          externalForce: 0,
          quadraticTargets: 0,
        },
        quadraticTargetEnergies: [],
        tetrahedronEnergies: new Float64Array(0),
        deformationDeterminants: new Float64Array(0),
        minimumDeformationDeterminant: 1,
      }),
      energy: (_coordinates: Float64Array) => 0,
      gradient: (_coordinates: Float64Array) => gradient.slice(),
      hessian: (_coordinates: Float64Array) => hessian.slice(),
    };
    const restSystem = {
      dimension,
      activeVertices: new Uint32Array([0, 1]),
      vertexToActiveDof: new Int32Array([0, 3]),
      hessian: hessian.slice(),
      choleskyLower: factor.slice(),
    };
    const local = evaluateExactStableNeoHookeanJGS2Local({
      oracle: zeroEnergyOracle,
      restSystem,
      vertex: 1,
      coordinates: new Float64Array(dimension),
      equilibriumBasis: basis,
    });
    expect(local.newtonUpdate).toBeDefined();
    expect(local.newtonResidualNorm).toBeLessThan(1e-12);
  });

  it("reconstructs incident-source subtraction term by term", () => {
    const dimension = 6;
    const localBase = 0;
    const basis = new Float64Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      0.4, -0.1, 0.2,
      -0.2, 0.3, 0.1,
      0.15, 0.05, -0.25,
    ]);
    const incidentGradient = new Float64Array([1, 2, -1, 0.5, -0.2, 0.3]);
    const nonIncidentGradient = new Float64Array([0, 0, 0, -0.4, 0.7, 0.1]);
    const incidentHessian = new Float64Array(dimension * dimension);
    const nonIncidentHessian = new Float64Array(dimension * dimension);
    for (let row = 0; row < dimension; row += 1) {
      for (let column = 0; column < dimension; column += 1) {
        incidentHessian[row * dimension + column] =
          row === column ? 2 + row * 0.1 : 0.03 * (row + column + 1);
        if (row >= 3 && column >= 3) {
          nonIncidentHessian[row * dimension + column] =
            row === column ? 1.5 : -0.04 * (row + column - 4);
        }
      }
    }

    const incident = projectDenseJGS2LocalTerm(
      { gradient: incidentGradient, hessian: incidentHessian },
      basis,
      localBase,
    );
    const nonIncident = projectDenseJGS2LocalTerm(
      { gradient: nonIncidentGradient, hessian: nonIncidentHessian },
      basis,
      localBase,
    );
    const total = projectDenseJGS2LocalTerm(
      {
        gradient: add(incidentGradient, nonIncidentGradient),
        hessian: add(incidentHessian, nonIncidentHessian),
      },
      basis,
      localBase,
    );

    expect(Math.hypot(...nonIncident.sourceGradient)).toBe(0);
    expect(Math.hypot(...nonIncident.sourceHessian)).toBe(0);
    expect(
      relativeError(
        add(
          total.sourceGradient,
          add(incident.remainderGradient, nonIncident.remainderGradient),
        ),
        total.gradient,
      ),
    ).toBeLessThan(1e-14);
    expect(
      relativeError(
        add(
          total.sourceHessian,
          add(incident.remainderHessian, nonIncident.remainderHessian),
        ),
        total.hessian,
      ),
    ).toBeLessThan(1e-14);
  });

  it("restricts the nonlinear oracle at the current pose and trial poses", () => {
    const mesh = generateRegularCuboidMesh({
      cells: [1, 1, 1],
      fixed: (_position, gridCoordinate) => gridCoordinate[0] === 0,
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
    for (let index = 0; index < coordinates.length; index += 1) {
      coordinates[index] += 0.018 * Math.sin(0.4 + index * 1.3);
    }
    const externalForce = Float64Array.from(
      coordinates,
      (_value, index) => 0.3 * Math.cos(index * 0.8),
    );
    const oracle = createStableNeoHookeanImplicitEulerOracle({
      mesh,
      restData,
      materials: [MATERIAL],
      lumpedMasses,
      restSystem,
      timestep: TIMESTEP,
      predictedPositions: mesh.positions,
      externalForce,
    });
    const vertex = restSystem.activeVertices[restSystem.activeVertices.length - 1]!;
    const exactBasis = computeExactVertexBasis(mesh, restSystem, vertex).basis;
    const equilibriumBasis = exactBasisToActiveMatrix(exactBasis, restSystem);
    const model = createExactStableNeoHookeanJGS2LocalModel({
      oracle,
      restSystem,
      vertex,
      coordinates,
      equilibriumBasis,
    });
    const localDisplacement = new Float64Array([0.003, -0.002, 0.0015]);
    const local = model.evaluate(localDisplacement);
    const direct = oracle.evaluate(local.activeCoordinates);
    const independent = independentProjection(
      direct.gradient,
      direct.hessian,
      equilibriumBasis,
    );

    expect(local.energy).toBe(direct.energy);
    expect(relativeError(local.gradient, independent.gradient)).toBeLessThan(1e-13);
    expect(relativeError(local.hessian, independent.hessian)).toBeLessThan(1e-13);
    expect(local.newtonUpdate).toBeDefined();
    expect(local.newtonResidualNorm).toBeLessThan(1e-7);

    const finiteGradient = centralDifferenceGradient(model.energy, localDisplacement);
    const finiteHessian = centralDifferenceHessian(
      model.gradient,
      localDisplacement,
    );
    expect(relativeError(local.gradient, finiteGradient)).toBeLessThan(2e-7);
    expect(relativeError(local.hessian, finiteHessian)).toBeLessThan(2e-6);
  });
});
