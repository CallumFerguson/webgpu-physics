import { describe, expect, it } from "vitest";

import { buildSceneDefinition, type SceneId } from "../../scenes";
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
import { getTetrahedronCount } from "./mesh";
import {
  activeCoordinatesFromFullPositions,
  computeCorotatedLinearRotations,
  computeDirectComplementaryEquilibriumBasis,
  createCorotatedLinearImplicitEulerOracle,
  evaluateCorotatedLinearTetrahedron,
  solveEquilibriumBasisNewtonStep,
  solveFullNewtonStep,
} from "./oracle";
import {
  computeExactVertexBasis,
  exactBasisToActiveMatrix,
} from "./precompute";

function multiplyMatrixVector(
  matrix: Float64Array,
  vector: Float64Array,
): Float64Array {
  const dimension = vector.length;
  if (matrix.length !== dimension * dimension) {
    throw new RangeError("Test matrix dimensions do not match.");
  }
  const result = new Float64Array(dimension);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < dimension; column += 1) {
      result[row] += matrix[row * dimension + column]! * vector[column]!;
    }
  }
  return result;
}

function buildOracleFixture() {
  const definition = buildSceneDefinition("minimal");
  const restData = computeRestTetraData(definition.mesh, definition.materials);
  const lumpedMasses = computeLumpedMasses(
    definition.mesh,
    definition.materials,
    restData,
  );
  const restSystem = assembleRestLinearSystem(
    definition.mesh,
    restData,
    lumpedMasses,
    definition.settings.timestep,
  );
  const rotationPositions = definition.mesh.positions.slice();
  const positions = definition.mesh.positions.slice();
  const predictedPositions = definition.mesh.positions.slice();
  for (const vertex of restSystem.activeVertices) {
    const activeBase = restSystem.vertexToActiveDof[vertex]!;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const phase = activeBase + coordinate + 1;
      rotationPositions[vertex * 3 + coordinate] +=
        0.08 * Math.sin(phase * 0.61);
      positions[vertex * 3 + coordinate] +=
        0.08 * Math.sin(phase * 0.61) + 0.025 * Math.cos(phase * 0.37);
      predictedPositions[vertex * 3 + coordinate] +=
        0.015 * Math.sin(phase * 0.43) -
        (coordinate === 1 ? 0.02 : 0);
    }
  }
  const oracle = createCorotatedLinearImplicitEulerOracle({
    mesh: definition.mesh,
    restData,
    lumpedMasses,
    restSystem,
    timestep: definition.settings.timestep,
    predictedPositions,
    rotationPositions,
    floorContact: { height: 1.5, stiffness: 25_000 },
  });
  const coordinates = activeCoordinatesFromFullPositions(positions, restSystem);
  return {
    definition,
    restData,
    lumpedMasses,
    restSystem,
    rotationPositions,
    positions,
    oracle,
    coordinates,
  };
}

describe("central finite-difference and error helpers", () => {
  it("recovers an analytic polynomial gradient and Hessian", () => {
    const point = new Float64Array([0.7, -1.2, 0.35]);
    const energy = (coordinates: Float64Array): number =>
      coordinates[0]! ** 4 +
      0.5 * coordinates[1]! ** 2 +
      2 * coordinates[0]! * coordinates[2]! +
      3 * coordinates[2]!;
    const gradient = (coordinates: Float64Array): Float64Array =>
      new Float64Array([
        4 * coordinates[0]! ** 3 + 2 * coordinates[2]!,
        coordinates[1]!,
        2 * coordinates[0]! + 3,
      ]);
    const analyticHessian = new Float64Array([
      12 * point[0]! ** 2,
      0,
      2,
      0,
      1,
      0,
      2,
      0,
      0,
    ]);

    expect(
      relativeError(centralDifferenceGradient(energy, point), gradient(point)),
    ).toBeLessThan(1e-9);
    expect(
      relativeError(
        centralDifferenceHessian(gradient, point),
        analyticHessian,
      ),
    ).toBeLessThan(1e-10);
  });

  it("uses the roadmap's explicit max-one relative normalization", () => {
    expect(relativeError(new Float64Array([0.5]), new Float64Array([0]))).toBe(
      0.5,
    );
    expect(
      relativeError(new Float64Array([3, 4]), new Float64Array([0, 0])),
    ).toBe(1);
    expect(
      relativeError(new Float64Array([2, 0]), new Float64Array([1, 0])),
    ).toBe(0.5);
  });
});

describe("Float64 co-rotated implicit Euler oracle", () => {
  it("matches central differences for total analytic gradient and Hessian", () => {
    const { oracle, coordinates } = buildOracleFixture();
    const evaluation = oracle.evaluate(coordinates);
    const finiteGradient = centralDifferenceGradient(oracle.energy, coordinates);
    const finiteHessian = centralDifferenceHessian(
      oracle.gradient,
      coordinates,
    );
    const gradientError = relativeError(evaluation.gradient, finiteGradient);
    const hessianError = relativeError(evaluation.hessian, finiteHessian);

    console.log(
      `CPU oracle derivatives: gradient ${gradientError.toExponential(3)}, ` +
        `Hessian ${hessianError.toExponential(3)}`,
    );
    expect(gradientError).toBeLessThan(1e-5);
    expect(hessianError).toBeLessThan(1e-4);
  });

  it("reports exact all-element energy components without Cubature", () => {
    const { oracle, coordinates, definition } = buildOracleFixture();
    const evaluation = oracle.evaluate(coordinates);
    const summedElementEnergy = [...evaluation.tetrahedronEnergies].reduce(
      (sum, value) => sum + value,
      0,
    );

    expect(evaluation.tetrahedronEnergies).toHaveLength(
      getTetrahedronCount(definition.mesh),
    );
    expect(evaluation.components.elasticity).toBeCloseTo(
      summedElementEnergy,
      12,
    );
    expect(evaluation.components.inertia).toBeGreaterThan(0);
    expect(evaluation.components.floorContact).toBeGreaterThan(0);
    expect(evaluation.energy).toBeCloseTo(
      evaluation.components.inertia +
        evaluation.components.elasticity +
        evaluation.components.floorContact,
      12,
    );
  });

  it("produces objective rotations and zero material response for rigid motion", () => {
    const definition = buildSceneDefinition("drop");
    const restData = computeRestTetraData(
      definition.mesh,
      definition.materials,
    );
    const angle = 0.47;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const rotation = new Float64Array([
      cosine,
      -sine,
      0,
      sine,
      cosine,
      0,
      0,
      0,
      1,
    ]);
    const positions = definition.mesh.positions.slice();
    for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
      const x = definition.mesh.positions[vertex * 3]!;
      const y = definition.mesh.positions[vertex * 3 + 1]!;
      const z = definition.mesh.positions[vertex * 3 + 2]!;
      positions[vertex * 3] = cosine * x - sine * y + 1.3;
      positions[vertex * 3 + 1] = sine * x + cosine * y - 0.4;
      positions[vertex * 3 + 2] = z + 0.8;
    }
    const rotations = computeCorotatedLinearRotations(
      definition.mesh,
      restData,
      positions,
    );

    for (
      let tetrahedron = 0;
      tetrahedron < getTetrahedronCount(definition.mesh);
      tetrahedron += 1
    ) {
      const recovered = rotations.subarray(
        tetrahedron * 9,
        tetrahedron * 9 + 9,
      );
      const element = evaluateCorotatedLinearTetrahedron(
        definition.mesh,
        restData,
        tetrahedron,
        positions,
        recovered,
      );
      expect(relativeError(recovered, rotation)).toBeLessThan(1e-12);
      expect(Math.abs(determinant3(recovered) - 1)).toBeLessThan(1e-12);
      expect(Math.abs(element.energy)).toBeLessThan(1e-8);
      expect(relativeError(element.gradient, new Float64Array(12))).toBeLessThan(
        1e-8,
      );
    }
  });

  it("solves the complete dense Newton system to Float64 residual", () => {
    const { oracle, coordinates } = buildOracleFixture();
    const evaluation = oracle.evaluate(coordinates);
    const step = solveFullNewtonStep(evaluation);
    const hessianStep = multiplyMatrixVector(evaluation.hessian, step);
    const negativeGradient = Float64Array.from(
      evaluation.gradient,
      (value) => -value,
    );
    const residualError = relativeError(hessianStep, negativeGradient);

    console.log(`CPU full Newton residual: ${residualError.toExponential(3)}`);
    expect(residualError).toBeLessThan(1e-12);
  });

  it("matches every full Newton block with an exact equilibrium-basis solve", () => {
    const { oracle, coordinates, restSystem } = buildOracleFixture();
    const evaluation = oracle.evaluate(coordinates);
    const fullStep = solveFullNewtonStep(evaluation);
    let maximumBlockError = 0;

    for (const vertex of restSystem.activeVertices) {
      const localBase = restSystem.vertexToActiveDof[vertex]!;
      const direct = computeDirectComplementaryEquilibriumBasis(
        evaluation.hessian,
        evaluation.gradient.length,
        localBase,
      );
      const localStep = solveEquilibriumBasisNewtonStep(
        evaluation,
        direct.basis,
      );
      const blockError = relativeError(
        localStep,
        fullStep.subarray(localBase, localBase + 3),
      );
      maximumBlockError = Math.max(maximumBlockError, blockError);
      expect(blockError).toBeLessThan(1e-8);
    }
    console.log(
      `CPU equilibrium Newton-block error: ${maximumBlockError.toExponential(3)}`,
    );
  });
});

describe("direct complementary-coordinate equilibrium bases", () => {
  it.each(["minimal", "stiffness"] satisfies readonly SceneId[])(
    "matches equations 19-23 full-coordinate solves for %s",
    (sceneId) => {
      const definition = buildSceneDefinition(sceneId);
      const restData = computeRestTetraData(
        definition.mesh,
        definition.materials,
      );
      const lumpedMasses = computeLumpedMasses(
        definition.mesh,
        definition.materials,
        restData,
      );
      const system = assembleRestLinearSystem(
        definition.mesh,
        restData,
        lumpedMasses,
        definition.settings.timestep,
      );
      let maximumBasisError = 0;
      let maximumSchurError = 0;

      for (const vertex of system.activeVertices) {
        const fullCoordinate = computeExactVertexBasis(
          definition.mesh,
          system,
          vertex,
        );
        const direct = computeDirectComplementaryEquilibriumBasis(
          system.hessian,
          system.dimension,
          system.vertexToActiveDof[vertex]!,
        );
        const basisError = relativeError(
          direct.basis,
          exactBasisToActiveMatrix(fullCoordinate.basis, system),
        );
        const schurError = relativeError(
          direct.schurComplement,
          fullCoordinate.schurInverse,
        );
        maximumBasisError = Math.max(maximumBasisError, basisError);
        maximumSchurError = Math.max(maximumSchurError, schurError);
        expect(basisError).toBeLessThan(1e-8);
        expect(schurError).toBeLessThan(1e-8);
      }
      console.log(
        `${sceneId} complementary basis: basis ` +
          `${maximumBasisError.toExponential(3)}, Schur ` +
          maximumSchurError.toExponential(3),
      );
    },
  );
});
