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
import { determinant3, multiply3 } from "./math";
import { transformTetrahedralMesh } from "./mesh";
import { computeExactVertexBasis, exactBasisToActiveMatrix } from "./precompute";
import {
  activeCoordinatesFromFullPositions,
  computeDirectComplementaryEquilibriumBasis,
  createCorotatedLinearImplicitEulerOracle,
  solveEquilibriumBasisNewtonStep,
  solveFullNewtonStep,
  type CorotatedLinearImplicitEulerOracle,
  type ImplicitEulerEvaluation,
} from "./oracle";
import type {
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  SceneDefinition,
  TetrahedralMesh,
  Vec3,
} from "./types";

export const PHASE0_ORACLE_FIXTURE_ID =
  "phase0.oracle-single-tetrahedron" as const;
export const PHASE0_ORACLE_CORPUS_ID = "phase0.oracle-poses" as const;
export const PHASE0_ORACLE_CORPUS_SEED = 65_537;
export const PHASE0_ORACLE_CORPUS_GENERATOR_VERSION = "1" as const;
export const PHASE0_ORACLE_CORPUS_CASE_COUNT = 64;
export const PHASE0_ORACLE_DISPLACEMENT_SCALE = 0.15;
export const PHASE0_ORACLE_MINIMUM_DETERMINANT = 0.25;
export const PHASE0_ORACLE_TIMESTEP = 1 / 60;

export const PHASE0_REFERENCE_MATERIAL: LinearMaterial = {
  name: "phase0-reference-solid",
  density: 1_000,
  youngModulus: 80_000,
  poissonRatio: 0.3,
  color: [0.35, 0.7, 0.95, 1],
};

export const PHASE0_ORACLE_REST_POSITIONS = new Float64Array([
  0, 0, 0,
  1, 0, 0,
  0.15, 0.9, 0,
  0.1, 0.2, 0.85,
]);

export type Phase0OraclePoseKind = "rest" | "rigid" | "deformed";

export interface Phase0OraclePose {
  readonly id: string;
  readonly index: number;
  readonly kind: Phase0OraclePoseKind;
  readonly positions: Float64Array;
  /** det(Ds * Dm^-1), which is one at rest and for rigid rotations. */
  readonly determinant: number;
}

export interface Phase0OracleFixture {
  readonly definition: SceneDefinition;
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
}

export interface Phase0OraclePoseEvaluation {
  readonly pose: Phase0OraclePose;
  readonly oracle: CorotatedLinearImplicitEulerOracle;
  readonly coordinates: Float64Array;
  readonly evaluation: ImplicitEulerEvaluation;
}

export interface Phase0OraclePoseErrors {
  readonly gradientRelativeError: number;
  readonly hessianRelativeError: number;
  readonly newtonResidualRelativeError: number;
  readonly maximumNewtonBlockRelativeError: number;
  readonly maximumEquilibriumBasisRelativeError: number;
}

function createCorpusRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b_79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function shapeDeterminant(positions: Float64Array): number {
  const rest = PHASE0_ORACLE_REST_POSITIONS;
  const restShape = new Float64Array([
    rest[3]! - rest[0]!, rest[6]! - rest[0]!, rest[9]! - rest[0]!,
    rest[4]! - rest[1]!, rest[7]! - rest[1]!, rest[10]! - rest[1]!,
    rest[5]! - rest[2]!, rest[8]! - rest[2]!, rest[11]! - rest[2]!,
  ]);
  const deformedShape = new Float64Array([
    positions[3]! - positions[0]!, positions[6]! - positions[0]!, positions[9]! - positions[0]!,
    positions[4]! - positions[1]!, positions[7]! - positions[1]!, positions[10]! - positions[1]!,
    positions[5]! - positions[2]!, positions[8]! - positions[2]!, positions[11]! - positions[2]!,
  ]);
  return determinant3(deformedShape) / determinant3(restShape);
}

function explicitOracleMesh(): TetrahedralMesh {
  return {
    positions: PHASE0_ORACLE_REST_POSITIONS.slice(),
    tetrahedra: new Uint32Array([0, 1, 2, 3]),
    materialIds: new Uint16Array([0]),
    fixed: new Uint8Array([1, 0, 0, 0]),
    bodyIds: new Uint16Array([0, 0, 0, 0]),
  };
}

export function buildPhase0OracleDefinition(): SceneDefinition {
  return {
    id: PHASE0_ORACLE_FIXTURE_ID,
    title: "Canonical single-tetrahedron oracle",
    description:
      "The frozen Phase 0 Float64 derivative, Newton-block, and equilibrium-basis oracle fixture.",
    mesh: explicitOracleMesh(),
    materials: [PHASE0_REFERENCE_MATERIAL],
    settings: {
      timestep: PHASE0_ORACLE_TIMESTEP,
      gravity: [0, 0, 0],
      floorY: 0,
      solverIterations: 1,
      cubatureSamples: 4,
    },
    camera: {
      eye: [2.5, 2, 3.5],
      target: [0.3, 0.3, 0.25],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 3,
      label: "oracle tetrahedron apex",
      expectedMotion: [1, 1, 1],
    },
  };
}

export function buildPhase0OracleFixture(): Phase0OracleFixture {
  const definition = buildPhase0OracleDefinition();
  const restData = computeRestTetraData(
    definition.mesh,
    definition.materials,
  );
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
  return {
    definition,
    mesh: definition.mesh,
    restData,
    lumpedMasses,
    restSystem,
  };
}

function rigidPose(index: number, rotationEuler: Vec3): Phase0OraclePose {
  const mesh = transformTetrahedralMesh(explicitOracleMesh(), {
    rotationEuler,
  });
  return {
    id: `${PHASE0_ORACLE_CORPUS_ID}/${index.toString().padStart(2, "0")}`,
    index,
    kind: "rigid",
    positions: mesh.positions,
    determinant: shapeDeterminant(mesh.positions),
  };
}

/**
 * Frozen v1 positive-determinant corpus. Case 0 is rest, cases 1-3 are exact
 * rigid rotations about the fixed origin, and cases 4-63 are seeded strains.
 */
export function generatePhase0OraclePoseCorpus(): readonly Phase0OraclePose[] {
  const poses: Phase0OraclePose[] = [
    {
      id: `${PHASE0_ORACLE_CORPUS_ID}/00`,
      index: 0,
      kind: "rest",
      positions: PHASE0_ORACLE_REST_POSITIONS.slice(),
      determinant: 1,
    },
    rigidPose(1, [0.31, -0.22, 0.17]),
    rigidPose(2, [-0.4, 0.13, 0.52]),
    rigidPose(3, [0.71, -0.38, -0.27]),
  ];
  const random = createCorpusRandom(PHASE0_ORACLE_CORPUS_SEED);

  while (poses.length < PHASE0_ORACLE_CORPUS_CASE_COUNT) {
    const positions = PHASE0_ORACLE_REST_POSITIONS.slice();
    // Vertex zero is the permanent origin constraint. Perturb only active DOFs.
    for (let coordinate = 3; coordinate < positions.length; coordinate += 1) {
      positions[coordinate] +=
        (random() * 2 - 1) * PHASE0_ORACLE_DISPLACEMENT_SCALE;
    }
    const determinant = shapeDeterminant(positions);
    if (determinant < PHASE0_ORACLE_MINIMUM_DETERMINANT) {
      continue;
    }
    const index = poses.length;
    poses.push({
      id: `${PHASE0_ORACLE_CORPUS_ID}/${index.toString().padStart(2, "0")}`,
      index,
      kind: "deformed",
      positions,
      determinant,
    });
  }
  return poses;
}

export function evaluatePhase0OraclePose(
  fixture: Phase0OracleFixture,
  pose: Phase0OraclePose,
): Phase0OraclePoseEvaluation {
  const oracle = createCorotatedLinearImplicitEulerOracle({
    mesh: fixture.mesh,
    restData: fixture.restData,
    lumpedMasses: fixture.lumpedMasses,
    restSystem: fixture.restSystem,
    timestep: fixture.definition.settings.timestep,
    predictedPositions: pose.positions,
    rotationPositions: pose.positions,
  });
  const coordinates = activeCoordinatesFromFullPositions(
    pose.positions,
    fixture.restSystem,
  );
  return { pose, oracle, coordinates, evaluation: oracle.evaluate(coordinates) };
}

function multiplyMatrixVector(
  matrix: Float64Array,
  vector: Float64Array,
): Float64Array {
  const dimension = vector.length;
  const result = new Float64Array(dimension);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < dimension; column += 1) {
      result[row] += matrix[row * dimension + column]! * vector[column]!;
    }
  }
  return result;
}

/** Execute every frozen CPU oracle gate for one corpus pose. */
export function measurePhase0OraclePoseErrors(
  fixture: Phase0OracleFixture,
  pose: Phase0OraclePose,
): Phase0OraclePoseErrors {
  const { oracle, coordinates, evaluation } = evaluatePhase0OraclePose(
    fixture,
    pose,
  );
  const finiteGradient = centralDifferenceGradient(oracle.energy, coordinates);
  const finiteHessian = centralDifferenceHessian(oracle.gradient, coordinates);
  const fullStep = solveFullNewtonStep(evaluation);
  const hessianStep = multiplyMatrixVector(evaluation.hessian, fullStep);
  const negativeGradient = Float64Array.from(
    evaluation.gradient,
    (value) => -value,
  );
  let maximumNewtonBlockRelativeError = 0;
  let maximumEquilibriumBasisRelativeError = 0;
  const rotation = oracle.rotations.subarray(0, 9);
  const rotationTranspose = new Float64Array([
    rotation[0]!, rotation[3]!, rotation[6]!,
    rotation[1]!, rotation[4]!, rotation[7]!,
    rotation[2]!, rotation[5]!, rotation[8]!,
  ]);
  for (const vertex of fixture.restSystem.activeVertices) {
    const localBase = fixture.restSystem.vertexToActiveDof[vertex]!;
    const basis = computeDirectComplementaryEquilibriumBasis(
      evaluation.hessian,
      evaluation.gradient.length,
      localBase,
    );
    const localStep = solveEquilibriumBasisNewtonStep(
      evaluation,
      basis.basis,
    );
    maximumNewtonBlockRelativeError = Math.max(
      maximumNewtonBlockRelativeError,
      relativeError(localStep, fullStep.subarray(localBase, localBase + 3)),
    );

    const restBasis = computeExactVertexBasis(
      fixture.mesh,
      fixture.restSystem,
      vertex,
    ).basis;
    const transformedBasis = new Float64Array(restBasis.length);
    for (let targetVertex = 0; targetVertex < fixture.mesh.positions.length / 3; targetVertex += 1) {
      const block = restBasis.subarray(
        targetVertex * 9,
        targetVertex * 9 + 9,
      );
      transformedBasis.set(
        multiply3(multiply3(rotation, block), rotationTranspose),
        targetVertex * 9,
      );
    }
    maximumEquilibriumBasisRelativeError = Math.max(
      maximumEquilibriumBasisRelativeError,
      relativeError(
        basis.basis,
        exactBasisToActiveMatrix(transformedBasis, fixture.restSystem),
      ),
    );
  }

  return {
    gradientRelativeError: relativeError(evaluation.gradient, finiteGradient),
    hessianRelativeError: relativeError(evaluation.hessian, finiteHessian),
    newtonResidualRelativeError: relativeError(
      hessianStep,
      negativeGradient,
    ),
    maximumNewtonBlockRelativeError,
    maximumEquilibriumBasisRelativeError,
  };
}
