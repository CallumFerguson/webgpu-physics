import { computeLumpedMasses, computeRestTetraData, assembleRestLinearSystem } from "./fem";
import { buildLowFrequencyTrainingPoses, selectCubatureSamples } from "./cubature";
import { invert3, solveCholesky } from "./math";
import { extractBoundarySurface, getVertexCount } from "./mesh";
import type {
  PrecomputedScene,
  PrecomputeOptions,
  RestLinearSystem,
  SceneDefinition,
  TetrahedralMesh,
  VertexPrecomputation,
} from "./types";

const CUBATURE_TRAINING_MODE_COUNT = 8;
const CUBATURE_INVERSE_ITERATIONS = 24;

interface ExactBasisResult {
  readonly basis: Float64Array;
  readonly schurInverse: Float64Array;
}

export function computeExactVertexBasis(
  mesh: TetrahedralMesh,
  system: RestLinearSystem,
  vertex: number,
): ExactBasisResult {
  const localBase = system.vertexToActiveDof[vertex]!;
  if (localBase < 0) {
    throw new Error(`Cannot build a perturbation basis for fixed vertex ${vertex}.`);
  }

  const inverseTimesSelection = new Float64Array(system.dimension * 3);
  for (let column = 0; column < 3; column += 1) {
    const rightHandSide = new Float64Array(system.dimension);
    rightHandSide[localBase + column] = 1;
    const solution = solveCholesky(
      system.choleskyLower,
      rightHandSide,
      system.dimension,
    );
    for (let row = 0; row < system.dimension; row += 1) {
      inverseTimesSelection[row * 3 + column] = solution[row]!;
    }
  }

  const compliance = new Float64Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      compliance[row * 3 + column] =
        0.5 *
        (inverseTimesSelection[(localBase + row) * 3 + column]! +
          inverseTimesSelection[(localBase + column) * 3 + row]!);
    }
  }
  const schurInverse = invert3(compliance);
  const activeBasis = new Float64Array(system.dimension * 3);
  for (let row = 0; row < system.dimension; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      let value = 0;
      for (let inner = 0; inner < 3; inner += 1) {
        value +=
          inverseTimesSelection[row * 3 + inner]! *
          schurInverse[inner * 3 + column]!;
      }
      activeBasis[row * 3 + column] = value;
    }
  }

  const basis = new Float64Array(getVertexCount(mesh) * 9);
  for (let globalVertex = 0; globalVertex < getVertexCount(mesh); globalVertex += 1) {
    const activeBase = system.vertexToActiveDof[globalVertex]!;
    if (activeBase < 0) {
      continue;
    }
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      basis.set(
        activeBasis.subarray(
          (activeBase + coordinate) * 3,
          (activeBase + coordinate) * 3 + 3,
        ),
        globalVertex * 9 + coordinate * 3,
      );
    }
  }

  return { basis, schurInverse };
}

export function exactBasisToActiveMatrix(
  exactBasis: Float64Array,
  system: RestLinearSystem,
): Float64Array {
  const activeBasis = new Float64Array(system.dimension * 3);
  for (const vertex of system.activeVertices) {
    const activeBase = system.vertexToActiveDof[vertex]!;
    activeBasis.set(
      exactBasis.subarray(vertex * 9, vertex * 9 + 9),
      activeBase * 3,
    );
  }
  return activeBasis;
}

export function buildPrecomputedScene(
  definition: SceneDefinition,
  options: PrecomputeOptions = {},
): PrecomputedScene {
  const surface = extractBoundarySurface(definition.mesh);
  const restTetraData = computeRestTetraData(
    definition.mesh,
    definition.materials,
  );
  const lumpedMasses = computeLumpedMasses(
    definition.mesh,
    definition.materials,
    restTetraData,
  );
  const restSystem = assembleRestLinearSystem(
    definition.mesh,
    restTetraData,
    lumpedMasses,
    definition.settings.timestep,
  );
  const trainingPoses = buildLowFrequencyTrainingPoses(
    definition.mesh,
    restSystem,
    CUBATURE_TRAINING_MODE_COUNT,
    CUBATURE_INVERSE_ITERATIONS,
  );
  const vertexPrecomputations: VertexPrecomputation[] = [];

  for (const vertex of restSystem.activeVertices) {
    const exact = computeExactVertexBasis(definition.mesh, restSystem, vertex);
    const cubature = selectCubatureSamples(
      definition.mesh,
      restTetraData,
      lumpedMasses,
      definition.settings.timestep,
      vertex,
      exact.basis,
      trainingPoses,
      definition.settings.cubatureSamples,
    );
    vertexPrecomputations.push({
      vertex,
      ...(options.retainExactBases ? { exactBasis: exact.basis } : {}),
      schurInverse: exact.schurInverse,
      cubature: cubature.samples,
      trainingResidual: cubature.residual,
    });
  }

  return {
    ...definition,
    surface,
    restTetraData,
    lumpedMasses,
    restSystem,
    vertexPrecomputations,
  };
}
