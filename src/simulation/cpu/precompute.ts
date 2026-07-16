import { computeLumpedMasses, computeRestTetraData, assembleRestLinearSystem } from "./fem";
import { buildLowFrequencyTrainingPoses, selectCubatureSamples } from "./cubature";
import { buildNonlinearCubaturePoseCorpus } from "./nonlinear-cubature";
import {
  NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
  selectStableNeoHookeanCubatureSamples,
} from "./nonlinear-cubature-training";
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
  const referencedMaterialModels = new Set(
    [...definition.mesh.materialIds].map(
      (materialId) =>
        definition.materials[materialId]?.model ?? "corotated-linear",
    ),
  );
  const stableNeoHookean = referencedMaterialModels.has(
    "stable-neo-hookean",
  );
  if (stableNeoHookean && referencedMaterialModels.size !== 1) {
    throw new RangeError(
      "Mixed stable Neo-Hookean/co-rotated nonlinear Cubature preprocessing is not implemented.",
    );
  }
  const linearTrainingPoses = stableNeoHookean
    ? undefined
    : buildLowFrequencyTrainingPoses(
        definition.mesh,
        restSystem,
        CUBATURE_TRAINING_MODE_COUNT,
        CUBATURE_INVERSE_ITERATIONS,
      );
  const nonlinearCorpus = stableNeoHookean
    ? buildNonlinearCubaturePoseCorpus({
        mesh: definition.mesh,
        restData: restTetraData,
        materials: definition.materials,
        lumpedMasses,
        restSystem,
      })
    : undefined;
  const vertexPrecomputations: VertexPrecomputation[] = [];

  for (const vertex of restSystem.activeVertices) {
    const exact = computeExactVertexBasis(definition.mesh, restSystem, vertex);
    if (nonlinearCorpus) {
      const cubature = selectStableNeoHookeanCubatureSamples(
        {
          mesh: definition.mesh,
          restData: restTetraData,
          materials: definition.materials,
          lumpedMasses,
          restSystem,
          timestep: definition.settings.timestep,
          predictedPositions: definition.mesh.positions,
          sourceVertex: vertex,
          exactBasis: exact.basis,
        },
        nonlinearCorpus.training,
        definition.settings.cubatureSamples,
      );
      if (
        cubature.packedResidual >
        NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE + 1e-12
      ) {
        throw new Error(
          `Stable Neo-Hookean Cubature source vertex ${vertex} has packed ` +
            `training residual ${cubature.packedResidual}; required at most ` +
            `${NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE}.`,
        );
      }
      vertexPrecomputations.push({
        vertex,
        ...(options.retainExactBases ? { exactBasis: exact.basis } : {}),
        schurInverse: exact.schurInverse,
        cubature: cubature.packedSamples,
        cubatureModel: "stable-neo-hookean",
        trainingResidualF64: cubature.residual,
        trainingResidual: cubature.packedResidual,
        validTrainingPoseCount: cubature.validTrainingPoseCount,
        trivialTrainingPoseCount: cubature.trivialTrainingPoseCount,
        nonzeroTrainingCandidateCount: cubature.nonzeroCandidateCount,
        trainingColumnRank: cubature.trainingColumnRank,
        packedNonzeroTrainingCandidateCount:
          cubature.packedNonzeroCandidateCount,
        packedTrainingColumnRank: cubature.packedTrainingColumnRank,
      });
      continue;
    }
    const cubature = selectCubatureSamples(
      definition.mesh,
      restTetraData,
      lumpedMasses,
      definition.settings.timestep,
      vertex,
      exact.basis,
      linearTrainingPoses!,
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
