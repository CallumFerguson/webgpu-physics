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
  SurfaceTopology,
  TetrahedralMesh,
  VertexPrecomputation,
} from "./types";

const CUBATURE_TRAINING_MODE_COUNT = 8;
const CUBATURE_INVERSE_ITERATIONS = 24;

function triangleArea(
  positions: Float64Array,
  i0: number,
  i1: number,
  i2: number,
): number {
  const a = i0 * 3;
  const b = i1 * 3;
  const c = i2 * 3;
  const abx = positions[b]! - positions[a]!;
  const aby = positions[b + 1]! - positions[a + 1]!;
  const abz = positions[b + 2]! - positions[a + 2]!;
  const acx = positions[c]! - positions[a]!;
  const acy = positions[c + 1]! - positions[a + 1]!;
  const acz = positions[c + 2]! - positions[a + 2]!;
  const cx = aby * acz - abz * acy;
  const cy = abz * acx - abx * acz;
  const cz = abx * acy - aby * acx;
  return 0.5 * Math.hypot(cx, cy, cz);
}

function validateAndAddClothMasses(
  definition: SceneDefinition,
  masses: Float64Array,
): void {
  const cloth = definition.cloth;
  if (!cloth) return;
  const vertexCount = getVertexCount(definition.mesh);
  if (cloth.triangles.length === 0 || cloth.triangles.length % 3 !== 0) {
    throw new RangeError("Cloth triangles must contain one or more index triples.");
  }
  for (const [name, value] of [
    ["density", cloth.density],
    ["youngModulus", cloth.youngModulus],
    ["thickness", cloth.thickness],
  ] as const) {
    if (!(value > 0) || !Number.isFinite(value)) {
      throw new RangeError(`Cloth ${name} must be finite and positive.`);
    }
  }
  if (
    !Number.isFinite(cloth.poissonRatio) ||
    cloth.poissonRatio < 0 ||
    cloth.poissonRatio >= 0.5
  ) {
    throw new RangeError("Cloth poissonRatio must be in [0, 0.5).");
  }
  if (
    !Number.isFinite(cloth.bendingStiffness) ||
    cloth.bendingStiffness < 0
  ) {
    throw new RangeError("Cloth bendingStiffness must be finite and nonnegative.");
  }
  const triangleCount = cloth.triangles.length / 3;
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const indices = [
      cloth.triangles[triangle * 3]!,
      cloth.triangles[triangle * 3 + 1]!,
      cloth.triangles[triangle * 3 + 2]!,
    ] as const;
    if (
      new Set(indices).size !== 3 ||
      indices.some((vertex) => vertex >= vertexCount)
    ) {
      throw new RangeError(`Cloth triangle ${triangle} has invalid vertex indices.`);
    }
    const area = triangleArea(definition.mesh.positions, ...indices);
    if (!(area > 1e-12) || !Number.isFinite(area)) {
      throw new RangeError(`Cloth triangle ${triangle} must be nondegenerate.`);
    }
    const vertexMass = (cloth.density * cloth.thickness * area) / 3;
    for (const vertex of indices) masses[vertex] += vertexMass;
  }
}

function surfaceWithCloth(
  tetrahedronSurface: SurfaceTopology,
  definition: SceneDefinition,
): SurfaceTopology {
  const cloth = definition.cloth;
  if (!cloth) return tetrahedronSurface;
  const triangles = new Uint32Array(
    tetrahedronSurface.triangles.length + cloth.triangles.length,
  );
  triangles.set(tetrahedronSurface.triangles);
  triangles.set(cloth.triangles, tetrahedronSurface.triangles.length);
  const edgeKeys = new Set<string>();
  const edgeValues: number[] = [];
  const addEdge = (first: number, second: number): void => {
    const low = Math.min(first, second);
    const high = Math.max(first, second);
    const key = `${low}:${high}`;
    if (edgeKeys.has(key)) return;
    edgeKeys.add(key);
    edgeValues.push(low, high);
  };
  for (let edge = 0; edge < tetrahedronSurface.edges.length; edge += 2) {
    addEdge(
      tetrahedronSurface.edges[edge]!,
      tetrahedronSurface.edges[edge + 1]!,
    );
  }
  for (let triangle = 0; triangle < cloth.triangles.length; triangle += 3) {
    const i0 = cloth.triangles[triangle]!;
    const i1 = cloth.triangles[triangle + 1]!;
    const i2 = cloth.triangles[triangle + 2]!;
    addEdge(i0, i1);
    addEdge(i1, i2);
    addEdge(i2, i0);
  }
  return { triangles, edges: Uint32Array.from(edgeValues) };
}

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
  const surface = surfaceWithCloth(
    extractBoundarySurface(definition.mesh),
    definition,
  );
  const restTetraData = computeRestTetraData(
    definition.mesh,
    definition.materials,
  );
  const lumpedMasses = computeLumpedMasses(
    definition.mesh,
    definition.materials,
    restTetraData,
  );
  validateAndAddClothMasses(definition, lumpedMasses);
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
  const incidentTetrahedronCounts = new Uint32Array(
    getVertexCount(definition.mesh),
  );
  for (const vertex of definition.mesh.tetrahedra) {
    incidentTetrahedronCounts[vertex] += 1;
  }
  const hasActiveTetrahedronVertex = [...restSystem.activeVertices].some(
    (vertex) => incidentTetrahedronCounts[vertex]! > 0,
  );
  const nonlinearCorpus = stableNeoHookean && hasActiveTetrahedronVertex
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
    if (stableNeoHookean && incidentTetrahedronCounts[vertex] === 0) {
      vertexPrecomputations.push({
        vertex,
        ...(options.retainExactBases ? { exactBasis: exact.basis } : {}),
        schurInverse: exact.schurInverse,
        cubature: [],
        cubatureModel: "stable-neo-hookean",
        trainingResidualF64: 0,
        trainingResidual: 0,
        validTrainingPoseCount: 0,
        trivialTrainingPoseCount: 0,
        nonzeroTrainingCandidateCount: 0,
        trainingColumnRank: 0,
        packedNonzeroTrainingCandidateCount: 0,
        packedTrainingColumnRank: 0,
      });
      continue;
    }
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
