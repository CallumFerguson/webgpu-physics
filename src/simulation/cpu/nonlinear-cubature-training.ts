import {
  fitNonnegativeColumnWeights,
  selectGreedyNonnegativeColumns,
} from "./cubature";
import {
  buildVertexDeformationGradientStencil,
  computeVertexPolarFrames,
} from "./deformation-gradient";
import { multiply3, solveDenseLinearSystem, squaredNorm } from "./math";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import type { NonlinearCubaturePose } from "./nonlinear-cubature";
import { evaluateStableNeoHookeanTetrahedron } from "./stable-neo-hookean";
import type {
  CubatureSample,
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

const LOCAL_DIMENSION = 3;
const ELEMENT_DIMENSION = 12;
export const NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE = 0.01;
export const NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE = 0.02;
export const NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE = 1e-3;
export const NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS = 2;
export const NONLINEAR_CUBATURE_COLUMN_RANK_RELATIVE_TOLERANCE = 1e-9;
export const NONLINEAR_CUBATURE_CO_ROTATION =
  "R_v * Ubar_vi * transpose(R_i)" as const;
export const NONLINEAR_CUBATURE_VERTEX_FRAME =
  "polar(rest-volume-weighted-average-F)" as const;
export const NONLINEAR_CUBATURE_TARGET_NORMALIZATION =
  "per-pose-l2-complementary-gradient" as const;
export const NONLINEAR_CUBATURE_RESIDUAL_METRIC =
  "l2(packed-columns*packed-weights-f64-target)/l2(f64-target)" as const;
export const NONLINEAR_CUBATURE_UPDATE_RMS_METRIC =
  "sqrt(sum(l2(selected-update-exact-update)^2)/sum(l2(exact-update)^2))" as const;
export const NONLINEAR_CUBATURE_SELECTION_METHOD =
  "deterministic-greedy-nnls-with-small-fixture-subset-audit" as const;

export interface StableNeoHookeanCubatureContext {
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly materials: readonly LinearMaterial[];
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
  readonly timestep: number;
  readonly predictedPositions: Float64Array;
  readonly sourceVertex: number;
  /** Full-coordinate row-major rest basis: one 3x3 block per vertex. */
  readonly exactBasis: Float64Array;
}

export interface NonlinearCubatureCandidateEvaluation {
  readonly tetrahedron: number;
  /** Current co-rotated 12x3 element basis. */
  readonly currentBasisBlocks: Float64Array;
  readonly projectedGradient: Float64Array;
  readonly projectedHessian: Float64Array;
  /** Candidate column after subtracting an incident exact source block. */
  readonly remainderGradient: Float64Array;
  readonly remainderHessian: Float64Array;
  /** Distributed contribution to the exact source gather, or zeros. */
  readonly sourceGradient: Float64Array;
  readonly sourceHessian: Float64Array;
  readonly deformationDeterminant: number;
}

export interface StableNeoHookeanJGS2LocalSystem {
  readonly sourceGradient: Float64Array;
  readonly sourceHessian: Float64Array;
  readonly remainderGradient: Float64Array;
  readonly remainderHessian: Float64Array;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
  readonly newtonUpdate: Float64Array | undefined;
  readonly candidates: readonly NonlinearCubatureCandidateEvaluation[];
  readonly minimumDeformationDeterminant: number;
}

export interface StableNeoHookeanCubatureSelection {
  readonly samples: readonly CubatureSample[];
  readonly packedSamples: readonly CubatureSample[];
  readonly residual: number;
  readonly packedResidual: number;
  readonly selectedTetrahedra: readonly number[];
  readonly weights: Float64Array;
  readonly stackedTargetNorm: number;
  readonly validTrainingPoseCount: number;
  readonly trivialTrainingPoseCount: number;
  readonly nonzeroCandidateCount: number;
  readonly trainingColumnRank: number;
  readonly packedNonzeroCandidateCount: number;
  readonly packedTrainingColumnRank: number;
}

function assertFiniteArray(
  values: ArrayLike<number>,
  expectedLength: number,
  label: string,
): void {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `${label} contains ${values.length} entries; expected ${expectedLength}.`,
    );
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${label}[${index}] must be finite.`);
    }
  }
}

function transpose3(matrix: ArrayLike<number>): Float64Array {
  return new Float64Array([
    matrix[0]!, matrix[3]!, matrix[6]!,
    matrix[1]!, matrix[4]!, matrix[7]!,
    matrix[2]!, matrix[5]!, matrix[8]!,
  ]);
}

function addScaled(
  destination: Float64Array,
  source: Float64Array,
  scale: number,
): void {
  for (let index = 0; index < destination.length; index += 1) {
    destination[index] += scale * source[index]!;
  }
}

function buildIncidentCounts(mesh: TetrahedralMesh): Uint32Array {
  const counts = new Uint32Array(getVertexCount(mesh));
  for (const vertex of mesh.tetrahedra) {
    counts[vertex] += 1;
  }
  for (let vertex = 0; vertex < counts.length; vertex += 1) {
    if (counts[vertex] === 0) {
      throw new Error(`Vertex ${vertex} has no incident Cubature element.`);
    }
  }
  return counts;
}

function validateContext(context: StableNeoHookeanCubatureContext): void {
  const vertexCount = getVertexCount(context.mesh);
  if (
    !Number.isSafeInteger(context.sourceVertex) ||
    context.sourceVertex < 0 ||
    context.sourceVertex >= vertexCount
  ) {
    throw new RangeError("The nonlinear Cubature source vertex is out of range.");
  }
  if (context.restSystem.vertexToActiveDof[context.sourceVertex]! < 0) {
    throw new RangeError("The nonlinear Cubature source vertex must be active.");
  }
  if (!(context.timestep > 0) || !Number.isFinite(context.timestep)) {
    throw new RangeError("The nonlinear Cubature timestep must be positive.");
  }
  assertFiniteArray(
    context.predictedPositions,
    vertexCount * 3,
    "Nonlinear Cubature predicted positions",
  );
  assertFiniteArray(
    context.exactBasis,
    vertexCount * 9,
    "Nonlinear Cubature exact basis",
  );
  if (context.lumpedMasses.length !== vertexCount) {
    throw new RangeError("Nonlinear Cubature requires one mass per vertex.");
  }
  const referencedMaterialIds = new Set(context.mesh.materialIds);
  for (const materialId of referencedMaterialIds) {
    const material = context.materials[materialId];
    if (!material) {
      throw new RangeError(
        `Nonlinear Cubature material ID ${materialId} is out of range.`,
      );
    }
    if (material.model !== "stable-neo-hookean") {
      throw new RangeError(
        `Nonlinear Cubature material ${material.name} must select stable-neo-hookean.`,
      );
    }
  }
}

/** Apply Eq. 14 blockwise: B_v = R_v Ubar_v R_i^T. */
export function buildCurrentCoRotatedEquilibriumBasis(
  context: StableNeoHookeanCubatureContext,
  positions: Float64Array,
): Float64Array {
  validateContext(context);
  const vertexCount = getVertexCount(context.mesh);
  assertFiniteArray(
    positions,
    vertexCount * 3,
    "Nonlinear Cubature current positions",
  );
  const stencil = buildVertexDeformationGradientStencil(
    context.mesh,
    context.restData,
  );
  const frames = computeVertexPolarFrames(stencil, positions);
  const inverseSourceFrame = transpose3(
    frames.subarray(
      context.sourceVertex * 9,
      context.sourceVertex * 9 + 9,
    ),
  );
  const basis = new Float64Array(vertexCount * 9);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const left = multiply3(
      frames.subarray(vertex * 9, vertex * 9 + 9),
      context.exactBasis.subarray(vertex * 9, vertex * 9 + 9),
    );
    basis.set(multiply3(left, inverseSourceFrame), vertex * 9);
  }
  return basis;
}

export function currentEquilibriumBasisToActiveMatrix(
  currentBasis: Float64Array,
  restSystem: RestLinearSystem,
): Float64Array {
  const vertexCount = restSystem.vertexToActiveDof.length;
  assertFiniteArray(
    currentBasis,
    vertexCount * 9,
    "Current equilibrium basis",
  );
  const active = new Float64Array(restSystem.dimension * 3);
  for (const vertex of restSystem.activeVertices) {
    const activeBase = restSystem.vertexToActiveDof[vertex]!;
    active.set(
      currentBasis.subarray(vertex * 9, vertex * 9 + 9),
      activeBase * 3,
    );
  }
  return active;
}

function copyRestBasisBlocks(
  mesh: TetrahedralMesh,
  exactBasis: Float64Array,
  tetrahedron: number,
): Float64Array {
  const blocks = new Float64Array(36);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    blocks.set(
      exactBasis.subarray(vertex * 9, vertex * 9 + 9),
      localVertex * 9,
    );
  }
  return blocks;
}

/** Apply Eq. 14 to the four stored Ubar blocks of one runtime Cubature record. */
export function buildCurrentCoRotatedCubatureBasisBlocks(
  context: StableNeoHookeanCubatureContext,
  positions: Float64Array,
  tetrahedron: number,
  restBasisBlocks: ArrayLike<number>,
): Float64Array {
  validateContext(context);
  const vertexCount = getVertexCount(context.mesh);
  assertFiniteArray(
    positions,
    vertexCount * 3,
    "Cubature basis current positions",
  );
  assertFiniteArray(restBasisBlocks, 36, "Cubature rest basis blocks");
  if (
    !Number.isSafeInteger(tetrahedron) ||
    tetrahedron < 0 ||
    tetrahedron >= getTetrahedronCount(context.mesh)
  ) {
    throw new RangeError("Cubature basis tetrahedron is out of range.");
  }

  const stencil = buildVertexDeformationGradientStencil(
    context.mesh,
    context.restData,
  );
  const frames = computeVertexPolarFrames(stencil, positions);
  const rest = Float64Array.from(restBasisBlocks);
  const inverseSourceFrame = transpose3(
    frames.subarray(
      context.sourceVertex * 9,
      context.sourceVertex * 9 + 9,
    ),
  );
  const current = new Float64Array(36);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = context.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    const left = multiply3(
      frames.subarray(vertex * 9, vertex * 9 + 9),
      rest.subarray(
        localVertex * 9,
        localVertex * 9 + 9,
      ),
    );
    current.set(multiply3(left, inverseSourceFrame), localVertex * 9);
  }
  return current;
}

export function evaluateStableNeoHookeanCubatureCandidate(
  context: StableNeoHookeanCubatureContext,
  positions: Float64Array,
  currentBasis: Float64Array,
  incidentCounts: Uint32Array,
  tetrahedron: number,
  currentBasisBlocksOverride?: Float64Array,
): NonlinearCubatureCandidateEvaluation {
  validateContext(context);
  const vertexCount = getVertexCount(context.mesh);
  assertFiniteArray(positions, vertexCount * 3, "Cubature candidate positions");
  assertFiniteArray(
    currentBasis,
    vertexCount * 9,
    "Cubature candidate current basis",
  );
  if (incidentCounts.length !== vertexCount) {
    throw new RangeError("Cubature incident counts have the wrong length.");
  }
  if (
    !Number.isSafeInteger(tetrahedron) ||
    tetrahedron < 0 ||
    tetrahedron >= getTetrahedronCount(context.mesh)
  ) {
    throw new RangeError("Cubature candidate tetrahedron is out of range.");
  }

  const element = evaluateStableNeoHookeanTetrahedron(
    context.mesh,
    context.restData,
    context.materials,
    tetrahedron,
    positions,
  );
  const elementGradient = element.gradient.slice();
  const elementHessian = element.hessian.slice();
  if (currentBasisBlocksOverride) {
    assertFiniteArray(
      currentBasisBlocksOverride,
      36,
      "Cubature candidate current basis blocks",
    );
  }
  const currentBasisBlocks =
    currentBasisBlocksOverride?.slice() ?? new Float64Array(36);
  const inverseTimestepSquared = 1 / (context.timestep * context.timestep);
  let sourceSlot = -1;
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = context.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    if (!currentBasisBlocksOverride) {
      currentBasisBlocks.set(
        currentBasis.subarray(vertex * 9, vertex * 9 + 9),
        localVertex * 9,
      );
    }
    const distributedInertia =
      (context.lumpedMasses[vertex]! * inverseTimestepSquared) /
      incidentCounts[vertex]!;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const local = localVertex * 3 + coordinate;
      elementGradient[local] +=
        distributedInertia *
        (positions[vertex * 3 + coordinate]! -
          context.predictedPositions[vertex * 3 + coordinate]!);
      elementHessian[local * ELEMENT_DIMENSION + local] += distributedInertia;
    }
    if (vertex === context.sourceVertex) sourceSlot = localVertex;
  }

  const projectedGradient = new Float64Array(3);
  const projectedHessian = new Float64Array(9);
  for (let reducedRow = 0; reducedRow < 3; reducedRow += 1) {
    for (let elementRow = 0; elementRow < ELEMENT_DIMENSION; elementRow += 1) {
      projectedGradient[reducedRow] +=
        currentBasisBlocks[elementRow * 3 + reducedRow]! *
        elementGradient[elementRow]!;
    }
    for (let reducedColumn = 0; reducedColumn < 3; reducedColumn += 1) {
      let value = 0;
      for (let elementRow = 0; elementRow < ELEMENT_DIMENSION; elementRow += 1) {
        for (
          let elementColumn = 0;
          elementColumn < ELEMENT_DIMENSION;
          elementColumn += 1
        ) {
          value +=
            currentBasisBlocks[elementRow * 3 + reducedRow]! *
            elementHessian[elementRow * ELEMENT_DIMENSION + elementColumn]! *
            currentBasisBlocks[elementColumn * 3 + reducedColumn]!;
        }
      }
      projectedHessian[reducedRow * 3 + reducedColumn] = value;
    }
  }

  const sourceGradient = new Float64Array(3);
  const sourceHessian = new Float64Array(9);
  if (sourceSlot >= 0) {
    sourceGradient.set(
      elementGradient.subarray(sourceSlot * 3, sourceSlot * 3 + 3),
    );
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        sourceHessian[row * 3 + column] =
          elementHessian[
            (sourceSlot * 3 + row) * ELEMENT_DIMENSION +
              sourceSlot * 3 +
              column
          ]!;
      }
    }
  }
  const remainderGradient = projectedGradient.slice();
  const remainderHessian = projectedHessian.slice();
  addScaled(remainderGradient, sourceGradient, -1);
  addScaled(remainderHessian, sourceHessian, -1);
  return {
    tetrahedron,
    currentBasisBlocks,
    projectedGradient,
    projectedHessian,
    remainderGradient,
    remainderHessian,
    sourceGradient,
    sourceHessian,
    deformationDeterminant: element.deformationDeterminant,
  };
}

function evaluateAllCandidates(
  context: StableNeoHookeanCubatureContext,
  positions: Float64Array,
): readonly NonlinearCubatureCandidateEvaluation[] {
  const incidentCounts = buildIncidentCounts(context.mesh);
  const currentBasis = buildCurrentCoRotatedEquilibriumBasis(context, positions);
  return Array.from(
    { length: getTetrahedronCount(context.mesh) },
    (_unused, tetrahedron) =>
      evaluateStableNeoHookeanCubatureCandidate(
        context,
        positions,
        currentBasis,
        incidentCounts,
        tetrahedron,
      ),
  );
}

function buildStackedTrainingSystem(
  context: StableNeoHookeanCubatureContext,
  trainingPoses: readonly Pick<NonlinearCubaturePose, "positions">[],
  normalizationNorms?: Float64Array,
): {
  readonly columns: readonly Float64Array[];
  readonly target: Float64Array;
  readonly poseTargetNorms: Float64Array;
  readonly validPoseCount: number;
  readonly trivialPoseCount: number;
} {
  const tetrahedronCount = getTetrahedronCount(context.mesh);
  const columns = Array.from(
    { length: tetrahedronCount },
    () => new Float64Array(trainingPoses.length * 3),
  );
  const target = new Float64Array(trainingPoses.length * 3);
  const poseTargetNorms = new Float64Array(trainingPoses.length);
  if (
    normalizationNorms !== undefined &&
    normalizationNorms.length !== trainingPoses.length
  ) {
    throw new RangeError(
      "Nonlinear Cubature normalization norms have the wrong length.",
    );
  }
  let validPoseCount = 0;
  let trivialPoseCount = 0;
  for (let poseIndex = 0; poseIndex < trainingPoses.length; poseIndex += 1) {
    const candidates = evaluateAllCandidates(
      context,
      trainingPoses[poseIndex]!.positions,
    );
    const poseTarget = new Float64Array(3);
    for (const candidate of candidates) {
      addScaled(poseTarget, candidate.remainderGradient, 1);
    }
    const candidateScale = Math.sqrt(
      candidates.reduce(
        (sum, candidate) => sum + squaredNorm(candidate.remainderGradient),
        0,
      ),
    );
    const poseNorm = Math.sqrt(squaredNorm(poseTarget));
    poseTargetNorms[poseIndex] = poseNorm;
    if (candidateScale === 0) {
      trivialPoseCount += 1;
      continue;
    }
    if (poseNorm <= 1e-12 * candidateScale) {
      throw new Error(
        `Nonlinear Cubature training pose ${poseIndex} has a cancellation-` +
          `dominated complementary target.`,
      );
    }
    validPoseCount += 1;
    const normalizationNorm = normalizationNorms?.[poseIndex] ?? poseNorm;
    if (!(normalizationNorm > 0) || !Number.isFinite(normalizationNorm)) {
      throw new Error(
        `Nonlinear Cubature training pose ${poseIndex} has an invalid ` +
          `normalization norm.`,
      );
    }
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const row = poseIndex * 3 + coordinate;
      target[row] = poseTarget[coordinate]! / normalizationNorm;
      for (const candidate of candidates) {
        columns[candidate.tetrahedron]![row] =
          candidate.remainderGradient[coordinate]! / normalizationNorm;
      }
    }
  }
  return {
    columns,
    target,
    poseTargetNorms,
    validPoseCount,
    trivialPoseCount,
  };
}

function selectedResidual(
  columns: readonly Float64Array[],
  target: Float64Array,
  tetrahedra: readonly number[],
  weights: Float64Array,
): number {
  const residual = target.slice();
  for (let sample = 0; sample < tetrahedra.length; sample += 1) {
    addScaled(residual, columns[tetrahedra[sample]!]!, -weights[sample]!);
  }
  const targetNorm = Math.sqrt(squaredNorm(target));
  return targetNorm > 1e-12
    ? Math.sqrt(squaredNorm(residual)) / targetNorm
    : 0;
}

function numericalColumnRank(
  columns: readonly Float64Array[],
  relativeTolerance = NONLINEAR_CUBATURE_COLUMN_RANK_RELATIVE_TOLERANCE,
): { readonly nonzeroCount: number; readonly rank: number } {
  const maximumNorm = Math.max(
    0,
    ...columns.map((column) => Math.sqrt(squaredNorm(column))),
  );
  if (maximumNorm === 0) return { nonzeroCount: 0, rank: 0 };
  const threshold = relativeTolerance * maximumNorm;
  const nonzeroCount = columns.filter(
    (column) => Math.sqrt(squaredNorm(column)) > threshold,
  ).length;
  const basis: Float64Array[] = [];
  for (const column of columns) {
    const candidate = column.slice();
    for (let pass = 0; pass < 2; pass += 1) {
      for (const direction of basis) {
        let projection = 0;
        for (let row = 0; row < candidate.length; row += 1) {
          projection += candidate[row]! * direction[row]!;
        }
        for (let row = 0; row < candidate.length; row += 1) {
          candidate[row] -= projection * direction[row]!;
        }
      }
    }
    const norm = Math.sqrt(squaredNorm(candidate));
    if (norm <= threshold) continue;
    for (let row = 0; row < candidate.length; row += 1) {
      candidate[row] /= norm;
    }
    basis.push(candidate);
  }
  return { nonzeroCount, rank: basis.length };
}

/** Assemble exact-source plus exact-all or selected-Cubature remainder. */
export function assembleStableNeoHookeanJGS2LocalSystem(
  context: StableNeoHookeanCubatureContext,
  positions: Float64Array,
  samples?: readonly (
    Pick<CubatureSample, "tetrahedron" | "weight"> & {
      readonly basisBlocks?: ArrayLike<number>;
    }
  )[],
): StableNeoHookeanJGS2LocalSystem {
  const candidates = evaluateAllCandidates(context, positions);
  const sourceGradient = new Float64Array(3);
  const sourceHessian = new Float64Array(9);
  for (const candidate of candidates) {
    addScaled(sourceGradient, candidate.sourceGradient, 1);
    addScaled(sourceHessian, candidate.sourceHessian, 1);
  }

  const remainderGradient = new Float64Array(3);
  const remainderHessian = new Float64Array(9);
  if (samples === undefined) {
    for (const candidate of candidates) {
      addScaled(remainderGradient, candidate.remainderGradient, 1);
      addScaled(remainderHessian, candidate.remainderHessian, 1);
    }
  } else {
    const used = new Set<number>();
    const incidentCounts = buildIncidentCounts(context.mesh);
    const currentBasis = buildCurrentCoRotatedEquilibriumBasis(
      context,
      positions,
    );
    for (const sample of samples) {
      if (
        !Number.isSafeInteger(sample.tetrahedron) ||
        sample.tetrahedron < 0 ||
        sample.tetrahedron >= candidates.length ||
        used.has(sample.tetrahedron)
      ) {
        throw new RangeError("Selected nonlinear Cubature tetrahedra must be unique and in range.");
      }
      if (!(sample.weight >= 0) || !Number.isFinite(sample.weight)) {
        throw new RangeError("Selected nonlinear Cubature weights must be finite and nonnegative.");
      }
      used.add(sample.tetrahedron);
      const candidate = sample.basisBlocks
        ? evaluateStableNeoHookeanCubatureCandidate(
            context,
            positions,
            currentBasis,
            incidentCounts,
            sample.tetrahedron,
            buildCurrentCoRotatedCubatureBasisBlocks(
              context,
              positions,
              sample.tetrahedron,
              sample.basisBlocks,
            ),
          )
        : candidates[sample.tetrahedron]!;
      addScaled(
        remainderGradient,
        candidate.remainderGradient,
        sample.weight,
      );
      addScaled(
        remainderHessian,
        candidate.remainderHessian,
        sample.weight,
      );
    }
  }
  const gradient = sourceGradient.slice();
  const hessian = sourceHessian.slice();
  addScaled(gradient, remainderGradient, 1);
  addScaled(hessian, remainderHessian, 1);
  const newtonUpdate = solveDenseLinearSystem(
    hessian,
    Float64Array.from(gradient, (value) => -value),
    LOCAL_DIMENSION,
  );
  return {
    sourceGradient,
    sourceHessian,
    remainderGradient,
    remainderHessian,
    gradient,
    hessian,
    newtonUpdate,
    candidates,
    minimumDeformationDeterminant: Math.min(
      ...candidates.map((candidate) => candidate.deformationDeterminant),
    ),
  };
}

/** Train Eq. 18 on nonlinear current poses using current co-rotated bases. */
export function selectStableNeoHookeanCubatureSamples(
  context: StableNeoHookeanCubatureContext,
  trainingPoses: readonly Pick<NonlinearCubaturePose, "positions">[],
  maximumSamples: number,
): StableNeoHookeanCubatureSelection {
  validateContext(context);
  if (!Number.isSafeInteger(maximumSamples) || maximumSamples < 1) {
    throw new RangeError("Nonlinear Cubature sample count must be positive.");
  }
  if (trainingPoses.length === 0) {
    throw new RangeError("Nonlinear Cubature requires at least one training pose.");
  }
  const tetrahedronCount = getTetrahedronCount(context.mesh);
  const {
    columns,
    target,
    poseTargetNorms,
    validPoseCount,
    trivialPoseCount,
  } =
    buildStackedTrainingSystem(
    context,
    trainingPoses,
    );
  if (Math.sqrt(squaredNorm(target)) <= 1e-12) {
    throw new Error(
      `Nonlinear Cubature source vertex ${context.sourceVertex} has no valid ` +
        `nontrivial training target.`,
    );
  }
  const selection = selectGreedyNonnegativeColumns({
    columns,
    target,
    maximumColumns: maximumSamples,
    normalizedResidualTolerance:
      NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
    refinementPasses: 12,
  });
  let selectedTetrahedra = [...selection.selectedColumnIndices];
  let selectedWeights = selection.weights;
  let normalizedResidual = selection.normalizedResidual;
  const columnDiagnostics = numericalColumnRank(columns);

  // The checked-in parity fixtures are deliberately tiny. Exhaustively audit
  // their six-column subset after greedy selection so a local greedy minimum
  // cannot be mistaken for failure of the paper's 1% representability claim.
  if (tetrahedronCount <= 12 && maximumSamples < tetrahedronCount) {
    const subsetSize = Math.min(maximumSamples, tetrahedronCount);
    const subset: number[] = [];
    const visit = (next: number): void => {
      if (subset.length === subsetSize) {
        const fit = fitNonnegativeColumnWeights(
          subset.map((tetrahedron) => columns[tetrahedron]!),
          target,
        );
        if (fit.normalizedResidual < normalizedResidual - 1e-14) {
          normalizedResidual = fit.normalizedResidual;
          selectedTetrahedra = subset.slice();
          selectedWeights = fit.weights;
        }
        return;
      }
      const remaining = subsetSize - subset.length;
      for (
        let tetrahedron = next;
        tetrahedron <= tetrahedronCount - remaining;
        tetrahedron += 1
      ) {
        subset.push(tetrahedron);
        visit(tetrahedron + 1);
        subset.pop();
      }
    };
    visit(0);
  }

  const positive = selectedTetrahedra
    .map((tetrahedron, index) => ({
      tetrahedron,
      weight: selectedWeights[index]!,
    }))
    .filter(({ weight }) => weight > 1e-10);
  selectedTetrahedra = positive.map(({ tetrahedron }) => tetrahedron);
  selectedWeights = Float64Array.from(positive, ({ weight }) => weight);
  const samples = selectedTetrahedra.map(
    (tetrahedron, index): CubatureSample => ({
      tetrahedron,
      weight: selectedWeights[index]!,
      basisBlocks: copyRestBasisBlocks(
        context.mesh,
        context.exactBasis,
        tetrahedron,
      ),
    }),
  );
  const packedContext: StableNeoHookeanCubatureContext = {
    ...context,
    exactBasis: Float64Array.from(Float32Array.from(context.exactBasis)),
  };
  const packedWeights = Float64Array.from(
    Float32Array.from(selectedWeights),
  );
  const packedTraining = buildStackedTrainingSystem(
    packedContext,
    trainingPoses,
    poseTargetNorms,
  );
  const packedColumnDiagnostics = numericalColumnRank(
    packedTraining.columns,
  );
  if (
    packedTraining.validPoseCount !== validPoseCount ||
    packedTraining.trivialPoseCount !== trivialPoseCount
  ) {
    throw new Error(
      `Packing changed nonlinear Cubature pose validity for source vertex ` +
        `${context.sourceVertex}.`,
    );
  }
  const packedSamples = selectedTetrahedra.map(
    (tetrahedron, index): CubatureSample => ({
      tetrahedron,
      weight: packedWeights[index]!,
      basisBlocks: Float64Array.from(
        Float32Array.from(
          copyRestBasisBlocks(
            context.mesh,
            context.exactBasis,
            tetrahedron,
          ),
        ),
      ),
    }),
  );
  return {
    samples,
    packedSamples,
    residual: normalizedResidual,
    packedResidual: selectedResidual(
      packedTraining.columns,
      target,
      selectedTetrahedra,
      packedWeights,
    ),
    selectedTetrahedra,
    weights: selectedWeights,
    stackedTargetNorm: selection.targetNorm,
    validTrainingPoseCount: validPoseCount,
    trivialTrainingPoseCount: trivialPoseCount,
    nonzeroCandidateCount: columnDiagnostics.nonzeroCount,
    trainingColumnRank: columnDiagnostics.rank,
    packedNonzeroCandidateCount: packedColumnDiagnostics.nonzeroCount,
    packedTrainingColumnRank: packedColumnDiagnostics.rank,
  };
}
