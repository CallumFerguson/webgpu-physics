import {
  dot,
  solveCholesky,
  solveDenseLinearSystem,
  squaredNorm,
} from "./math";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import type {
  CubatureSample,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

interface IndexedCandidateColumn {
  /** Original input order. It is also the deterministic tie-break key. */
  readonly sourceIndex: number;
  readonly values: Float64Array;
}

export interface GreedyNonnegativeColumnSelectionOptions {
  /** Candidate columns in deterministic tie-break order. */
  readonly columns: readonly ArrayLike<number>[];
  /** The stacked training target approximated by a nonnegative column sum. */
  readonly target: ArrayLike<number>;
  readonly maximumColumns: number;
  /** Stop growing once ||residual|| / ||target|| is below this value. */
  readonly normalizedResidualTolerance?: number;
  /** Deterministic one-for-one replacement passes after greedy growth. */
  readonly refinementPasses?: number;
}

export interface GreedyNonnegativeColumnSelection {
  /** Indices into the original `columns` array, in selected-slot order. */
  readonly selectedColumnIndices: readonly number[];
  /** One nonnegative weight per selected column. */
  readonly weights: Float64Array;
  readonly targetNorm: number;
  readonly residualNorm: number;
  readonly normalizedResidual: number;
  readonly residualVector: Float64Array;
}

export interface NonnegativeColumnFit {
  readonly weights: Float64Array;
  readonly residualVector: Float64Array;
  readonly residualNorm: number;
  readonly normalizedResidual: number;
}

const DEFAULT_NORMALIZED_RESIDUAL_TOLERANCE = 0.01;
const DEFAULT_REFINEMENT_PASSES = 4;
const MAXIMUM_EXHAUSTIVE_NNLS_COLUMNS = 12;

function canonicalizeVectorSign(vector: Float64Array): void {
  let largestIndex = 0;
  for (let index = 1; index < vector.length; index += 1) {
    if (Math.abs(vector[index]!) > Math.abs(vector[largestIndex]!)) {
      largestIndex = index;
    }
  }
  if (vector[largestIndex]! < 0) {
    for (let index = 0; index < vector.length; index += 1) {
      vector[index] *= -1;
    }
  }
}

function orthonormalizeBlock(
  candidates: readonly Float64Array[],
  dimension: number,
): Float64Array[] {
  const result: Float64Array[] = [];

  for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
    let candidate: Float64Array = candidates[candidateIndex]!.slice();
    for (let pass = 0; pass < 2; pass += 1) {
      for (const basis of result) {
        const projection = dot(candidate, basis);
        for (let row = 0; row < dimension; row += 1) {
          candidate[row] -= projection * basis[row]!;
        }
      }
    }

    let norm = Math.sqrt(squaredNorm(candidate));
    if (norm <= 1e-11) {
      let replacement: Float64Array | undefined;
      for (let offset = 0; offset < dimension; offset += 1) {
        const trial = new Float64Array(dimension);
        trial[(candidateIndex + offset) % dimension] = 1;
        for (const basis of result) {
          const projection = dot(trial, basis);
          for (let row = 0; row < dimension; row += 1) {
            trial[row] -= projection * basis[row]!;
          }
        }
        const trialNorm = Math.sqrt(squaredNorm(trial));
        if (trialNorm > 1e-11) {
          for (let row = 0; row < dimension; row += 1) {
            trial[row] /= trialNorm;
          }
          replacement = trial;
          break;
        }
      }
      if (!replacement) {
        throw new Error("Could not construct the requested low-frequency basis.");
      }
      candidate = replacement;
      norm = 1;
    }

    for (let row = 0; row < dimension; row += 1) {
      candidate[row] /= norm;
    }
    canonicalizeVectorSign(candidate);
    result.push(candidate);
  }

  return result;
}

interface SymmetricEigendecomposition {
  readonly values: Float64Array;
  /** Eigenvectors are stored in columns. */
  readonly vectors: Float64Array;
}

function jacobiSymmetricEigendecomposition(
  input: Float64Array,
  dimension: number,
): SymmetricEigendecomposition {
  const matrix = input.slice();
  const vectors = new Float64Array(dimension * dimension);
  for (let index = 0; index < dimension; index += 1) {
    vectors[index * dimension + index] = 1;
  }

  for (let iteration = 0; iteration < dimension * dimension * 16; iteration += 1) {
    let pivotRow = 0;
    let pivotColumn = 1;
    let largest = 0;
    for (let row = 0; row < dimension; row += 1) {
      for (let column = row + 1; column < dimension; column += 1) {
        const magnitude = Math.abs(matrix[row * dimension + column]!);
        if (magnitude > largest) {
          largest = magnitude;
          pivotRow = row;
          pivotColumn = column;
        }
      }
    }
    const diagonalScale = Math.max(
      Math.abs(matrix[pivotRow * dimension + pivotRow]!),
      Math.abs(matrix[pivotColumn * dimension + pivotColumn]!),
      1,
    );
    if (largest <= 1e-12 * diagonalScale) {
      break;
    }

    const app = matrix[pivotRow * dimension + pivotRow]!;
    const aqq = matrix[pivotColumn * dimension + pivotColumn]!;
    const apq = matrix[pivotRow * dimension + pivotColumn]!;
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);

    for (let index = 0; index < dimension; index += 1) {
      if (index === pivotRow || index === pivotColumn) {
        continue;
      }
      const aip = matrix[index * dimension + pivotRow]!;
      const aiq = matrix[index * dimension + pivotColumn]!;
      const rotatedP = cosine * aip - sine * aiq;
      const rotatedQ = sine * aip + cosine * aiq;
      matrix[index * dimension + pivotRow] = rotatedP;
      matrix[pivotRow * dimension + index] = rotatedP;
      matrix[index * dimension + pivotColumn] = rotatedQ;
      matrix[pivotColumn * dimension + index] = rotatedQ;
    }
    matrix[pivotRow * dimension + pivotRow] =
      cosine * cosine * app -
      2 * sine * cosine * apq +
      sine * sine * aqq;
    matrix[pivotColumn * dimension + pivotColumn] =
      sine * sine * app +
      2 * sine * cosine * apq +
      cosine * cosine * aqq;
    matrix[pivotRow * dimension + pivotColumn] = 0;
    matrix[pivotColumn * dimension + pivotRow] = 0;

    for (let row = 0; row < dimension; row += 1) {
      const vip = vectors[row * dimension + pivotRow]!;
      const viq = vectors[row * dimension + pivotColumn]!;
      vectors[row * dimension + pivotRow] = cosine * vip - sine * viq;
      vectors[row * dimension + pivotColumn] = sine * vip + cosine * viq;
    }
  }

  const ordering = Array.from({ length: dimension }, (_unused, index) => index).sort(
    (left, right) =>
      matrix[left * dimension + left]! - matrix[right * dimension + right]! ||
      left - right,
  );
  const values = new Float64Array(dimension);
  const orderedVectors = new Float64Array(dimension * dimension);
  for (let column = 0; column < dimension; column += 1) {
    const source = ordering[column]!;
    values[column] = matrix[source * dimension + source]!;
    for (let row = 0; row < dimension; row += 1) {
      orderedVectors[row * dimension + column] =
        vectors[row * dimension + source]!;
    }
  }
  return { values, vectors: orderedVectors };
}

/**
 * Approximate the lowest rest-Hessian eigenvectors with deterministic block
 * inverse iteration, then perform a Rayleigh-Ritz extraction in that subspace.
 */
export function buildLowFrequencyTrainingPoses(
  mesh: TetrahedralMesh,
  system: RestLinearSystem,
  requestedModeCount = 12,
  inverseIterations = 10,
): readonly Float64Array[] {
  const modeCount = Math.min(requestedModeCount, system.dimension);
  let activeModes: Float64Array[] = Array.from(
    { length: modeCount },
    (_unused, mode) => {
    const vector = new Float64Array(system.dimension);
    for (let row = 0; row < system.dimension; row += 1) {
      vector[row] =
        Math.sin((row + 1) * (mode + 1) * 0.754_877_666) +
        0.5 * Math.cos((row + 1) * (mode + 3) * 0.569_840_291);
    }
      return vector;
    },
  );
  activeModes = orthonormalizeBlock(activeModes, system.dimension);

  for (let iteration = 0; iteration < inverseIterations; iteration += 1) {
    activeModes = orthonormalizeBlock(
      activeModes.map((mode) =>
        solveCholesky(system.choleskyLower, mode, system.dimension),
      ),
      system.dimension,
    );
  }

  const projectedHessian = new Float64Array(modeCount * modeCount);
  const hessianTimesMode = activeModes.map((mode) => {
    const result = new Float64Array(system.dimension);
    for (let row = 0; row < system.dimension; row += 1) {
      for (let column = 0; column < system.dimension; column += 1) {
        result[row] +=
          system.hessian[row * system.dimension + column]! * mode[column]!;
      }
    }
    return result;
  });
  for (let row = 0; row < modeCount; row += 1) {
    for (let column = 0; column < modeCount; column += 1) {
      projectedHessian[row * modeCount + column] = dot(
        activeModes[row]!,
        hessianTimesMode[column]!,
      );
    }
  }
  const ritz = jacobiSymmetricEigendecomposition(projectedHessian, modeCount);
  const orderedModes = Array.from({ length: modeCount }, (_unused, mode) => {
    const vector = new Float64Array(system.dimension);
    for (let basis = 0; basis < modeCount; basis += 1) {
      const coefficient = ritz.vectors[basis * modeCount + mode]!;
      for (let row = 0; row < system.dimension; row += 1) {
        vector[row] += activeModes[basis]![row]! * coefficient;
      }
    }
    canonicalizeVectorSign(vector);
    return vector;
  });

  return orderedModes.map((activeMode) => {
    const fullMode = new Float64Array(getVertexCount(mesh) * 3);
    for (const vertex of system.activeVertices) {
      const activeBase = system.vertexToActiveDof[vertex]!;
      fullMode.set(
        activeMode.subarray(activeBase, activeBase + 3),
        vertex * 3,
      );
    }
    return fullMode;
  });
}

function localVertexSlot(
  mesh: TetrahedralMesh,
  tetrahedron: number,
  vertex: number,
): number {
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    if (mesh.tetrahedra[tetrahedron * 4 + localVertex] === vertex) {
      return localVertex;
    }
  }
  return -1;
}

function evaluateReducedTetGradient(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  lumpedMasses: Float64Array,
  inverseTimestepSquared: number,
  incidentCounts: Uint32Array,
  sourceVertex: number,
  exactBasis: Float64Array,
  pose: Float64Array,
  tetrahedron: number,
): Float64Array {
  const elasticGradient = new Float64Array(12);
  const stiffnessOffset = tetrahedron * 144;

  for (let row = 0; row < 12; row += 1) {
    let value = 0;
    for (let column = 0; column < 12; column += 1) {
      const vertex = mesh.tetrahedra[
        tetrahedron * 4 + Math.floor(column / 3)
      ]!;
      const coordinate = column % 3;
      value +=
        restData.stiffnessMatrices[
          stiffnessOffset + row * 12 + column
        ]! * pose[vertex * 3 + coordinate]!;
      if (row === column) {
        value +=
          (lumpedMasses[vertex]! * inverseTimestepSquared *
            pose[vertex * 3 + coordinate]!) /
          incidentCounts[vertex]!;
      }
    }
    elasticGradient[row] = value;
  }

  const reduced = new Float64Array(3);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    const basisOffset = vertex * 9;
    for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
      const gradientValue = elasticGradient[localVertex * 3 + rowCoordinate]!;
      for (let column = 0; column < 3; column += 1) {
        reduced[column] +=
          exactBasis[basisOffset + rowCoordinate * 3 + column]! * gradientValue;
      }
    }
  }

  const sourceSlot = localVertexSlot(mesh, tetrahedron, sourceVertex);
  if (sourceSlot >= 0) {
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      reduced[coordinate] -=
        elasticGradient[sourceSlot * 3 + coordinate]!;
    }
  }

  return reduced;
}

function computeResidual(
  columns: readonly Float64Array[],
  weights: ArrayLike<number>,
  target: Float64Array,
): Float64Array {
  const residual = target.slice();
  for (let column = 0; column < columns.length; column += 1) {
    const weight = weights[column]!;
    for (let row = 0; row < target.length; row += 1) {
      residual[row] -= columns[column]![row]! * weight;
    }
  }
  return residual;
}

/** Exhaustive active-set NNLS for the selector's deliberately small sets. */
function solveSmallNnls(
  columns: readonly Float64Array[],
  target: Float64Array,
): Float64Array {
  const columnCount = columns.length;
  const bestWeights = new Float64Array(columnCount);
  let bestError = squaredNorm(target);

  for (let mask = 1; mask < 1 << columnCount; mask += 1) {
    const active: number[] = [];
    for (let column = 0; column < columnCount; column += 1) {
      if ((mask & (1 << column)) !== 0) {
        active.push(column);
      }
    }

    const dimension = active.length;
    const normalMatrix = new Float64Array(dimension * dimension);
    const normalRightHandSide = new Float64Array(dimension);
    for (let row = 0; row < dimension; row += 1) {
      const rowColumn = columns[active[row]!]!;
      normalRightHandSide[row] = dot(rowColumn, target);
      for (let column = 0; column < dimension; column += 1) {
        normalMatrix[row * dimension + column] = dot(
          rowColumn,
          columns[active[column]!]!,
        );
      }
    }

    const solution = solveDenseLinearSystem(
      normalMatrix,
      normalRightHandSide,
      dimension,
    );
    if (!solution || solution.some((weight) => weight < -1e-10)) {
      continue;
    }

    const candidateWeights = new Float64Array(columnCount);
    for (let index = 0; index < active.length; index += 1) {
      candidateWeights[active[index]!] = Math.max(0, solution[index]!);
    }
    const error = squaredNorm(
      computeResidual(columns, candidateWeights, target),
    );
    if (error < bestError - 1e-14 * Math.max(1, bestError)) {
      bestError = error;
      bestWeights.set(candidateWeights);
    }
  }

  return bestWeights;
}

function requireNonnegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a nonnegative safe integer.`);
  }
  return value;
}

function requireFiniteVector(
  value: ArrayLike<number>,
  expectedLength: number | undefined,
  label: string,
): Float64Array {
  if (expectedLength === undefined && value.length === 0) {
    throw new RangeError(`${label} must not be empty.`);
  }
  if (expectedLength !== undefined && value.length !== expectedLength) {
    throw new RangeError(
      `${label} must contain ${expectedLength} entries; received ${value.length}.`,
    );
  }
  // Reuse the native training representation without duplicating potentially
  // large stacked column matrices. The selector never mutates its inputs.
  const result =
    value instanceof Float64Array ? value : Float64Array.from(value);
  for (let index = 0; index < result.length; index += 1) {
    if (!Number.isFinite(result[index])) {
      throw new RangeError(`${label}[${index}] must be finite.`);
    }
  }
  return result;
}

/** Exact active-set NNLS for a caller-supplied set of at most 12 columns. */
export function fitNonnegativeColumnWeights(
  columns: readonly ArrayLike<number>[],
  targetValues: ArrayLike<number>,
): NonnegativeColumnFit {
  const target = requireFiniteVector(targetValues, undefined, "NNLS target");
  if (columns.length > MAXIMUM_EXHAUSTIVE_NNLS_COLUMNS) {
    throw new RangeError(
      `Exact NNLS supports at most ${MAXIMUM_EXHAUSTIVE_NNLS_COLUMNS} columns.`,
    );
  }
  const finiteColumns = columns.map((column, index) =>
    requireFiniteVector(column, target.length, `NNLS column ${index}`),
  );
  const weights = solveSmallNnls(finiteColumns, target);
  const residualVector = computeResidual(finiteColumns, weights, target);
  const residualNorm = Math.sqrt(squaredNorm(residualVector));
  const targetNorm = Math.sqrt(squaredNorm(target));
  return {
    weights,
    residualVector,
    residualNorm,
    normalizedResidual: targetNorm > 1e-12 ? residualNorm / targetNorm : 0,
  };
}

/**
 * Deterministic greedy nonnegative column selection for stacked training data.
 *
 * Candidate order is semantically significant only for exact numerical ties:
 * the lowest original index wins. Zero columns are ignored. The returned
 * normalized residual is zero for a numerically zero target and one when a
 * nonzero target has no usable candidate columns.
 */
export function selectGreedyNonnegativeColumns(
  options: GreedyNonnegativeColumnSelectionOptions,
): GreedyNonnegativeColumnSelection {
  const target = requireFiniteVector(
    options.target,
    undefined,
    "Selection target",
  );
  const maximumColumns = requireNonnegativeInteger(
    options.maximumColumns,
    "maximumColumns",
  );
  const refinementPasses = requireNonnegativeInteger(
    options.refinementPasses ?? DEFAULT_REFINEMENT_PASSES,
    "refinementPasses",
  );
  const normalizedResidualTolerance =
    options.normalizedResidualTolerance ??
    DEFAULT_NORMALIZED_RESIDUAL_TOLERANCE;
  if (
    !Number.isFinite(normalizedResidualTolerance) ||
    normalizedResidualTolerance < 0 ||
    normalizedResidualTolerance > 1
  ) {
    throw new RangeError(
      "normalizedResidualTolerance must be finite and in [0, 1].",
    );
  }

  const candidates: IndexedCandidateColumn[] = options.columns
    .map((column, sourceIndex) => ({
      sourceIndex,
      values: requireFiniteVector(
        column,
        target.length,
        `Selection column ${sourceIndex}`,
      ),
    }))
    .filter((candidate) => squaredNorm(candidate.values) > 1e-18);
  const selectionLimit = Math.min(maximumColumns, candidates.length);
  if (selectionLimit > MAXIMUM_EXHAUSTIVE_NNLS_COLUMNS) {
    throw new RangeError(
      `Exhaustive NNLS supports at most ${MAXIMUM_EXHAUSTIVE_NNLS_COLUMNS} selected columns.`,
    );
  }

  const targetNorm = Math.sqrt(squaredNorm(target));
  if (targetNorm <= 1e-12) {
    return {
      selectedColumnIndices: [],
      weights: new Float64Array(0),
      targetNorm,
      residualNorm: targetNorm,
      normalizedResidual: 0,
      residualVector: target.slice(),
    };
  }
  if (candidates.length === 0 || selectionLimit === 0) {
    return {
      selectedColumnIndices: [],
      weights: new Float64Array(0),
      targetNorm,
      residualNorm: targetNorm,
      normalizedResidual: 1,
      residualVector: target.slice(),
    };
  }

  const selected: IndexedCandidateColumn[] = [];
  let weights: Float64Array = new Float64Array(0);
  let residual: Float64Array = target.slice();

  for (let selection = 0; selection < selectionLimit; selection += 1) {
    let best: IndexedCandidateColumn | undefined;
    let bestWeights: Float64Array | undefined;
    let bestResidual: Float64Array | undefined;
    let bestError = squaredNorm(residual);
    for (const candidate of candidates) {
      if (selected.includes(candidate)) {
        continue;
      }
      const trialSelection = [...selected, candidate];
      const trialColumns = trialSelection.map((entry) => entry.values);
      const trialWeights = solveSmallNnls(trialColumns, target);
      const trialResidual = computeResidual(
        trialColumns,
        trialWeights,
        target,
      );
      const trialError = squaredNorm(trialResidual);
      if (
        trialError < bestError - 1e-14 * Math.max(1, bestError) ||
        (Math.abs(trialError - bestError) <=
          1e-14 * Math.max(1, bestError) &&
          best !== undefined &&
          candidate.sourceIndex < best.sourceIndex)
      ) {
        best = candidate;
        bestWeights = trialWeights;
        bestResidual = trialResidual;
        bestError = trialError;
      }
    }

    if (!best || !bestWeights || !bestResidual) {
      break;
    }
    selected.push(best);
    weights = bestWeights;
    residual = bestResidual;
    if (
      Math.sqrt(squaredNorm(residual)) / targetNorm <
      normalizedResidualTolerance
    ) {
      break;
    }
  }

  for (
    let refinement = 0;
    refinement < refinementPasses && selected.length > 0;
    refinement += 1
  ) {
    let replacementSlot = -1;
    let replacementCandidate: IndexedCandidateColumn | undefined;
    let replacementWeights: Float64Array | undefined;
    let replacementResidual: Float64Array | undefined;
    let replacementError = squaredNorm(residual);

    for (let slot = 0; slot < selected.length; slot += 1) {
      for (const candidate of candidates) {
        if (selected.includes(candidate)) {
          continue;
        }
        const trialSelection = selected.slice();
        trialSelection[slot] = candidate;
        const trialColumns = trialSelection.map((entry) => entry.values);
        const trialWeights = solveSmallNnls(trialColumns, target);
        const trialResidual = computeResidual(
          trialColumns,
          trialWeights,
          target,
        );
        const trialError = squaredNorm(trialResidual);
        if (
          trialError <
            replacementError - 1e-14 * Math.max(1, replacementError) ||
          (Math.abs(trialError - replacementError) <=
            1e-14 * Math.max(1, replacementError) &&
            replacementCandidate !== undefined &&
            candidate.sourceIndex < replacementCandidate.sourceIndex)
        ) {
          replacementSlot = slot;
          replacementCandidate = candidate;
          replacementWeights = trialWeights;
          replacementResidual = trialResidual;
          replacementError = trialError;
        }
      }
    }

    if (
      replacementSlot < 0 ||
      !replacementCandidate ||
      !replacementWeights ||
      !replacementResidual
    ) {
      break;
    }
    selected[replacementSlot] = replacementCandidate;
    weights = replacementWeights;
    residual = replacementResidual;
  }

  const residualNorm = Math.sqrt(squaredNorm(residual));
  return {
    selectedColumnIndices: selected.map((candidate) => candidate.sourceIndex),
    weights,
    targetNorm,
    residualNorm,
    normalizedResidual: residualNorm / targetNorm,
    residualVector: residual,
  };
}

function copyTetBasisBlocks(
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

export interface CubatureSelection {
  readonly samples: readonly CubatureSample[];
  readonly residual: number;
}

export function selectCubatureSamples(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  lumpedMasses: Float64Array,
  timestep: number,
  vertex: number,
  exactBasis: Float64Array,
  trainingPoses: readonly Float64Array[],
  maximumSamples: number,
): CubatureSelection {
  if (lumpedMasses.length !== getVertexCount(mesh)) {
    throw new RangeError("Cubature masses must have one entry per vertex.");
  }
  if (!(timestep > 0) || !Number.isFinite(timestep)) {
    throw new RangeError("Cubature timestep must be finite and positive.");
  }
  const incidentCounts = new Uint32Array(getVertexCount(mesh));
  for (const tetVertex of mesh.tetrahedra) {
    incidentCounts[tetVertex] += 1;
  }
  const inverseTimestepSquared = 1 / (timestep * timestep);
  const candidateTetrahedra = Array.from(
    { length: getTetrahedronCount(mesh) },
    (_unused, tetrahedron) => tetrahedron,
  );

  const rowCount = trainingPoses.length * 3;
  const rawColumns = candidateTetrahedra.map(
    () => new Float64Array(rowCount),
  );
  const target = new Float64Array(rowCount);

  for (let poseIndex = 0; poseIndex < trainingPoses.length; poseIndex += 1) {
    const poseTarget = new Float64Array(3);
    for (
      let candidate = 0;
      candidate < candidateTetrahedra.length;
      candidate += 1
    ) {
      const reduced = evaluateReducedTetGradient(
        mesh,
        restData,
        lumpedMasses,
        inverseTimestepSquared,
        incidentCounts,
        vertex,
        exactBasis,
        trainingPoses[poseIndex]!,
        candidateTetrahedra[candidate]!,
      );
      rawColumns[candidate]!.set(reduced, poseIndex * 3);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        poseTarget[coordinate] += reduced[coordinate]!;
      }
    }

    const poseNorm = Math.sqrt(squaredNorm(poseTarget));
    if (poseNorm > 1e-12) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const row = poseIndex * 3 + coordinate;
        target[row] = poseTarget[coordinate]! / poseNorm;
        for (
          let candidate = 0;
          candidate < rawColumns.length;
          candidate += 1
        ) {
          rawColumns[candidate]![row] /= poseNorm;
        }
      }
    }
  }

  const targetNorm = Math.sqrt(squaredNorm(target));
  if (
    targetNorm <= 1e-12 ||
    !rawColumns.some((column) => squaredNorm(column) > 1e-18)
  ) {
    return { samples: [], residual: 0 };
  }
  const selection = selectGreedyNonnegativeColumns({
    columns: rawColumns,
    target,
    maximumColumns: maximumSamples,
  });

  const samples: CubatureSample[] = [];
  for (
    let index = 0;
    index < selection.selectedColumnIndices.length;
    index += 1
  ) {
    const tetrahedron = candidateTetrahedra[
      selection.selectedColumnIndices[index]!
    ]!;
    const weight = selection.weights[index]!;
    if (weight > 1e-10) {
      samples.push({
        tetrahedron,
        weight,
        basisBlocks: copyTetBasisBlocks(
          mesh,
          exactBasis,
          tetrahedron,
        ),
      });
    }
  }

  return {
    samples,
    residual: selection.normalizedResidual,
  };
}
