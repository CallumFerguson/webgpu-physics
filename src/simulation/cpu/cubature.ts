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

interface CandidateColumn {
  readonly tetrahedron: number;
  readonly values: Float64Array;
}

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

/** Exhaustive active-set NNLS; selected sets never exceed six columns. */
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

  const candidates: CandidateColumn[] = candidateTetrahedra
    .map((tetrahedron, index) => ({
      tetrahedron,
      values: rawColumns[index]!,
    }))
    .filter((candidate) => squaredNorm(candidate.values) > 1e-18);
  const targetNorm = Math.sqrt(squaredNorm(target));
  if (targetNorm <= 1e-12 || candidates.length === 0) {
    return { samples: [], residual: 0 };
  }

  const selected: CandidateColumn[] = [];
  let weights: Float64Array = new Float64Array(0);
  let residual: Float64Array = target.slice();

  for (
    let selection = 0;
    selection < Math.min(maximumSamples, candidates.length);
    selection += 1
  ) {
    let best: CandidateColumn | undefined;
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
        (Math.abs(trialError - bestError) <= 1e-14 * Math.max(1, bestError) &&
          best !== undefined &&
          candidate.tetrahedron < best.tetrahedron)
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
    if (Math.sqrt(squaredNorm(residual)) / targetNorm < 0.01) {
      break;
    }
  }

  for (let refinement = 0; refinement < 4 && selected.length > 0; refinement += 1) {
    let replacementSlot = -1;
    let replacementCandidate: CandidateColumn | undefined;
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
            candidate.tetrahedron < replacementCandidate.tetrahedron)
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

  const samples: CubatureSample[] = [];
  for (let index = 0; index < selected.length; index += 1) {
    const weight = weights[index]!;
    if (weight > 1e-10) {
      samples.push({
        tetrahedron: selected[index]!.tetrahedron,
        weight,
        basisBlocks: copyTetBasisBlocks(
          mesh,
          exactBasis,
          selected[index]!.tetrahedron,
        ),
      });
    }
  }

  return {
    samples,
    residual: Math.sqrt(squaredNorm(residual)) / targetNorm,
  };
}
