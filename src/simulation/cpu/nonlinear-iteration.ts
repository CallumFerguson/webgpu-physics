import { minimumTetrahedralDeformationDeterminant } from "./stable-neo-hookean";
import type { RestTetraData, TetrahedralMesh } from "./types";
import { getVertexCount } from "./mesh";
import { PHASE1_ACCEPTED_DETERMINANT_FLOOR } from "./nonlinear-globalization";

export const JGS2_RUNTIME_MAXIMUM_CONVERGENCE_TOLERANCE = 1e-3;
export const JGS2_TINY_REFERENCE_RESIDUAL_TOLERANCE = 1e-5;

export interface JGS2JacobiFeasibilityResult {
  readonly accepted: boolean;
  readonly reverted: boolean;
  readonly positions: Float64Array;
  /** Finite zero sentinel when candidate positions or determinant arithmetic are invalid. */
  readonly candidateMinimumDeformationDeterminant: number;
  readonly candidateMinimumDeformationDeterminantValid: boolean;
  readonly acceptedMinimumDeformationDeterminant: number;
  /** Separately recorded assembled energies; zero sentinels when unavailable. */
  readonly sourceEnergy: number;
  readonly candidateEnergy: number;
  readonly acceptedEnergy: number;
  readonly energyValid: boolean;
  readonly revertCount: 0 | 1;
}

export interface JGS2JacobiEnergyPair {
  readonly source: number;
  readonly candidate: number;
}

/**
 * Enforce the assembled-pose invariant after parallel local Jacobi updates.
 * A failed candidate reverts the complete pose, never a subset of vertices.
 */
export function acceptOrRevertFeasibleJacobiIteration(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  sourcePositions: Float64Array,
  candidatePositions: Float64Array,
  energies?: JGS2JacobiEnergyPair,
): JGS2JacobiFeasibilityResult {
  const expectedLength = getVertexCount(mesh) * 3;
  if (
    sourcePositions.length !== expectedLength ||
    candidatePositions.length !== expectedLength
  ) {
    throw new RangeError("Jacobi feasibility positions do not match the mesh.");
  }
  const sourceMinimum = minimumTetrahedralDeformationDeterminant(
    mesh,
    restData,
    sourcePositions,
  );
  if (
    !Number.isFinite(sourceMinimum) ||
    !(sourceMinimum > PHASE1_ACCEPTED_DETERMINANT_FLOOR)
  ) {
    throw new RangeError(
      `Jacobi source pose must be feasible; minimum J is ${sourceMinimum}.`,
    );
  }
  if (
    energies !== undefined &&
    (!Number.isFinite(energies.source) || !Number.isFinite(energies.candidate))
  ) {
    throw new RangeError("Jacobi assembled energies must be finite.");
  }

  let candidateFinite = true;
  for (const value of candidatePositions) {
    candidateFinite &&= Number.isFinite(value);
  }
  const rawCandidateMinimum = candidateFinite
    ? minimumTetrahedralDeformationDeterminant(
        mesh,
        restData,
        candidatePositions,
      )
    : 0;
  const candidateMinimumValid =
    candidateFinite && Number.isFinite(rawCandidateMinimum);
  const candidateMinimum = candidateMinimumValid ? rawCandidateMinimum : 0;
  const accepted =
    candidateMinimumValid &&
    candidateMinimum > PHASE1_ACCEPTED_DETERMINANT_FLOOR;
  const sourceEnergy = energies?.source ?? 0;
  const candidateEnergy = energies?.candidate ?? 0;
  return {
    accepted,
    reverted: !accepted,
    positions: accepted ? candidatePositions.slice() : sourcePositions.slice(),
    candidateMinimumDeformationDeterminant: candidateMinimum,
    candidateMinimumDeformationDeterminantValid: candidateMinimumValid,
    acceptedMinimumDeformationDeterminant: accepted
      ? candidateMinimum
      : sourceMinimum,
    sourceEnergy,
    candidateEnergy,
    acceptedEnergy: accepted ? candidateEnergy : sourceEnergy,
    energyValid: energies !== undefined,
    revertCount: accepted ? 0 : 1,
  };
}

export interface JGS2ConvergenceGradientComponents {
  readonly inertia: ArrayLike<number>;
  readonly material: ArrayLike<number>;
  readonly externalForce: ArrayLike<number>;
  readonly target: ArrayLike<number>;
  readonly contact: ArrayLike<number>;
}

export interface JGS2ConvergenceOptions {
  readonly gradients: JGS2ConvergenceGradientComponents;
  /** Accepted assembled xyz updates, one triplet per active vertex. */
  readonly acceptedUpdates: ArrayLike<number>;
  readonly sceneScale: number;
  readonly residualTolerance: number;
  readonly normalizedUpdateTolerance: number;
  readonly feasible: boolean;
  /** True when assembled feasibility rejected and reverted this iteration. */
  readonly reverted?: boolean;
  readonly localFailureCount?: number;
}

export interface JGS2ConvergenceResult {
  readonly converged: boolean;
  readonly finite: boolean;
  readonly totalGradient: Float64Array;
  readonly gradientNorm: number;
  readonly componentGradientNorms: Readonly<{
    inertia: number;
    material: number;
    externalForce: number;
    target: number;
    contact: number;
  }>;
  readonly residualDenominator: number;
  readonly relativeResidual: number;
  readonly maximumUpdate: number;
  readonly normalizedMaximumUpdate: number;
  readonly residualSatisfied: boolean;
  readonly updateSatisfied: boolean;
  readonly feasible: boolean;
  readonly reverted: boolean;
  readonly localFailureCount: number;
}

interface FiniteDiagnostic {
  readonly value: number;
  readonly valid: boolean;
}

/** Overflow-safe Euclidean norm with a finite failure sentinel. */
function finiteEuclideanNorm(values: ArrayLike<number>): FiniteDiagnostic {
  let scale = 0;
  let scaledSquares = 1;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isFinite(value)) {
      return { value: Number.MAX_VALUE, valid: false };
    }
    const magnitude = Math.abs(value);
    if (magnitude === 0) continue;
    if (scale < magnitude) {
      const ratio = scale / magnitude;
      scaledSquares = 1 + scaledSquares * ratio * ratio;
      scale = magnitude;
    } else {
      const ratio = magnitude / scale;
      scaledSquares += ratio * ratio;
    }
  }
  if (scale === 0) return { value: 0, valid: true };
  const norm = scale * Math.sqrt(scaledSquares);
  return Number.isFinite(norm)
    ? { value: norm, valid: true }
    : { value: Number.MAX_VALUE, valid: false };
}

/** Sum a short component vector without an avoidable intermediate overflow. */
function finiteScaledSum(values: readonly number[]): FiniteDiagnostic {
  let scale = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) return { value: 0, valid: false };
    scale = Math.max(scale, Math.abs(value));
  }
  if (scale === 0) return { value: 0, valid: true };
  let normalized = 0;
  for (const value of values) normalized += value / scale;
  const sum = scale * normalized;
  return Number.isFinite(sum)
    ? { value: sum, valid: true }
    : { value: 0, valid: false };
}

function finitePositiveSum(values: readonly number[]): FiniteDiagnostic {
  let sum = 0;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || value > Number.MAX_VALUE - sum) {
      return { value: Number.MAX_VALUE, valid: false };
    }
    sum += value;
  }
  return { value: sum, valid: true };
}

function validateTolerance(value: number, label: string): void {
  if (
    !(value > 0) ||
    !Number.isFinite(value) ||
    value > JGS2_RUNTIME_MAXIMUM_CONVERGENCE_TOLERANCE
  ) {
    throw new RangeError(
      `${label} must be finite, positive, and no greater than ` +
        `${JGS2_RUNTIME_MAXIMUM_CONVERGENCE_TOLERANCE}.`,
    );
  }
}

/** Component-aware Phase 1 residual and scene-normalized update gate. */
export function evaluateJGS2Convergence(
  options: JGS2ConvergenceOptions,
): JGS2ConvergenceResult {
  validateTolerance(options.residualTolerance, "Residual tolerance");
  validateTolerance(
    options.normalizedUpdateTolerance,
    "Normalized update tolerance",
  );
  if (!(options.sceneScale >= 0) || !Number.isFinite(options.sceneScale)) {
    throw new RangeError("Scene scale must be finite and nonnegative.");
  }
  const localFailureCount = options.localFailureCount ?? 0;
  if (!Number.isSafeInteger(localFailureCount) || localFailureCount < 0) {
    throw new RangeError("Local failure count must be a nonnegative integer.");
  }

  const components = options.gradients;
  const dimension = components.inertia.length;
  if (dimension < 1 || dimension % 3 !== 0) {
    throw new RangeError(
      "Convergence gradients must contain one or more xyz triplets.",
    );
  }
  const entries = [
    components.inertia,
    components.material,
    components.externalForce,
    components.target,
    components.contact,
  ];
  if (entries.some((entry) => entry.length !== dimension)) {
    throw new RangeError("Convergence gradient components must have equal lengths.");
  }
  if (options.acceptedUpdates.length !== dimension) {
    throw new RangeError(
      "Accepted updates must match the convergence gradient dimension.",
    );
  }
  const totalGradient = new Float64Array(dimension);
  let finite = true;
  for (let coordinate = 0; coordinate < dimension; coordinate += 1) {
    const coordinateSum = finiteScaledSum(
      entries.map((entry) => entry[coordinate]!),
    );
    totalGradient[coordinate] = coordinateSum.value;
    finite &&= coordinateSum.valid;
  }
  for (let coordinate = 0; coordinate < options.acceptedUpdates.length; coordinate += 1) {
    finite &&= Number.isFinite(options.acceptedUpdates[coordinate]);
  }

  const normDiagnostics = entries.map(finiteEuclideanNorm);
  finite &&= normDiagnostics.every((diagnostic) => diagnostic.valid);
  const norms = normDiagnostics.map((diagnostic) => diagnostic.value);
  const totalNorm = finiteEuclideanNorm(totalGradient);
  finite &&= totalNorm.valid;
  const gradientNorm = totalNorm.value;
  const denominatorSum = finitePositiveSum(norms);
  finite &&= denominatorSum.valid;
  const residualDenominator = Math.max(1, denominatorSum.value);
  const computedRelativeResidual = gradientNorm / residualDenominator;
  finite &&= Number.isFinite(computedRelativeResidual);
  const relativeResidual = finite
    ? computedRelativeResidual
    : Number.MAX_VALUE;
  let maximumUpdate = 0;
  if (finite) {
    for (let vertex = 0; vertex < options.acceptedUpdates.length / 3; vertex += 1) {
      const updateNorm = finiteEuclideanNorm(
        Array.from(
          { length: 3 },
          (_unused, coordinate) =>
            options.acceptedUpdates[vertex * 3 + coordinate]!,
        ),
      );
      finite &&= updateNorm.valid;
      maximumUpdate = Math.max(maximumUpdate, updateNorm.value);
    }
  }
  const sceneScaleDenominator = Math.max(options.sceneScale, 1e-12);
  const computedNormalizedMaximumUpdate =
    maximumUpdate / sceneScaleDenominator;
  finite &&= Number.isFinite(computedNormalizedMaximumUpdate);
  const normalizedMaximumUpdate = finite
    ? computedNormalizedMaximumUpdate
    : Number.MAX_VALUE;
  const residualSatisfied =
    finite && relativeResidual <= options.residualTolerance;
  const updateSatisfied =
    finite && normalizedMaximumUpdate <= options.normalizedUpdateTolerance;
  const converged =
    residualSatisfied &&
    updateSatisfied &&
    options.feasible &&
    !options.reverted &&
    localFailureCount === 0;
  return {
    converged,
    finite,
    totalGradient,
    gradientNorm,
    componentGradientNorms: Object.freeze({
      inertia: norms[0]!,
      material: norms[1]!,
      externalForce: norms[2]!,
      target: norms[3]!,
      contact: norms[4]!,
    }),
    residualDenominator,
    relativeResidual,
    maximumUpdate,
    normalizedMaximumUpdate,
    residualSatisfied,
    updateSatisfied,
    feasible: options.feasible,
    reverted: options.reverted ?? false,
    localFailureCount,
  };
}
