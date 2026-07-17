import { determinant3, dot } from "./math";
import type {
  ExactStableNeoHookeanJGS2LocalEvaluation,
  ExactStableNeoHookeanJGS2LocalModel,
} from "./jgs2-local";

/** Smith elasticity remains finite through inversion, but accepted poses do not. */
export const PHASE1_ACCEPTED_DETERMINANT_FLOOR = 1e-4;

/** 256 f32 ulps, as frozen by the Phase 1 positive-definite-treatment record. */
export const JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR = 256 * 2 ** -24;
export const JGS2_LOCAL_MAX_NORMALIZED_SHIFT = 1e-3;
export const JGS2_LOCAL_ARMIJO_C1 = 1e-4;
export const JGS2_LOCAL_BACKTRACK_FACTOR = 0.5;
export const JGS2_LOCAL_MAX_BACKTRACKS = 12;

export interface JGS2LocalPositiveDefiniteOptions {
  /** The source vertex's m / h^2 contribution. */
  readonly inertiaScale: number;
}

export type JGS2LocalDirectionStatus =
  | "accepted"
  | "shift-limit-exceeded"
  | "non-descent-direction";

export interface JGS2LocalDirectionResult {
  readonly status: JGS2LocalDirectionStatus;
  readonly accepted: boolean;
  /** Symmetric row-major Hessian used to calculate the spectrum. */
  readonly symmetricHessian: Float64Array;
  readonly shiftedHessian: Float64Array;
  readonly direction: Float64Array;
  readonly minimumEigenvalue: number;
  readonly scale: number;
  readonly eigenvalueFloor: number;
  readonly diagonalShift: number;
  readonly normalizedShift: number;
  readonly maximumRelativeAsymmetry: number;
  readonly gradientDotDirection: number;
  readonly linearResidual: number;
}

export interface JGS2RestrictedLocalLineSearchOptions {
  readonly initialEnergy: number;
  readonly gradient: ArrayLike<number>;
  readonly direction: ArrayLike<number>;
  /** Geometry-only feasibility check; called before energy for every trial. */
  readonly minimumDeformationDeterminant: (alpha: number) => number;
  /** Called only after the determinant is finite and above the floor. */
  readonly energy: (alpha: number) => number;
}

export type JGS2RestrictedLocalLineSearchStatus =
  | "accepted"
  | "zero-direction"
  | "non-descent-direction"
  | "no-acceptable-step";

export interface JGS2RestrictedLocalLineSearchResult {
  readonly status: JGS2RestrictedLocalLineSearchStatus;
  readonly accepted: boolean;
  /** Zero when no trial is accepted. */
  readonly alpha: number;
  readonly step: Float64Array;
  readonly initialEnergy: number;
  /** Equal to initialEnergy when no trial is accepted. */
  readonly acceptedEnergy: number;
  readonly armijoBound: number;
  readonly gradientDotDirection: number;
  /** Finite zero sentinel when no trial produced a finite determinant. */
  readonly minimumDeformationDeterminant: number;
  readonly minimumDeformationDeterminantValid: boolean;
  readonly backtrackCount: number;
  readonly evaluatedTrialCount: number;
  readonly energyEvaluationCount: number;
  readonly infeasibleTrialCount: number;
  readonly nonFiniteTrialCount: number;
}

export interface ExactJGS2LocalGlobalizationOptions
  extends JGS2LocalPositiveDefiniteOptions {
  readonly model: ExactStableNeoHookeanJGS2LocalModel;
  /** Geometry-only determinant oracle for x + Bq. */
  readonly minimumDeformationDeterminant: (
    localDisplacement: Float64Array,
  ) => number;
}

export interface ExactJGS2LocalGlobalizationResult {
  readonly initial: ExactStableNeoHookeanJGS2LocalEvaluation;
  readonly direction: JGS2LocalDirectionResult;
  /** Undefined when the positive-definite treatment rejects the local system. */
  readonly lineSearch: JGS2RestrictedLocalLineSearchResult | undefined;
}

function requireFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite; got ${value}.`);
  }
  return value;
}

function requireFiniteVector(
  values: ArrayLike<number>,
  expectedLength: number,
  label: string,
): Float64Array {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `${label} must contain ${expectedLength} values; got ${values.length}.`,
    );
  }
  const result = Float64Array.from(values);
  for (const value of result) {
    requireFinite(value, label);
  }
  return result;
}

function symmetrize3(matrix: ArrayLike<number>): Float64Array {
  const source = requireFiniteVector(matrix, 9, "Local Hessian");
  const a01 = 0.5 * (source[1]! + source[3]!);
  const a02 = 0.5 * (source[2]! + source[6]!);
  const a12 = 0.5 * (source[5]! + source[7]!);
  return new Float64Array([
    source[0]!, a01, a02,
    a01, source[4]!, a12,
    a02, a12, source[8]!,
  ]);
}

/** Deterministic closed-form minimum eigenvalue of a symmetric 3x3 matrix. */
export function minimumSymmetricEigenvalue3(matrix: ArrayLike<number>): number {
  const symmetric = symmetrize3(matrix);
  const a00 = symmetric[0]!;
  const a11 = symmetric[4]!;
  const a22 = symmetric[8]!;
  const offDiagonalSquared =
    symmetric[1]! ** 2 + symmetric[2]! ** 2 + symmetric[5]! ** 2;
  if (offDiagonalSquared === 0) {
    return Math.min(a00, a11, a22);
  }

  const mean = (a00 + a11 + a22) / 3;
  const centeredSquared =
    (a00 - mean) ** 2 +
    (a11 - mean) ** 2 +
    (a22 - mean) ** 2 +
    2 * offDiagonalSquared;
  const scale = Math.sqrt(centeredSquared / 6);
  if (!(scale > 0)) {
    return mean;
  }

  const normalized = symmetric.slice();
  normalized[0] = (normalized[0]! - mean) / scale;
  normalized[4] = (normalized[4]! - mean) / scale;
  normalized[8] = (normalized[8]! - mean) / scale;
  for (const index of [1, 2, 3, 5, 6, 7]) {
    normalized[index] /= scale;
  }
  const halfDeterminant = Math.max(
    -1,
    Math.min(1, determinant3(normalized) / 2),
  );
  const angle = Math.acos(halfDeterminant) / 3;
  const largest = mean + 2 * scale * Math.cos(angle);
  const smallest = mean + 2 * scale * Math.cos(angle + (2 * Math.PI) / 3);
  const middle = 3 * mean - largest - smallest;
  return Math.min(largest, middle, smallest);
}

function frobeniusNorm3(matrix: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < 9; index += 1) {
    sum += matrix[index]! * matrix[index]!;
  }
  return Math.sqrt(sum);
}

function maximumRelativeAsymmetry3(matrix: ArrayLike<number>): number {
  const values = requireFiniteVector(matrix, 9, "Local Hessian");
  let maximumMagnitude = 0;
  for (const value of values) maximumMagnitude = Math.max(maximumMagnitude, Math.abs(value));
  const maximumDifference = Math.max(
    Math.abs(values[1]! - values[3]!),
    Math.abs(values[2]! - values[6]!),
    Math.abs(values[5]! - values[7]!),
  );
  return maximumMagnitude > 0 ? maximumDifference / maximumMagnitude : 0;
}

function relativeLinearResidual(
  matrix: ArrayLike<number>,
  solution: ArrayLike<number>,
  rightHandSide: ArrayLike<number>,
): number {
  let residualSquared = 0;
  let rightHandSideSquared = 0;
  for (let row = 0; row < 3; row += 1) {
    let product = 0;
    for (let column = 0; column < 3; column += 1) {
      product += matrix[row * 3 + column]! * solution[column]!;
    }
    residualSquared += (product - rightHandSide[row]!) ** 2;
    rightHandSideSquared += rightHandSide[row]! ** 2;
  }
  return Math.sqrt(residualSquared) / Math.max(1, Math.sqrt(rightHandSideSquared));
}

function solvePositiveDefiniteSymmetric3(
  matrix: Float64Array,
  rightHandSide: Float64Array,
  scale: number,
): Float64Array {
  const tolerance = scale * Number.EPSILON * 16;
  const l00Squared = matrix[0]!;
  if (!(l00Squared > tolerance)) {
    throw new Error("Shifted local Hessian lost positive definiteness at pivot 0.");
  }
  const l00 = Math.sqrt(l00Squared);
  const l10 = matrix[3]! / l00;
  const l20 = matrix[6]! / l00;
  const l11Squared = matrix[4]! - l10 * l10;
  if (!(l11Squared > tolerance)) {
    throw new Error("Shifted local Hessian lost positive definiteness at pivot 1.");
  }
  const l11 = Math.sqrt(l11Squared);
  const l21 = (matrix[7]! - l20 * l10) / l11;
  const l22Squared = matrix[8]! - l20 * l20 - l21 * l21;
  if (!(l22Squared > tolerance)) {
    throw new Error("Shifted local Hessian lost positive definiteness at pivot 2.");
  }
  const l22 = Math.sqrt(l22Squared);

  const y0 = rightHandSide[0]! / l00;
  const y1 = (rightHandSide[1]! - l10 * y0) / l11;
  const y2 = (rightHandSide[2]! - l20 * y0 - l21 * y1) / l22;
  const x2 = y2 / l22;
  const x1 = (y1 - l21 * x2) / l11;
  const x0 = (y0 - l10 * x1 - l20 * x2) / l00;
  return new Float64Array([x0, x1, x2]);
}

/**
 * Apply the Phase 1 solve-only uniform shift and calculate H_shift p = -g.
 * The exact material/oracle Hessian remains untouched.
 */
export function computeJGS2LocalDescentDirection(
  hessian: ArrayLike<number>,
  gradientValues: ArrayLike<number>,
  options: JGS2LocalPositiveDefiniteOptions,
): JGS2LocalDirectionResult {
  const symmetricHessian = symmetrize3(hessian);
  const gradient = requireFiniteVector(gradientValues, 3, "Local gradient");
  const inertiaScale = requireFinite(options.inertiaScale, "Inertia scale");
  if (!(inertiaScale > 0)) {
    throw new RangeError("Inertia scale must be positive.");
  }

  const minimumEigenvalue = minimumSymmetricEigenvalue3(symmetricHessian);
  const maximumRelativeAsymmetry = maximumRelativeAsymmetry3(hessian);
  const largestAbsoluteDiagonal = Math.max(
    Math.abs(symmetricHessian[0]!),
    Math.abs(symmetricHessian[4]!),
    Math.abs(symmetricHessian[8]!),
  );
  const scale = Math.max(
    largestAbsoluteDiagonal,
    frobeniusNorm3(symmetricHessian) / Math.sqrt(3),
    inertiaScale,
  );
  const eigenvalueFloor = JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR * scale;
  const diagonalShift = Math.max(0, eigenvalueFloor - minimumEigenvalue);
  const normalizedShift = diagonalShift / scale;
  const shiftedHessian = symmetricHessian.slice();
  shiftedHessian[0] += diagonalShift;
  shiftedHessian[4] += diagonalShift;
  shiftedHessian[8] += diagonalShift;

  const rejectedByShift =
    normalizedShift > JGS2_LOCAL_MAX_NORMALIZED_SHIFT;
  let direction: Float64Array = new Float64Array(3);
  let gradientDotDirection = 0;
  let linearResidual = 0;
  if (!rejectedByShift) {
    direction = solvePositiveDefiniteSymmetric3(
      shiftedHessian,
      Float64Array.from(gradient, (value) => -value),
      scale,
    );
    gradientDotDirection = dot(gradient, direction);
    linearResidual = relativeLinearResidual(
      shiftedHessian,
      direction,
      Float64Array.from(gradient, (value) => -value),
    );
  }

  const gradientSquaredNorm = dot(gradient, gradient);
  const descent = gradientSquaredNorm === 0 || gradientDotDirection < 0;
  const status: JGS2LocalDirectionStatus = rejectedByShift
    ? "shift-limit-exceeded"
    : descent
      ? "accepted"
      : "non-descent-direction";
  if (status !== "accepted") {
    direction = new Float64Array(3);
  }
  return {
    status,
    accepted: status === "accepted",
    symmetricHessian,
    shiftedHessian,
    direction,
    minimumEigenvalue,
    scale,
    eigenvalueFloor,
    diagonalShift,
    normalizedShift,
    maximumRelativeAsymmetry,
    gradientDotDirection,
    linearResidual,
  };
}

/** Backtrack a restricted JGS2 local direction with Armijo and J feasibility. */
export function lineSearchRestrictedJGS2LocalDirection(
  options: JGS2RestrictedLocalLineSearchOptions,
): JGS2RestrictedLocalLineSearchResult {
  const initialEnergy = requireFinite(options.initialEnergy, "Initial energy");
  const gradient = requireFiniteVector(options.gradient, 3, "Local gradient");
  const direction = requireFiniteVector(options.direction, 3, "Local direction");
  const gradientDotDirection = dot(gradient, direction);
  const directionSquaredNorm = dot(direction, direction);
  const zeroStep = new Float64Array(3);
  const rejectedResult = (
    status: Exclude<JGS2RestrictedLocalLineSearchStatus, "accepted">,
    backtrackCount: number,
    evaluatedTrialCount: number,
    infeasibleTrialCount: number,
    nonFiniteTrialCount: number,
    minimumDeformationDeterminant = 0,
    minimumDeformationDeterminantValid = false,
    energyEvaluationCount = 0,
  ): JGS2RestrictedLocalLineSearchResult => ({
    status,
    accepted: false,
    alpha: 0,
    step: zeroStep,
    initialEnergy,
    acceptedEnergy: initialEnergy,
    armijoBound: initialEnergy,
    gradientDotDirection,
    minimumDeformationDeterminant,
    minimumDeformationDeterminantValid,
    backtrackCount,
    evaluatedTrialCount,
    energyEvaluationCount,
    infeasibleTrialCount,
    nonFiniteTrialCount,
  });

  if (directionSquaredNorm === 0) {
    return rejectedResult("zero-direction", 0, 0, 0, 0);
  }
  if (!(gradientDotDirection < 0)) {
    return rejectedResult("non-descent-direction", 0, 0, 0, 0);
  }

  let alpha = 1;
  let infeasibleTrialCount = 0;
  let nonFiniteTrialCount = 0;
  let lastMinimumDeterminant = 0;
  let minimumDeformationDeterminantValid = false;
  let energyEvaluationCount = 0;
  for (
    let backtrack = 0;
    backtrack <= JGS2_LOCAL_MAX_BACKTRACKS;
    backtrack += 1
  ) {
    const minimumDeformationDeterminant =
      options.minimumDeformationDeterminant(alpha);
    if (Number.isFinite(minimumDeformationDeterminant)) {
      lastMinimumDeterminant = minimumDeformationDeterminant;
      minimumDeformationDeterminantValid = true;
    }
    if (!Number.isFinite(minimumDeformationDeterminant)) {
      nonFiniteTrialCount += 1;
    } else if (
      !(minimumDeformationDeterminant > PHASE1_ACCEPTED_DETERMINANT_FLOOR)
    ) {
      infeasibleTrialCount += 1;
    } else {
      const trialEnergy = options.energy(alpha);
      energyEvaluationCount += 1;
      if (!Number.isFinite(trialEnergy)) {
        nonFiniteTrialCount += 1;
        alpha *= JGS2_LOCAL_BACKTRACK_FACTOR;
        continue;
      }
      const armijoBound =
        initialEnergy + JGS2_LOCAL_ARMIJO_C1 * alpha * gradientDotDirection;
      if (trialEnergy <= armijoBound) {
        return {
          status: "accepted",
          accepted: true,
          alpha,
          step: Float64Array.from(direction, (value) => alpha * value),
          initialEnergy,
          acceptedEnergy: trialEnergy,
          armijoBound,
          gradientDotDirection,
          minimumDeformationDeterminant,
          minimumDeformationDeterminantValid: true,
          backtrackCount: backtrack,
          evaluatedTrialCount: backtrack + 1,
          energyEvaluationCount,
          infeasibleTrialCount,
          nonFiniteTrialCount,
        };
      }
    }
    alpha *= JGS2_LOCAL_BACKTRACK_FACTOR;
  }
  return rejectedResult(
    "no-acceptable-step",
    JGS2_LOCAL_MAX_BACKTRACKS,
    JGS2_LOCAL_MAX_BACKTRACKS + 1,
    infeasibleTrialCount,
    nonFiniteTrialCount,
    lastMinimumDeterminant,
    minimumDeformationDeterminantValid,
    energyEvaluationCount,
  );
}

/**
 * Tiny-system oracle that applies the frozen globalization policy to the exact
 * all-element local objective E(x + Bq). Production uses the same policy with
 * its source-exact, complementary-Cubature restricted energy.
 */
export function globalizeExactStableNeoHookeanJGS2Local(
  options: ExactJGS2LocalGlobalizationOptions,
): ExactJGS2LocalGlobalizationResult {
  const zero = new Float64Array(3);
  const sourceMinimumDeterminant = options.minimumDeformationDeterminant(zero);
  if (
    !Number.isFinite(sourceMinimumDeterminant) ||
    !(sourceMinimumDeterminant > PHASE1_ACCEPTED_DETERMINANT_FLOOR)
  ) {
    throw new RangeError(
      `Exact JGS2 local source pose must have minimum deformation ` +
        `determinant greater than ${PHASE1_ACCEPTED_DETERMINANT_FLOOR}; got ` +
        `${sourceMinimumDeterminant}.`,
    );
  }
  const initial = options.model.evaluate(zero);
  const direction = computeJGS2LocalDescentDirection(
    initial.hessian,
    initial.gradient,
    options,
  );
  if (!direction.accepted) {
    return { initial, direction, lineSearch: undefined };
  }
  const lineSearch = lineSearchRestrictedJGS2LocalDirection({
    initialEnergy: initial.energy,
    gradient: initial.gradient,
    direction: direction.direction,
    minimumDeformationDeterminant: (alpha) => {
      const localDisplacement = Float64Array.from(
        direction.direction,
        (value) => alpha * value,
      );
      return options.minimumDeformationDeterminant(localDisplacement);
    },
    energy: (alpha) =>
      options.model.energy(
        Float64Array.from(direction.direction, (value) => alpha * value),
      ),
  });
  return { initial, direction, lineSearch };
}
