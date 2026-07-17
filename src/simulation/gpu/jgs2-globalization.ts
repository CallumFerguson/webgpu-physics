/** Maximum number of nonlinear-iteration records retained for explicit readback. */
export const JGS2_GLOBALIZATION_HISTORY_CAPACITY = 64;

/** Per-active-vertex globalization record width, measured in vec4 values. */
export const JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S = 4;
/** Per-tetrahedron assembled-feasibility record width, measured in vec4 values. */
export const JGS2_GLOBALIZATION_TET_DIAGNOSTIC_VEC4S = 2;
/** Per-vertex convergence-component record width, measured in vec4 values. */
export const JGS2_GLOBALIZATION_CONVERGENCE_COMPONENT_VEC4S = 5;
/** Concise ABI alias used by the dynamic-layout packer. */
export const JGS2_CONVERGENCE_COMPONENT_VEC4S =
  JGS2_GLOBALIZATION_CONVERGENCE_COMPONENT_VEC4S;
/** Global assembled-acceptance control width, measured in vec4 values. */
export const JGS2_GLOBALIZATION_CONTROL_VEC4S = 4;
/** Per-iteration convergence-history record width, measured in vec4 values. */
export const JGS2_GLOBALIZATION_HISTORY_VEC4S = 8;

export const JGS2_GLOBALIZATION_RELATIVE_EIGENVALUE_FLOOR = 2 ** -16;
export const JGS2_GLOBALIZATION_MAX_NORMALIZED_SHIFT = 1e-3;
export const JGS2_GLOBALIZATION_DETERMINANT_FLOOR = 1e-4;
export const JGS2_GLOBALIZATION_ARMIJO_C1 = 1e-4;
export const JGS2_GLOBALIZATION_BACKTRACK_FACTOR = 0.5;
export const JGS2_GLOBALIZATION_MAX_BACKTRACKS = 12;
export const JGS2_GLOBALIZATION_MAX_RUNTIME_TOLERANCE = 1e-3;
/** Sub-eight-ulp world-space directions are numerically zero in f32 storage. */
export const JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER = 8;

/** Bitfield stored exactly in control/history lane 0.w. */
export const JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT = 1 << 0;
export const JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT = 1 << 1;
export const JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT = 1 << 2;
export const JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT = 1 << 3;
export const JGS2_GLOBALIZATION_VALIDITY_BITS_MASK =
  JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT |
  JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT |
  JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT |
  JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT;

export const JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED = 0 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_PINNED = 1 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT = 2 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_SHIFT_LIMIT = 3 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_NON_DESCENT = 4 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_LINE_SEARCH_FAILED = 5 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE = 6 as const;
export const JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED = 7 as const;

export const JGS2_GLOBALIZATION_LOCAL_STATUS_CODES = Object.freeze({
  accepted: JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
  pinned: JGS2_GLOBALIZATION_LOCAL_STATUS_PINNED,
  zeroGradient: JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT,
  shiftLimit: JGS2_GLOBALIZATION_LOCAL_STATUS_SHIFT_LIMIT,
  nonDescent: JGS2_GLOBALIZATION_LOCAL_STATUS_NON_DESCENT,
  lineSearchFailed: JGS2_GLOBALIZATION_LOCAL_STATUS_LINE_SEARCH_FAILED,
  nonfinite: JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
  choleskyFailed: JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED,
} as const);

export type JGS2GlobalizationLocalStatusCode =
  (typeof JGS2_GLOBALIZATION_LOCAL_STATUS_CODES)[keyof typeof JGS2_GLOBALIZATION_LOCAL_STATUS_CODES];

export type JGS2GlobalizationLocalStatus =
  | "accepted"
  | "pinned"
  | "zero-gradient"
  | "shift-limit"
  | "non-descent"
  | "line-search-failed"
  | "nonfinite"
  | "cholesky-failed";

const LOCAL_STATUS_NAMES = Object.freeze([
  "accepted",
  "pinned",
  "zero-gradient",
  "shift-limit",
  "non-descent",
  "line-search-failed",
  "nonfinite",
  "cholesky-failed",
] as const satisfies readonly JGS2GlobalizationLocalStatus[]);

export interface JGS2GlobalizationLocalDiagnostic {
  readonly direction: readonly [number, number, number];
  readonly alpha: number;
  readonly minimumEigenvalue: number;
  readonly scale: number;
  readonly diagonalShift: number;
  readonly normalizedShift: number;
  readonly acceptedEnergyDelta: number;
  readonly armijoDeltaBound: number;
  readonly gradientDotDirection: number;
  readonly shiftedLinearResidual: number;
  readonly minimumTrialDeformationDeterminant: number;
  readonly backtrackCount: number;
  readonly energyEvaluationCount: number;
  readonly statusCode: JGS2GlobalizationLocalStatusCode;
  readonly status: JGS2GlobalizationLocalStatus;
  readonly accepted: boolean;
  /** False when any direction lane was replaced by the finite zero sentinel. */
  readonly directionValid: boolean;
  /** False when alpha was replaced by the finite zero sentinel. */
  readonly alphaValid: boolean;
  /** Covers lambdaMin, scale, shift, and normalized shift. */
  readonly positiveDefiniteDiagnosticsValid: boolean;
  /** Covers the two energy deltas, g dot p, and shifted-system residual. */
  readonly energyDiagnosticsValid: boolean;
  /** False when minTrialJ was replaced by the finite zero sentinel. */
  readonly minimumTrialDeformationDeterminantValid: boolean;
  /** Numeric-record validity; a deliberate `nonfinite` status also clears it. */
  readonly finite: boolean;
}

export interface JGS2GlobalizationHistoryRecord {
  readonly sourceMinimumDeformationDeterminant: number;
  readonly candidateMinimumDeformationDeterminant: number;
  readonly acceptedMinimumDeformationDeterminant: number;
  readonly sourceGeometryValid: boolean;
  readonly candidateGeometryValid: boolean;
  readonly acceptedMinimumDeformationDeterminantValid: boolean;
  readonly localNumericsValid: boolean;
  /** Source and candidate determinant arithmetic both completed finitely. */
  readonly geometryValid: boolean;
  readonly assembledAccepted: boolean;
  readonly assembledReverted: boolean;
  readonly localFailureCount: number;
  readonly revertCount: number;
  readonly sourceEnergy: number;
  readonly candidateEnergy: number;
  readonly acceptedEnergy: number;
  readonly energyValid: boolean;
  readonly componentGradientNorms: Readonly<{
    readonly inertia: number;
    readonly material: number;
    readonly externalForce: number;
    readonly target: number;
    readonly contact: number;
  }>;
  readonly componentGradientNormsValid: boolean;
  readonly gradientNorm: number;
  readonly residualDenominator: number;
  readonly relativeResidual: number;
  readonly residualDiagnosticsValid: boolean;
  readonly maximumUpdate: number;
  readonly normalizedMaximumUpdate: number;
  readonly updateDiagnosticsValid: boolean;
  readonly residualSatisfied: boolean;
  readonly updateSatisfied: boolean;
  readonly converged: boolean;
  /** The packed finite flag combined with decoder-side numeric validation. */
  readonly finite: boolean;
  readonly iterationIndex: number;
  readonly maximumLocalNormalizedShift: number;
  readonly maximumLocalNormalizedShiftValid: boolean;
  readonly maximumBacktrackCount: number;
  readonly infeasibleTrialCount: number;
  readonly energyEvaluationCount: number;
  /** True only for a history slot whose nonlinear iteration executed. */
  readonly active: boolean;
}

interface FiniteScalar {
  readonly value: number;
  readonly valid: boolean;
}

const FLOATS_PER_VEC4 = 4;
const LOCAL_DIAGNOSTIC_FLOATS =
  JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S * FLOATS_PER_VEC4;
const HISTORY_RECORD_FLOATS =
  JGS2_GLOBALIZATION_HISTORY_VEC4S * FLOATS_PER_VEC4;

function assertExactLength(
  packed: ArrayLike<number>,
  expected: number,
  label: string,
): void {
  if (packed.length !== expected) {
    throw new RangeError(
      `${label} contains ${packed.length} floats; expected exactly ${expected}.`,
    );
  }
}

function finiteOrZero(value: number): FiniteScalar {
  return Number.isFinite(value)
    ? { value, valid: true }
    : { value: 0, valid: false };
}

function finiteNonnegativeOrZero(value: number): FiniteScalar {
  return Number.isFinite(value) && value >= 0
    ? { value, valid: true }
    : { value: 0, valid: false };
}

function decodeFlag(value: number, label: string): boolean {
  if (value !== 0 && value !== 1) {
    throw new RangeError(`${label} must be encoded as exactly 0 or 1.`);
  }
  return value === 1;
}

function decodeCount(
  value: number,
  label: string,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value > maximum
  ) {
    throw new RangeError(
      `${label} must be a nonnegative integer no greater than ${maximum}.`,
    );
  }
  return value;
}

function decodeStatusCode(value: number): JGS2GlobalizationLocalStatusCode {
  if (
    !Number.isSafeInteger(value) ||
    value < JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED ||
    value > JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED
  ) {
    throw new RangeError(
      `Local globalization status must be an integer from ` +
        `${JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED} through ` +
        `${JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED}.`,
    );
  }
  return value as JGS2GlobalizationLocalStatusCode;
}

/** Decode one 4-vec4 local-globalization record. */
export function decodeJGS2GlobalizationLocalDiagnostic(
  packed: ArrayLike<number>,
): JGS2GlobalizationLocalDiagnostic {
  assertExactLength(
    packed,
    LOCAL_DIAGNOSTIC_FLOATS,
    "Local globalization diagnostic",
  );

  const directionScalars = [
    finiteOrZero(packed[0]!),
    finiteOrZero(packed[1]!),
    finiteOrZero(packed[2]!),
  ] as const;
  const alpha = finiteNonnegativeOrZero(packed[3]!);
  const minimumEigenvalue = finiteOrZero(packed[4]!);
  const scale = finiteNonnegativeOrZero(packed[5]!);
  const diagonalShift = finiteNonnegativeOrZero(packed[6]!);
  const normalizedShift = finiteNonnegativeOrZero(packed[7]!);
  const acceptedEnergyDelta = finiteOrZero(packed[8]!);
  const armijoDeltaBound = finiteOrZero(packed[9]!);
  const gradientDotDirection = finiteOrZero(packed[10]!);
  const shiftedLinearResidual = finiteNonnegativeOrZero(packed[11]!);
  const minimumTrialDeformationDeterminant = finiteOrZero(packed[12]!);
  const backtrackCount = decodeCount(
    packed[13]!,
    "Local globalization backtrackCount",
    JGS2_GLOBALIZATION_MAX_BACKTRACKS,
  );
  const energyEvaluationCount = decodeCount(
    packed[14]!,
    "Local globalization energyEvaluationCount",
  );
  const statusCode = decodeStatusCode(packed[15]!);
  const status = LOCAL_STATUS_NAMES[statusCode];
  // The fixed-width GPU ABI uses a finite zero sentinel when geometry was not
  // evaluated or determinant arithmetic failed. Status makes that sentinel
  // unambiguous without consuming another lane. Accepted/ordinary failed
  // searches completed finite geometry, and a nonfinite-energy outcome keeps
  // its already-feasible determinant while still clearing whole-record finite.
  const finiteGeometryBeforeNonfiniteEnergy =
    statusCode === JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE &&
    minimumTrialDeformationDeterminant.value >
      JGS2_GLOBALIZATION_DETERMINANT_FLOOR;
  const trialGeometryEvaluated =
    statusCode === JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED ||
    statusCode === JGS2_GLOBALIZATION_LOCAL_STATUS_LINE_SEARCH_FAILED ||
    finiteGeometryBeforeNonfiniteEnergy;
  const minimumTrialDeformationDeterminantValid =
    minimumTrialDeformationDeterminant.valid && trialGeometryEvaluated;

  const directionValid = directionScalars.every((entry) => entry.valid);
  const positiveDefiniteDiagnosticsValid = [
    minimumEigenvalue,
    scale,
    diagonalShift,
    normalizedShift,
  ].every((entry) => entry.valid);
  const energyDiagnosticsValid = [
    acceptedEnergyDelta,
    armijoDeltaBound,
    gradientDotDirection,
    shiftedLinearResidual,
  ].every((entry) => entry.valid);
  const finite =
    directionValid &&
    alpha.valid &&
    positiveDefiniteDiagnosticsValid &&
    energyDiagnosticsValid &&
    minimumTrialDeformationDeterminant.valid &&
    statusCode !== JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE;

  return {
    direction: [
      directionScalars[0].value,
      directionScalars[1].value,
      directionScalars[2].value,
    ],
    alpha: alpha.value,
    minimumEigenvalue: minimumEigenvalue.value,
    scale: scale.value,
    diagonalShift: diagonalShift.value,
    normalizedShift: normalizedShift.value,
    acceptedEnergyDelta: acceptedEnergyDelta.value,
    armijoDeltaBound: armijoDeltaBound.value,
    gradientDotDirection: gradientDotDirection.value,
    shiftedLinearResidual: shiftedLinearResidual.value,
    minimumTrialDeformationDeterminant:
      minimumTrialDeformationDeterminant.value,
    backtrackCount,
    energyEvaluationCount,
    statusCode,
    status,
    accepted: statusCode === JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
    directionValid,
    alphaValid: alpha.valid,
    positiveDefiniteDiagnosticsValid,
    energyDiagnosticsValid,
    minimumTrialDeformationDeterminantValid:
      minimumTrialDeformationDeterminantValid,
    finite,
  };
}

/** Decode one 8-vec4 nonlinear convergence-history record. */
export function decodeJGS2GlobalizationHistoryRecord(
  packed: ArrayLike<number>,
): JGS2GlobalizationHistoryRecord {
  assertExactLength(
    packed,
    HISTORY_RECORD_FLOATS,
    "Globalization history record",
  );

  const sourceMinimum = finiteOrZero(packed[0]!);
  const candidateMinimum = finiteOrZero(packed[1]!);
  const acceptedMinimum = finiteOrZero(packed[2]!);
  const packedValidityBits = decodeCount(
    packed[3]!,
    "geometryValidityBits",
    JGS2_GLOBALIZATION_VALIDITY_BITS_MASK,
  );

  const assembledAccepted = decodeFlag(packed[4]!, "assembledAccepted");
  const assembledReverted = decodeFlag(packed[5]!, "assembledReverted");
  const localFailureCount = decodeCount(packed[6]!, "localFailureCount");
  const revertCount = decodeCount(packed[7]!, "revertCount", 1);

  const sourceEnergy = finiteOrZero(packed[8]!);
  const candidateEnergy = finiteOrZero(packed[9]!);
  const acceptedEnergy = finiteOrZero(packed[10]!);
  const packedEnergyValid = decodeFlag(packed[11]!, "energyValid");

  const inertiaNorm = finiteNonnegativeOrZero(packed[12]!);
  const materialNorm = finiteNonnegativeOrZero(packed[13]!);
  const externalForceNorm = finiteNonnegativeOrZero(packed[14]!);
  const targetNorm = finiteNonnegativeOrZero(packed[15]!);
  const contactNorm = finiteNonnegativeOrZero(packed[16]!);
  const gradientNorm = finiteNonnegativeOrZero(packed[17]!);
  const residualDenominator = finiteNonnegativeOrZero(packed[18]!);
  const relativeResidual = finiteNonnegativeOrZero(packed[19]!);

  const maximumUpdate = finiteNonnegativeOrZero(packed[20]!);
  const normalizedMaximumUpdate = finiteNonnegativeOrZero(packed[21]!);
  const residualSatisfied = decodeFlag(packed[22]!, "residualSatisfied");
  const updateSatisfied = decodeFlag(packed[23]!, "updateSatisfied");

  const converged = decodeFlag(packed[24]!, "converged");
  const packedFinite = decodeFlag(packed[25]!, "finite");
  const iterationIndex = decodeCount(
    packed[26]!,
    "iterationIndex",
    JGS2_GLOBALIZATION_HISTORY_CAPACITY - 1,
  );
  const maximumLocalNormalizedShift = finiteNonnegativeOrZero(packed[27]!);

  const maximumBacktrackCount = decodeCount(
    packed[28]!,
    "maximumBacktrackCount",
    JGS2_GLOBALIZATION_MAX_BACKTRACKS,
  );
  const infeasibleTrialCount = decodeCount(
    packed[29]!,
    "infeasibleTrialCount",
  );
  const energyEvaluationCount = decodeCount(
    packed[30]!,
    "energyEvaluationCount",
  );
  const active = decodeFlag(packed[31]!, "active");

  const sourceGeometryValid =
    (packedValidityBits &
      JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT) !==
      0 && sourceMinimum.valid;
  const candidateGeometryValid =
    (packedValidityBits &
      JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT) !==
      0 && candidateMinimum.valid;
  const acceptedMinimumDeformationDeterminantValid =
    (packedValidityBits &
      JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT) !==
      0 && acceptedMinimum.valid;
  const localNumericsValid =
    (packedValidityBits &
      JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT) !==
    0;
  const geometryNumericsValid = [
    sourceMinimum,
    candidateMinimum,
    acceptedMinimum,
  ].every((entry) => entry.valid);
  const energyNumericsValid = [
    sourceEnergy,
    candidateEnergy,
    acceptedEnergy,
  ].every((entry) => entry.valid);
  const componentNorms = [
    inertiaNorm,
    materialNorm,
    externalForceNorm,
    targetNorm,
    contactNorm,
  ];
  const componentGradientNormsValid = componentNorms.every(
    (entry) => entry.valid,
  );
  const residualDiagnosticsValid = [
    gradientNorm,
    residualDenominator,
    relativeResidual,
  ].every((entry) => entry.valid);
  const updateDiagnosticsValid = [
    maximumUpdate,
    normalizedMaximumUpdate,
  ].every((entry) => entry.valid);
  const allNumericsValid =
    geometryNumericsValid &&
    energyNumericsValid &&
    componentGradientNormsValid &&
    residualDiagnosticsValid &&
    updateDiagnosticsValid &&
    maximumLocalNormalizedShift.valid;

  return {
    sourceMinimumDeformationDeterminant: sourceMinimum.value,
    candidateMinimumDeformationDeterminant: candidateMinimum.value,
    acceptedMinimumDeformationDeterminant: acceptedMinimum.value,
    sourceGeometryValid,
    candidateGeometryValid,
    acceptedMinimumDeformationDeterminantValid,
    localNumericsValid,
    geometryValid: sourceGeometryValid && candidateGeometryValid,
    assembledAccepted,
    assembledReverted,
    localFailureCount,
    revertCount,
    sourceEnergy: sourceEnergy.value,
    candidateEnergy: candidateEnergy.value,
    acceptedEnergy: acceptedEnergy.value,
    energyValid: packedEnergyValid && energyNumericsValid,
    componentGradientNorms: Object.freeze({
      inertia: inertiaNorm.value,
      material: materialNorm.value,
      externalForce: externalForceNorm.value,
      target: targetNorm.value,
      contact: contactNorm.value,
    }),
    componentGradientNormsValid,
    gradientNorm: gradientNorm.value,
    residualDenominator: residualDenominator.value,
    relativeResidual: relativeResidual.value,
    residualDiagnosticsValid,
    maximumUpdate: maximumUpdate.value,
    normalizedMaximumUpdate: normalizedMaximumUpdate.value,
    updateDiagnosticsValid,
    residualSatisfied,
    updateSatisfied,
    converged,
    finite:
      packedFinite &&
      allNumericsValid &&
      sourceGeometryValid &&
      candidateGeometryValid &&
      acceptedMinimumDeformationDeterminantValid &&
      localNumericsValid,
    iterationIndex,
    maximumLocalNormalizedShift: maximumLocalNormalizedShift.value,
    maximumLocalNormalizedShiftValid: maximumLocalNormalizedShift.valid,
    maximumBacktrackCount,
    infeasibleTrialCount,
    energyEvaluationCount,
    active,
  };
}

/**
 * Decode a tightly packed sequence of complete history records. The GPU may
 * expose fewer than 64 records, but never more than the frozen capacity.
 */
export function decodeJGS2GlobalizationHistory(
  packed: ArrayLike<number>,
): readonly JGS2GlobalizationHistoryRecord[] {
  if (
    packed.length % HISTORY_RECORD_FLOATS !== 0 ||
    packed.length >
      JGS2_GLOBALIZATION_HISTORY_CAPACITY * HISTORY_RECORD_FLOATS
  ) {
    throw new RangeError(
      `Globalization history must contain complete ${HISTORY_RECORD_FLOATS}-float ` +
        `records and no more than ${JGS2_GLOBALIZATION_HISTORY_CAPACITY} records.`,
    );
  }
  const records: JGS2GlobalizationHistoryRecord[] = [];
  for (
    let offset = 0;
    offset < packed.length;
    offset += HISTORY_RECORD_FLOATS
  ) {
    const record = Array.from(
      { length: HISTORY_RECORD_FLOATS },
      (_unused, index) => packed[offset + index]!,
    );
    records.push(decodeJGS2GlobalizationHistoryRecord(record));
  }
  return Object.freeze(records);
}
