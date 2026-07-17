import { describe, expect, it } from "vitest";

import {
  JGS2_GLOBALIZATION_ARMIJO_C1,
  JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT,
  JGS2_GLOBALIZATION_BACKTRACK_FACTOR,
  JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT,
  JGS2_GLOBALIZATION_CONTROL_VEC4S,
  JGS2_GLOBALIZATION_CONVERGENCE_COMPONENT_VEC4S,
  JGS2_GLOBALIZATION_DETERMINANT_FLOOR,
  JGS2_GLOBALIZATION_HISTORY_CAPACITY,
  JGS2_GLOBALIZATION_HISTORY_VEC4S,
  JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S,
  JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_CODES,
  JGS2_GLOBALIZATION_LOCAL_STATUS_LINE_SEARCH_FAILED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
  JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT,
  JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT,
  JGS2_GLOBALIZATION_MAX_BACKTRACKS,
  JGS2_GLOBALIZATION_MAX_NORMALIZED_SHIFT,
  JGS2_GLOBALIZATION_MAX_RUNTIME_TOLERANCE,
  JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER,
  JGS2_GLOBALIZATION_RELATIVE_EIGENVALUE_FLOOR,
  JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT,
  JGS2_GLOBALIZATION_TET_DIAGNOSTIC_VEC4S,
  JGS2_GLOBALIZATION_VALIDITY_BITS_MASK,
  decodeJGS2GlobalizationHistory,
  decodeJGS2GlobalizationHistoryRecord,
  decodeJGS2GlobalizationLocalDiagnostic,
} from "./jgs2-globalization";

describe("JGS2 GPU globalization ABI", () => {
  it("freezes the production policy, record strides, and status codes", () => {
    expect(JGS2_GLOBALIZATION_HISTORY_CAPACITY).toBe(64);
    expect(JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S).toBe(4);
    expect(JGS2_GLOBALIZATION_TET_DIAGNOSTIC_VEC4S).toBe(2);
    expect(JGS2_GLOBALIZATION_CONVERGENCE_COMPONENT_VEC4S).toBe(5);
    expect(JGS2_GLOBALIZATION_CONTROL_VEC4S).toBe(4);
    expect(JGS2_GLOBALIZATION_HISTORY_VEC4S).toBe(8);

    expect(JGS2_GLOBALIZATION_RELATIVE_EIGENVALUE_FLOOR).toBe(2 ** -16);
    expect(JGS2_GLOBALIZATION_MAX_NORMALIZED_SHIFT).toBe(1e-3);
    expect(JGS2_GLOBALIZATION_DETERMINANT_FLOOR).toBe(1e-4);
    expect(JGS2_GLOBALIZATION_ARMIJO_C1).toBe(1e-4);
    expect(JGS2_GLOBALIZATION_BACKTRACK_FACTOR).toBe(0.5);
    expect(JGS2_GLOBALIZATION_MAX_BACKTRACKS).toBe(12);
    expect(JGS2_GLOBALIZATION_MAX_RUNTIME_TOLERANCE).toBe(1e-3);
    expect(JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER).toBe(8);
    expect(JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT).toBe(1);
    expect(JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT).toBe(2);
    expect(JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT).toBe(4);
    expect(JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT).toBe(8);
    expect(JGS2_GLOBALIZATION_VALIDITY_BITS_MASK).toBe(15);

    expect(JGS2_GLOBALIZATION_LOCAL_STATUS_CODES).toEqual({
      accepted: 0,
      pinned: 1,
      zeroGradient: 2,
      shiftLimit: 3,
      nonDescent: 4,
      lineSearchFailed: 5,
      nonfinite: 6,
      choleskyFailed: 7,
    });
    expect(Object.isFrozen(JGS2_GLOBALIZATION_LOCAL_STATUS_CODES)).toBe(true);
  });

  it("decodes every lane of a four-vec4 local record", () => {
    const decoded = decodeJGS2GlobalizationLocalDiagnostic([
      1, 2, 3, 0.5,
      -0.25, 10, 0.01, 0.001,
      -4, -3.5, -8, 2e-5,
      0.125, 2, 3, JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
    ]);

    expect(decoded).toEqual({
      direction: [1, 2, 3],
      alpha: 0.5,
      minimumEigenvalue: -0.25,
      scale: 10,
      diagonalShift: 0.01,
      normalizedShift: 0.001,
      acceptedEnergyDelta: -4,
      armijoDeltaBound: -3.5,
      gradientDotDirection: -8,
      shiftedLinearResidual: 2e-5,
      minimumTrialDeformationDeterminant: 0.125,
      backtrackCount: 2,
      energyEvaluationCount: 3,
      statusCode: 0,
      status: "accepted",
      accepted: true,
      directionValid: true,
      alphaValid: true,
      positiveDefiniteDiagnosticsValid: true,
      energyDiagnosticsValid: true,
      minimumTrialDeformationDeterminantValid: true,
      finite: true,
    });
  });

  it("replaces nonfinite local numerics with finite zero sentinels", () => {
    const decoded = decodeJGS2GlobalizationLocalDiagnostic([
      Number.NaN, 2, 3, Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY, 10, 0.01, 0.001,
      Number.NaN, -3.5, -8, Number.POSITIVE_INFINITY,
      Number.NaN, 0, 0, JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
    ]);

    expect(decoded.direction).toEqual([0, 2, 3]);
    expect(decoded.alpha).toBe(0);
    expect(decoded.minimumEigenvalue).toBe(0);
    expect(decoded.acceptedEnergyDelta).toBe(0);
    expect(decoded.shiftedLinearResidual).toBe(0);
    expect(decoded.minimumTrialDeformationDeterminant).toBe(0);
    expect(decoded.directionValid).toBe(false);
    expect(decoded.alphaValid).toBe(false);
    expect(decoded.positiveDefiniteDiagnosticsValid).toBe(false);
    expect(decoded.energyDiagnosticsValid).toBe(false);
    expect(decoded.minimumTrialDeformationDeterminantValid).toBe(false);
    expect(decoded.finite).toBe(false);
    expect(
      [
        ...decoded.direction,
        decoded.alpha,
        decoded.minimumEigenvalue,
        decoded.acceptedEnergyDelta,
        decoded.shiftedLinearResidual,
        decoded.minimumTrialDeformationDeterminant,
      ].every(Number.isFinite),
    ).toBe(true);
  });

  it("distinguishes a finite geometry result from no-trial and nonfinite sentinels", () => {
    const record = new Float32Array(16);
    record[5] = 1;

    record[12] = 0;
    record[15] = JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT;
    const noTrial = decodeJGS2GlobalizationLocalDiagnostic(record);
    expect(noTrial.minimumTrialDeformationDeterminant).toBe(0);
    expect(noTrial.minimumTrialDeformationDeterminantValid).toBe(false);
    expect(noTrial.finite).toBe(true);

    record[15] = JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE;
    const nonfinite = decodeJGS2GlobalizationLocalDiagnostic(record);
    expect(nonfinite.minimumTrialDeformationDeterminant).toBe(0);
    expect(nonfinite.minimumTrialDeformationDeterminantValid).toBe(false);
    expect(nonfinite.finite).toBe(false);

    record[12] = 0.25;
    const nonfiniteEnergy = decodeJGS2GlobalizationLocalDiagnostic(record);
    expect(nonfiniteEnergy.minimumTrialDeformationDeterminant).toBe(0.25);
    expect(
      nonfiniteEnergy.minimumTrialDeformationDeterminantValid,
    ).toBe(true);
    expect(nonfiniteEnergy.finite).toBe(false);

    record[15] = JGS2_GLOBALIZATION_LOCAL_STATUS_LINE_SEARCH_FAILED;
    const finiteRejected = decodeJGS2GlobalizationLocalDiagnostic(record);
    expect(finiteRejected.minimumTrialDeformationDeterminant).toBe(0.25);
    expect(
      finiteRejected.minimumTrialDeformationDeterminantValid,
    ).toBe(true);
    expect(finiteRejected.finite).toBe(true);
  });

  it("rejects malformed local lengths, counts, and status codes", () => {
    expect(() =>
      decodeJGS2GlobalizationLocalDiagnostic(new Float32Array(15)),
    ).toThrow(/expected exactly 16/i);
    expect(() =>
      decodeJGS2GlobalizationLocalDiagnostic(new Float32Array(17)),
    ).toThrow(/expected exactly 16/i);

    const packed = new Float32Array(16);
    packed[5] = 1;
    packed[13] = 0.5;
    expect(() => decodeJGS2GlobalizationLocalDiagnostic(packed)).toThrow(
      /backtrackCount must be a nonnegative integer/i,
    );
    packed[13] = JGS2_GLOBALIZATION_MAX_BACKTRACKS + 1;
    expect(() => decodeJGS2GlobalizationLocalDiagnostic(packed)).toThrow(
      /backtrackCount must be a nonnegative integer/i,
    );
    packed[13] = 0;
    packed[14] = -1;
    expect(() => decodeJGS2GlobalizationLocalDiagnostic(packed)).toThrow(
      /energyEvaluationCount must be a nonnegative integer/i,
    );
    packed[14] = 0;
    packed[15] = JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED + 1;
    expect(() => decodeJGS2GlobalizationLocalDiagnostic(packed)).toThrow(
      /status must be an integer from 0 through 7/i,
    );
  });
});

describe("JGS2 GPU convergence history decoder", () => {
  it("decodes every exact control/history validity mask", () => {
    const packed = new Float32Array(32);
    packed[0] = 0.5;
    packed[1] = 0.4;
    packed[2] = 0.4;
    packed[18] = 1;
    packed[25] = 1;
    packed[31] = 1;

    for (
      let mask = 0;
      mask <= JGS2_GLOBALIZATION_VALIDITY_BITS_MASK;
      mask += 1
    ) {
      packed[3] = mask;
      const decoded = decodeJGS2GlobalizationHistoryRecord(packed);
      const sourceValid =
        (mask & JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT) !== 0;
      const candidateValid =
        (mask & JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT) !== 0;
      const acceptedValid =
        (mask & JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT) !== 0;
      const localValid =
        (mask & JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT) !== 0;
      expect(decoded.sourceGeometryValid, `source mask ${mask}`).toBe(
        sourceValid,
      );
      expect(decoded.candidateGeometryValid, `candidate mask ${mask}`).toBe(
        candidateValid,
      );
      expect(
        decoded.acceptedMinimumDeformationDeterminantValid,
        `accepted mask ${mask}`,
      ).toBe(acceptedValid);
      expect(decoded.localNumericsValid, `local mask ${mask}`).toBe(
        localValid,
      );
      expect(decoded.geometryValid, `geometry mask ${mask}`).toBe(
        sourceValid && candidateValid,
      );
      expect(decoded.finite, `finite mask ${mask}`).toBe(
        mask === JGS2_GLOBALIZATION_VALIDITY_BITS_MASK,
      );
    }
  });

  it("decodes every lane of an eight-vec4 history record", () => {
    const decoded = decodeJGS2GlobalizationHistoryRecord([
      1, -0.21, 1, JGS2_GLOBALIZATION_VALIDITY_BITS_MASK,
      0, 1, 2, 1,
      20, 19, 20, 1,
      3, 4, 5, 6,
      7, 8, 9, 0.25,
      10, 0.5, 0, 1,
      0, 1, 5, 0.0007,
      3, 4, 5, 1,
    ]);

    expect(decoded).toEqual({
      sourceMinimumDeformationDeterminant: 1,
      candidateMinimumDeformationDeterminant: -0.21,
      acceptedMinimumDeformationDeterminant: 1,
      sourceGeometryValid: true,
      candidateGeometryValid: true,
      acceptedMinimumDeformationDeterminantValid: true,
      localNumericsValid: true,
      geometryValid: true,
      assembledAccepted: false,
      assembledReverted: true,
      localFailureCount: 2,
      revertCount: 1,
      sourceEnergy: 20,
      candidateEnergy: 19,
      acceptedEnergy: 20,
      energyValid: true,
      componentGradientNorms: {
        inertia: 3,
        material: 4,
        externalForce: 5,
        target: 6,
        contact: 7,
      },
      componentGradientNormsValid: true,
      gradientNorm: 8,
      residualDenominator: 9,
      relativeResidual: 0.25,
      residualDiagnosticsValid: true,
      maximumUpdate: 10,
      normalizedMaximumUpdate: 0.5,
      updateDiagnosticsValid: true,
      residualSatisfied: false,
      updateSatisfied: true,
      converged: false,
      finite: true,
      iterationIndex: 5,
      maximumLocalNormalizedShift: 0.0007,
      maximumLocalNormalizedShiftValid: true,
      maximumBacktrackCount: 3,
      infeasibleTrialCount: 4,
      energyEvaluationCount: 5,
      active: true,
    });
  });

  it("combines packed validity flags with finite zero-sentinel decoding", () => {
    const packed = new Float32Array(32);
    packed.set([
      Number.NaN, Number.POSITIVE_INFINITY, 1, 0,
      0, 1, 0, 1,
      Number.NaN, 2, Number.NEGATIVE_INFINITY, 0,
      Number.NaN, 4, 5, 6,
      7, Number.POSITIVE_INFINITY, 1, Number.NaN,
      Number.NaN, 0.5, 0, 0,
      0, 1, 0, Number.NaN,
      0, 0, 0, 1,
    ]);
    const decoded = decodeJGS2GlobalizationHistoryRecord(packed);

    expect(decoded.sourceMinimumDeformationDeterminant).toBe(0);
    expect(decoded.candidateMinimumDeformationDeterminant).toBe(0);
    expect(decoded.sourceGeometryValid).toBe(false);
    expect(decoded.candidateGeometryValid).toBe(false);
    expect(decoded.acceptedMinimumDeformationDeterminantValid).toBe(false);
    expect(decoded.localNumericsValid).toBe(false);
    expect(decoded.geometryValid).toBe(false);
    expect(decoded.sourceEnergy).toBe(0);
    expect(decoded.acceptedEnergy).toBe(0);
    expect(decoded.energyValid).toBe(false);
    expect(decoded.componentGradientNorms.inertia).toBe(0);
    expect(decoded.componentGradientNormsValid).toBe(false);
    expect(decoded.gradientNorm).toBe(0);
    expect(decoded.relativeResidual).toBe(0);
    expect(decoded.residualDiagnosticsValid).toBe(false);
    expect(decoded.maximumUpdate).toBe(0);
    expect(decoded.updateDiagnosticsValid).toBe(false);
    expect(decoded.maximumLocalNormalizedShift).toBe(0);
    expect(decoded.maximumLocalNormalizedShiftValid).toBe(false);
    expect(decoded.finite).toBe(false);

    const scalarValues = [
      decoded.sourceMinimumDeformationDeterminant,
      decoded.candidateMinimumDeformationDeterminant,
      decoded.sourceEnergy,
      decoded.acceptedEnergy,
      ...Object.values(decoded.componentGradientNorms),
      decoded.gradientNorm,
      decoded.relativeResidual,
      decoded.maximumUpdate,
      decoded.maximumLocalNormalizedShift,
    ];
    expect(scalarValues.every(Number.isFinite)).toBe(true);
  });

  it("preserves a valid reverted source minimum when the candidate is nonfinite", () => {
    const packed = new Float32Array(32);
    packed[0] = 0.5;
    packed[2] = 0.5;
    packed[3] =
      JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT |
      JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT |
      JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT;
    packed[5] = 1;
    packed[7] = 1;
    packed[18] = 1;
    packed[25] = 1;
    packed[31] = 1;

    const decoded = decodeJGS2GlobalizationHistoryRecord(packed);
    expect(decoded.sourceGeometryValid).toBe(true);
    expect(decoded.candidateGeometryValid).toBe(false);
    expect(decoded.acceptedMinimumDeformationDeterminant).toBe(0.5);
    expect(
      decoded.acceptedMinimumDeformationDeterminantValid,
    ).toBe(true);
    expect(decoded.localNumericsValid).toBe(true);
    expect(decoded.geometryValid).toBe(false);
    expect(decoded.finite).toBe(false);
  });

  it("keeps finite infeasible geometry finite but rejects invalid local numerics", () => {
    const packed = new Float32Array(32);
    packed[0] = 0.5;
    packed[1] = -0.25;
    packed[2] = 0.5;
    packed[3] = JGS2_GLOBALIZATION_VALIDITY_BITS_MASK;
    packed[5] = 1;
    packed[7] = 1;
    packed[18] = 1;
    packed[25] = 1;
    packed[31] = 1;

    const finiteInfeasible = decodeJGS2GlobalizationHistoryRecord(packed);
    expect(finiteInfeasible.geometryValid).toBe(true);
    expect(finiteInfeasible.assembledReverted).toBe(true);
    expect(finiteInfeasible.finite).toBe(true);

    packed[3] &= ~JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT;
    const invalidLocal = decodeJGS2GlobalizationHistoryRecord(packed);
    expect(invalidLocal.geometryValid).toBe(true);
    expect(invalidLocal.localNumericsValid).toBe(false);
    expect(invalidLocal.finite).toBe(false);
  });

  it("decodes bounded sequences of complete history records", () => {
    const record = new Float32Array(32);
    record[3] = JGS2_GLOBALIZATION_VALIDITY_BITS_MASK;
    record[11] = 1;
    record[18] = 1;
    record[25] = 1;
    record[31] = 1;
    const packed = new Float32Array(64);
    packed.set(record, 0);
    packed.set(record, 32);
    packed[32 + 26] = 1;

    const records = decodeJGS2GlobalizationHistory(packed);
    expect(records).toHaveLength(2);
    expect(records.map((entry) => entry.iterationIndex)).toEqual([0, 1]);
    expect(Object.isFrozen(records)).toBe(true);

    expect(() => decodeJGS2GlobalizationHistory(new Float32Array(31))).toThrow(
      /complete 32-float records/i,
    );
    expect(() =>
      decodeJGS2GlobalizationHistory(
        new Float32Array((JGS2_GLOBALIZATION_HISTORY_CAPACITY + 1) * 32),
      ),
    ).toThrow(/no more than 64 records/i);
  });

  it("rejects malformed history counts, indices, flags, and lengths", () => {
    expect(() =>
      decodeJGS2GlobalizationHistoryRecord(new Float32Array(31)),
    ).toThrow(/expected exactly 32/i);

    const packed = new Float32Array(32);
    packed[6] = 0.5;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /localFailureCount must be a nonnegative integer/i,
    );
    packed[6] = 0;
    packed[7] = 2;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /revertCount must be a nonnegative integer/i,
    );
    packed[7] = 0;
    packed[26] = JGS2_GLOBALIZATION_HISTORY_CAPACITY;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /iterationIndex must be a nonnegative integer/i,
    );
    packed[26] = 0;
    packed[3] = JGS2_GLOBALIZATION_VALIDITY_BITS_MASK + 1;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /geometryValidityBits must be a nonnegative integer/i,
    );
    packed[3] = 0.5;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /geometryValidityBits must be a nonnegative integer/i,
    );
    packed[3] = -1;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /geometryValidityBits must be a nonnegative integer/i,
    );
    packed[3] = 0;
    packed[28] = JGS2_GLOBALIZATION_MAX_BACKTRACKS + 1;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /maximumBacktrackCount must be a nonnegative integer/i,
    );
    packed[28] = 0;
    packed[31] = 2;
    expect(() => decodeJGS2GlobalizationHistoryRecord(packed)).toThrow(
      /active must be encoded as exactly 0 or 1/i,
    );
  });
});
