import { describe, expect, it } from "vitest";

import manifestJson from "../../manifests/phase1-globalization.v1.json?raw";
import { PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID } from "../scenes/phase1";
import {
  JGS2_LOCAL_ARMIJO_C1,
  JGS2_LOCAL_BACKTRACK_FACTOR,
  JGS2_LOCAL_MAX_BACKTRACKS,
  JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
  JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR,
  JGS2_RUNTIME_MAXIMUM_CONVERGENCE_TOLERANCE,
  JGS2_TINY_REFERENCE_RESIDUAL_TOLERANCE,
  PHASE1_ACCEPTED_DETERMINANT_FLOOR,
} from "../simulation/cpu";
import { validatePhase1GlobalizationManifest } from "./phase1-globalization-manifest";

function manifestValue(): unknown {
  return JSON.parse(manifestJson) as unknown;
}

describe("versioned Phase 1 globalization reference manifest", () => {
  it("binds every frozen CPU reference policy and corpus gate", () => {
    const manifest = validatePhase1GlobalizationManifest(manifestValue());
    expect(manifest.sourceFixtures.cubatureFixtureId).toBe(
      PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID,
    );
    expect(manifest.positiveDefiniteTreatment).toMatchObject({
      f32UnitRoundoff: 2 ** -24,
      eigenvalueFloorMultiplier: 256,
      relativeEigenvalueFloor: JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR,
      maximumNormalizedShift: JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
      scales: [1e-12, 1, 1e12],
    });
    expect(manifest.lineSearch).toEqual({
      armijoC1: JGS2_LOCAL_ARMIJO_C1,
      backtrackFactor: JGS2_LOCAL_BACKTRACK_FACTOR,
      maximumBacktracks: JGS2_LOCAL_MAX_BACKTRACKS,
      determinantFloor: PHASE1_ACCEPTED_DETERMINANT_FLOOR,
      feasibilityBeforeEnergy: true,
      failureUpdate: "zero",
    });
    expect(manifest.convergence).toMatchObject({
      tinyReferenceResidualTolerance: JGS2_TINY_REFERENCE_RESIDUAL_TOLERANCE,
      maximumRuntimeTolerance: JGS2_RUNTIME_MAXIMUM_CONVERGENCE_TOLERANCE,
      requiresResidualAndUpdate: true,
      sceneScale: "initial-dynamic-aabb-diagonal",
    });
    expect(manifest.referenceGates.canonicalPackedLocalSystemCount).toBe(240);
    expect(manifest.restrictedObjectiveTerms).toEqual([
      "implicit-euler-inertia",
      "stable-neo-hookean-material",
    ]);
    expect(manifest.runtimeStatus).toEqual({
      cpuMaterialInertiaReference: "implemented",
      compositeForcesAndTargets: "pending",
      gpuProduction: "pending",
      qualifiesPhase1Exit: false,
    });
    expect(manifest.runtimeStatus.qualifiesPhase1Exit).toBe(false);
  });

  it("rejects safety-policy weakening and incomplete case inventories", () => {
    const weakShift = structuredClone(manifestValue()) as {
      positiveDefiniteTreatment: { maximumNormalizedShift: number };
    };
    weakShift.positiveDefiniteTreatment.maximumNormalizedShift = Number.NaN;
    expect(() => validatePhase1GlobalizationManifest(weakShift)).toThrow(
      /maximumNormalizedShift must be finite/i,
    );

    const missingCases = structuredClone(manifestValue()) as { caseIds: string[] };
    missingCases.caseIds = [];
    expect(() => validatePhase1GlobalizationManifest(missingCases)).toThrow(
      /caseIds must be a nonempty array/i,
    );

    const unknownField = structuredClone(manifestValue()) as Record<
      string,
      unknown
    >;
    unknownField.unversionedPolicy = true;
    expect(() => validatePhase1GlobalizationManifest(unknownField)).toThrow(
      /keys must exactly match/i,
    );
  });
});
