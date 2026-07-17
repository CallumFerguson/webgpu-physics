import { describe, expect, it } from "vitest";

import manifestV1Json from "../../manifests/phase1-globalization.v1.json?raw";
import manifestV2Json from "../../manifests/phase1-globalization.v2.json?raw";
import nonlinearCubatureGpuSource from "../../tests/e2e/nonlinear-cubature-gpu.spec.ts?raw";
import nonlinearGlobalizationGpuSource from "../../tests/e2e/nonlinear-globalization-gpu.spec.ts?raw";
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
import {
  JGS2_GLOBALIZATION_CONTROL_VEC4S,
  JGS2_GLOBALIZATION_HISTORY_CAPACITY,
  JGS2_GLOBALIZATION_HISTORY_VEC4S,
  JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S,
  JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER,
  JGS2_GLOBALIZATION_TET_DIAGNOSTIC_VEC4S,
  JGS2_CONVERGENCE_COMPONENT_VEC4S,
  JGS2_UNIFORM_BYTES,
} from "../simulation/gpu";
import { validatePhase1GlobalizationManifest } from "./phase1-globalization-manifest";

function manifestValue(): unknown {
  return JSON.parse(manifestV2Json) as unknown;
}

function historicalManifestValue(): unknown {
  return JSON.parse(manifestV1Json) as unknown;
}

const GPU_TEST_SOURCE_TEXT: Readonly<Record<string, string>> = Object.freeze({
  "tests/e2e/nonlinear-globalization-gpu.spec.ts":
    nonlinearGlobalizationGpuSource,
  "tests/e2e/nonlinear-cubature-gpu.spec.ts": nonlinearCubatureGpuSource,
});

describe("versioned Phase 1 globalization reference manifest", () => {
  it("binds every frozen CPU reference policy and corpus gate", () => {
    const manifest = validatePhase1GlobalizationManifest(manifestValue());
    expect(manifest.schemaVersion).toBe(2);
    if (manifest.schemaVersion !== 2) {
      throw new Error("The checked-in v2 manifest validated as another version.");
    }
    expect(manifest.sourceFixtures.cubatureFixtureId).toBe(
      PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID,
    );
    expect(manifest.sourceFixtures.gpuTestFiles).toEqual([
      "tests/e2e/nonlinear-globalization-gpu.spec.ts",
      "tests/e2e/nonlinear-cubature-gpu.spec.ts",
    ]);
    expect(manifest.gpuTestSelectors).toEqual([
      {
        source: "tests/e2e/nonlinear-globalization-gpu.spec.ts",
        selector:
          "P1-GPU-GLOBALIZATION: assembled Jacobi feasibility reverts the complete pose",
      },
      {
        source: "tests/e2e/nonlinear-globalization-gpu.spec.ts",
        selector:
          "P1-GPU-GLOBALIZATION: initial source uses production GPU-f32 feasibility preflight",
      },
      {
        source: "tests/e2e/nonlinear-globalization-gpu.spec.ts",
        selector:
          "P1-GPU-GLOBALIZATION: local Armijo checks determinant feasibility before energy",
      },
      {
        source: "tests/e2e/nonlinear-globalization-gpu.spec.ts",
        selector:
          "P1-GPU-GLOBALIZATION: shared production shift solver matches the CPU policy",
      },
      {
        source: "tests/e2e/nonlinear-globalization-gpu.spec.ts",
        selector:
          "P1-GPU-GLOBALIZATION: stable floor penalty participates in Armijo and convergence",
      },
      {
        source: "tests/e2e/nonlinear-globalization-gpu.spec.ts",
        selector:
          "P1-GPU-GLOBALIZATION: convergence reduction requires residual, update, feasibility, and no failures",
      },
      {
        source: "tests/e2e/nonlinear-cubature-gpu.spec.ts",
        selector:
          "P1-EC-12-GPU: production nonlinear Cubature updates match the packed CPU reference",
      },
    ]);
    expect(manifest.positiveDefiniteTreatment).toMatchObject({
      f32UnitRoundoff: 2 ** -24,
      eigenvalueFloorMultiplier: 256,
      relativeEigenvalueFloor: JGS2_LOCAL_RELATIVE_EIGENVALUE_FLOOR,
      maximumNormalizedShift: JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
      positionResolutionMultiplier:
        JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER,
      scales: [1e-12, 1, 1e12],
    });
    expect(manifest.lineSearch).toEqual({
      armijoC1: JGS2_LOCAL_ARMIJO_C1,
      backtrackFactor: JGS2_LOCAL_BACKTRACK_FACTOR,
      maximumBacktracks: JGS2_LOCAL_MAX_BACKTRACKS,
      determinantFloor: PHASE1_ACCEPTED_DETERMINANT_FLOOR,
      feasibilityBeforeEnergy: true,
      energyComparison: "delta-from-frozen-source",
      failureUpdate: "zero",
    });
    expect(manifest.convergence).toMatchObject({
      tinyReferenceResidualTolerance: JGS2_TINY_REFERENCE_RESIDUAL_TOLERANCE,
      maximumRuntimeTolerance: JGS2_RUNTIME_MAXIMUM_CONVERGENCE_TOLERANCE,
      requiresResidualAndUpdate: true,
      sceneScale: "initial-dynamic-aabb-diagonal",
      gpuHistoryCapacity: JGS2_GLOBALIZATION_HISTORY_CAPACITY,
    });
    expect(manifest.referenceGates.canonicalPackedLocalSystemCount).toBe(240);
    expect(manifest.restrictedObjectiveTerms).toEqual([
      "implicit-euler-inertia",
      "stable-neo-hookean-material",
      "analytic-floor-penalty",
    ]);
    expect(manifest.gpuRuntime).toMatchObject({
      enabledMaterial: "stable-neo-hookean",
      mixedMaterialSolve: "rejected",
      uniformBytes: JGS2_UNIFORM_BYTES,
      localDiagnosticVec4s: JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S,
      tetDiagnosticVec4s: JGS2_GLOBALIZATION_TET_DIAGNOSTIC_VEC4S,
      convergenceComponentVec4s: JGS2_CONVERGENCE_COMPONENT_VEC4S,
      controlVec4s: JGS2_GLOBALIZATION_CONTROL_VEC4S,
      historyVec4s: JGS2_GLOBALIZATION_HISTORY_VEC4S,
      historyCapacity: JGS2_GLOBALIZATION_HISTORY_CAPACITY,
      steppingReadback: "none",
      initializationReadback: "gpu-source-feasibility-once",
      explicitDiagnosticReadback: "on-request",
      assembledEnergyAcceptance: "diagnostic-only",
      pinnedTargetApplication: "assembled-feasibility-gated",
    });
    expect(manifest.runtimeStatus).toEqual({
      cpuMaterialInertiaReference: "implemented",
      compositeForcesAndTargets: "pending",
      gpuProduction: "implemented-material-inertia-floor",
      qualifiesPhase1Exit: false,
    });
    expect(manifest.runtimeStatus.qualifiesPhase1Exit).toBe(false);
  });

  it("keeps the checked-in v1 manifest executable with historical semantics", () => {
    const manifest = validatePhase1GlobalizationManifest(
      historicalManifestValue(),
    );
    expect(manifest.schemaVersion).toBe(1);
    if (manifest.schemaVersion !== 1) {
      throw new Error("The checked-in v1 manifest validated as another version.");
    }

    expect(manifest.id).toBe(
      "phase1.globalization-material-inertia-reference",
    );
    expect(manifest.sourceFixtures).toEqual({
      materialManifest: "manifests/phase1-scenes.v1.json",
      cubatureManifest: "manifests/phase1-cubature.v1.json",
      cubatureFixtureId: "phase1.nonlinear-cubature-beam",
    });
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
    expect("gpuRuntime" in manifest).toBe(false);
    expect("gpuTestSelectors" in manifest).toBe(false);
  });

  it("resolves every frozen GPU selector to its cited test source", () => {
    const manifest = validatePhase1GlobalizationManifest(manifestValue());
    if (manifest.schemaVersion !== 2) {
      throw new Error("GPU selector provenance requires the v2 manifest.");
    }

    for (const entry of manifest.gpuTestSelectors) {
      const source = GPU_TEST_SOURCE_TEXT[entry.source];
      expect(source, entry.source).toBeDefined();
      expect(source, `${entry.source} > ${entry.selector}`).toContain(
        `test(${JSON.stringify(entry.selector)}`,
      );
    }
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

    const broadReadbackClaim = structuredClone(manifestValue()) as {
      gpuRuntime: Record<string, unknown>;
    };
    delete broadReadbackClaim.gpuRuntime.steppingReadback;
    delete broadReadbackClaim.gpuRuntime.initializationReadback;
    broadReadbackClaim.gpuRuntime.productionReadback = "none";
    expect(() =>
      validatePhase1GlobalizationManifest(broadReadbackClaim),
    ).toThrow(/gpuRuntime keys must exactly match/i);

    const missingInitializationReadback = structuredClone(manifestValue()) as {
      gpuRuntime: { initializationReadback: string };
    };
    missingInitializationReadback.gpuRuntime.initializationReadback = "none";
    expect(() =>
      validatePhase1GlobalizationManifest(missingInitializationReadback),
    ).toThrow(/initializationReadback must equal gpu-source-feasibility-once/i);

    const postGatePinnedSnap = structuredClone(manifestValue()) as {
      gpuRuntime: { pinnedTargetApplication: string };
    };
    postGatePinnedSnap.gpuRuntime.pinnedTargetApplication = "finalize-snap";
    expect(() =>
      validatePhase1GlobalizationManifest(postGatePinnedSnap),
    ).toThrow(
      /pinnedTargetApplication must equal assembled-feasibility-gated/i,
    );
  });

  it("rejects malformed or incomplete GPU source provenance", () => {
    const missingSource = structuredClone(manifestValue()) as {
      sourceFixtures: { gpuTestFiles: string[] };
    };
    missingSource.sourceFixtures.gpuTestFiles.pop();
    expect(() => validatePhase1GlobalizationManifest(missingSource)).toThrow(
      /frozen GPU test sources/i,
    );

    const wrongSource = structuredClone(manifestValue()) as {
      sourceFixtures: { gpuTestFiles: string[] };
    };
    wrongSource.sourceFixtures.gpuTestFiles[1] =
      "tests/e2e/unrelated.spec.ts";
    expect(() => validatePhase1GlobalizationManifest(wrongSource)).toThrow(
      /frozen GPU test sources/i,
    );

    const retrofittedV1 = structuredClone(historicalManifestValue()) as {
      sourceFixtures: Record<string, unknown>;
    };
    retrofittedV1.sourceFixtures.gpuTestFiles = [];
    expect(() => validatePhase1GlobalizationManifest(retrofittedV1)).toThrow(
      /sourceFixtures keys must exactly match/i,
    );
  });

  it("rejects missing, renamed, malformed, or misattributed GPU selectors", () => {
    const missingSelector = structuredClone(manifestValue()) as {
      gpuTestSelectors: unknown[];
    };
    missingSelector.gpuTestSelectors.pop();
    expect(() => validatePhase1GlobalizationManifest(missingSelector)).toThrow(
      /frozen Phase 1 GPU selector inventory/i,
    );

    const renamedSelector = structuredClone(manifestValue()) as {
      gpuTestSelectors: Array<{ selector: string }>;
    };
    renamedSelector.gpuTestSelectors[0]!.selector += " renamed";
    expect(() => validatePhase1GlobalizationManifest(renamedSelector)).toThrow(
      /gpuTestSelectors\[0\]\.selector must equal/i,
    );

    const misattributedSelector = structuredClone(manifestValue()) as {
      gpuTestSelectors: Array<{ source: string }>;
    };
    misattributedSelector.gpuTestSelectors[0]!.source =
      "tests/e2e/nonlinear-cubature-gpu.spec.ts";
    expect(() =>
      validatePhase1GlobalizationManifest(misattributedSelector),
    ).toThrow(/gpuTestSelectors\[0\]\.source must equal/i);

    const malformedSelector = structuredClone(manifestValue()) as {
      gpuTestSelectors: Array<Record<string, unknown>>;
    };
    malformedSelector.gpuTestSelectors[0]!.unversionedTag = true;
    expect(() => validatePhase1GlobalizationManifest(malformedSelector)).toThrow(
      /gpuTestSelectors\[0\] keys must exactly match/i,
    );
  });
});
