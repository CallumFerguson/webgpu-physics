export const PHASE1_GLOBALIZATION_MANIFEST_SCHEMA =
  "org.jgs2.phase1-globalization" as const;
export const PHASE1_GLOBALIZATION_LEGACY_MANIFEST_VERSION = 1 as const;
export const PHASE1_GLOBALIZATION_MANIFEST_VERSION = 2 as const;

const EXPECTED_CPU_SOURCE_FIXTURES = Object.freeze({
  materialManifest: "manifests/phase1-scenes.v1.json",
  cubatureManifest: "manifests/phase1-cubature.v1.json",
  cubatureFixtureId: "phase1.nonlinear-cubature-beam",
});
const EXPECTED_GPU_TEST_FILES = Object.freeze([
  "tests/e2e/nonlinear-globalization-gpu.spec.ts",
  "tests/e2e/nonlinear-cubature-gpu.spec.ts",
] as const);
const EXPECTED_GPU_TEST_SELECTORS = Object.freeze([
  {
    source: EXPECTED_GPU_TEST_FILES[0],
    selector:
      "P1-GPU-GLOBALIZATION: assembled Jacobi feasibility reverts the complete pose",
  },
  {
    source: EXPECTED_GPU_TEST_FILES[0],
    selector:
      "P1-GPU-GLOBALIZATION: initial source uses production GPU-f32 feasibility preflight",
  },
  {
    source: EXPECTED_GPU_TEST_FILES[0],
    selector:
      "P1-GPU-GLOBALIZATION: local Armijo checks determinant feasibility before energy",
  },
  {
    source: EXPECTED_GPU_TEST_FILES[0],
    selector:
      "P1-GPU-GLOBALIZATION: shared production shift solver matches the CPU policy",
  },
  {
    source: EXPECTED_GPU_TEST_FILES[0],
    selector:
      "P1-GPU-GLOBALIZATION: stable floor penalty participates in Armijo and convergence",
  },
  {
    source: EXPECTED_GPU_TEST_FILES[0],
    selector:
      "P1-GPU-GLOBALIZATION: convergence reduction requires residual, update, feasibility, and no failures",
  },
  {
    source: EXPECTED_GPU_TEST_FILES[1],
    selector:
      "P1-EC-12-GPU: production nonlinear Cubature updates match the packed CPU reference",
  },
] as const);
const EXPECTED_CASE_IDS = Object.freeze([
  "symmetric-spectrum",
  "scale-covariance",
  "shift-cap-boundary",
  "armijo-zero-one-multiple",
  "infeasible-before-energy",
  "nonfinite-determinant-propagation",
  "selected-packed-restricted-energy",
  "assembled-jacobi-revert",
  "component-aware-convergence",
  "canonical-packed-local-systems",
] as const);
const EXPECTED_V1_RESTRICTED_OBJECTIVE_TERMS = Object.freeze([
  "implicit-euler-inertia",
  "stable-neo-hookean-material",
] as const);
const EXPECTED_V2_RESTRICTED_OBJECTIVE_TERMS = Object.freeze([
  "implicit-euler-inertia",
  "stable-neo-hookean-material",
  "analytic-floor-penalty",
] as const);

interface Phase1GlobalizationManifestCommon {
  readonly schema: typeof PHASE1_GLOBALIZATION_MANIFEST_SCHEMA;
  readonly caseIds: readonly string[];
}

export interface Phase1GlobalizationManifestV1
  extends Phase1GlobalizationManifestCommon {
  readonly schemaVersion: typeof PHASE1_GLOBALIZATION_LEGACY_MANIFEST_VERSION;
  readonly id: "phase1.globalization-material-inertia-reference";
  readonly sourceFixtures: {
    readonly materialManifest: string;
    readonly cubatureManifest: string;
    readonly cubatureFixtureId: string;
  };
  readonly positiveDefiniteTreatment: {
    readonly f32UnitRoundoff: number;
    readonly eigenvalueFloorMultiplier: number;
    readonly relativeEigenvalueFloor: number;
    readonly maximumNormalizedShift: number;
    readonly scales: readonly number[];
  };
  readonly lineSearch: {
    readonly armijoC1: number;
    readonly backtrackFactor: number;
    readonly maximumBacktracks: number;
    readonly determinantFloor: number;
    readonly feasibilityBeforeEnergy: true;
    readonly failureUpdate: "zero";
  };
  readonly convergence: {
    readonly tinyReferenceResidualTolerance: number;
    readonly maximumRuntimeTolerance: number;
    readonly requiresResidualAndUpdate: true;
    readonly sceneScale: "initial-dynamic-aabb-diagonal";
  };
  readonly referenceGates: {
    readonly restrictedGradientRelativeError: number;
    readonly restrictedHessianRelativeError: number;
    readonly shiftedLinearResidual: number;
    readonly canonicalPackedLocalSystemCount: number;
    readonly assembledRevertDeterminant: number;
  };
  readonly restrictedObjectiveTerms: readonly [
    "implicit-euler-inertia",
    "stable-neo-hookean-material",
  ];
  readonly runtimeStatus: {
    readonly cpuMaterialInertiaReference: "implemented";
    readonly compositeForcesAndTargets: "pending";
    readonly gpuProduction: "pending";
    readonly qualifiesPhase1Exit: false;
  };
}

export type Phase1GlobalizationGpuTestFile =
  (typeof EXPECTED_GPU_TEST_FILES)[number];

export interface Phase1GlobalizationGpuTestSelector {
  readonly source: Phase1GlobalizationGpuTestFile;
  readonly selector: string;
}

export interface Phase1GlobalizationManifestV2
  extends Phase1GlobalizationManifestCommon {
  readonly schemaVersion: typeof PHASE1_GLOBALIZATION_MANIFEST_VERSION;
  readonly id: "phase1.globalization-material-inertia-cpu-gpu";
  readonly sourceFixtures: {
    readonly materialManifest: string;
    readonly cubatureManifest: string;
    readonly cubatureFixtureId: string;
    readonly gpuTestFiles: readonly [
      "tests/e2e/nonlinear-globalization-gpu.spec.ts",
      "tests/e2e/nonlinear-cubature-gpu.spec.ts",
    ];
  };
  readonly positiveDefiniteTreatment: {
    readonly f32UnitRoundoff: number;
    readonly eigenvalueFloorMultiplier: number;
    readonly relativeEigenvalueFloor: number;
    readonly maximumNormalizedShift: number;
    readonly positionResolutionMultiplier: number;
    readonly scales: readonly number[];
  };
  readonly lineSearch: {
    readonly armijoC1: number;
    readonly backtrackFactor: number;
    readonly maximumBacktracks: number;
    readonly determinantFloor: number;
    readonly feasibilityBeforeEnergy: true;
    readonly energyComparison: "delta-from-frozen-source";
    readonly failureUpdate: "zero";
  };
  readonly convergence: {
    readonly tinyReferenceResidualTolerance: number;
    readonly maximumRuntimeTolerance: number;
    readonly requiresResidualAndUpdate: true;
    readonly sceneScale: "initial-dynamic-aabb-diagonal";
    readonly gpuHistoryCapacity: number;
  };
  readonly referenceGates: {
    readonly restrictedGradientRelativeError: number;
    readonly restrictedHessianRelativeError: number;
    readonly shiftedLinearResidual: number;
    readonly gpuShiftedLinearResidual: number;
    readonly canonicalPackedLocalSystemCount: number;
    readonly assembledRevertDeterminant: number;
  };
  readonly gpuRuntime: {
    readonly enabledMaterial: "stable-neo-hookean";
    readonly mixedMaterialSolve: "rejected";
    readonly uniformBytes: number;
    readonly localDiagnosticVec4s: number;
    readonly tetDiagnosticVec4s: number;
    readonly convergenceComponentVec4s: number;
    readonly controlVec4s: number;
    readonly historyVec4s: number;
    readonly historyCapacity: number;
    readonly steppingReadback: "none";
    readonly initializationReadback: "gpu-source-feasibility-once";
    readonly explicitDiagnosticReadback: "on-request";
    readonly assembledEnergyAcceptance: "diagnostic-only";
    readonly pinnedTargetApplication: "assembled-feasibility-gated";
  };
  readonly restrictedObjectiveTerms: readonly [
    "implicit-euler-inertia",
    "stable-neo-hookean-material",
    "analytic-floor-penalty",
  ];
  readonly gpuTestSelectors: readonly Phase1GlobalizationGpuTestSelector[];
  readonly runtimeStatus: {
    readonly cpuMaterialInertiaReference: "implemented";
    readonly compositeForcesAndTargets: "pending";
    readonly gpuProduction: "implemented-material-inertia-floor";
    readonly qualifiesPhase1Exit: false;
  };
}

export type Phase1GlobalizationManifest =
  | Phase1GlobalizationManifestV1
  | Phase1GlobalizationManifestV2;

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void {
  const actual = Object.keys(value).sort();
  const required = [...expected].sort();
  if (
    actual.length !== required.length ||
    actual.some((key, index) => key !== required[index])
  ) {
    throw new Error(`${label} keys must exactly match ${required.join(", ")}.`);
  }
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function positive(value: unknown, label: string): number {
  const result = finite(value, label);
  if (!(result > 0)) throw new Error(`${label} must be positive.`);
  return result;
}

function literal<T>(value: unknown, expected: T, label: string): void {
  if (value !== expected) {
    throw new Error(`${label} must equal ${String(expected)}.`);
  }
}

function stableId(value: unknown, label: string): string {
  if (
    typeof value !== "string" ||
    !/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/.test(value)
  ) {
    throw new Error(`${label} must be a stable lowercase ID.`);
  }
  return value;
}

/** Fail closed on protocol drift before a run can be cited as Phase 1 evidence. */
export function validatePhase1GlobalizationManifest(
  value: unknown,
): Phase1GlobalizationManifest {
  const root = record(value, "Phase 1 globalization manifest");
  literal(root.schema, PHASE1_GLOBALIZATION_MANIFEST_SCHEMA, "schema");
  const schemaVersion = finite(root.schemaVersion, "schemaVersion");
  if (
    schemaVersion !== PHASE1_GLOBALIZATION_LEGACY_MANIFEST_VERSION &&
    schemaVersion !== PHASE1_GLOBALIZATION_MANIFEST_VERSION
  ) {
    throw new Error(
      `schemaVersion must equal ` +
        `${PHASE1_GLOBALIZATION_LEGACY_MANIFEST_VERSION} or ` +
        `${PHASE1_GLOBALIZATION_MANIFEST_VERSION}.`,
    );
  }
  const isV2 = schemaVersion === PHASE1_GLOBALIZATION_MANIFEST_VERSION;
  exactKeys(
    root,
    [
      "schema",
      "schemaVersion",
      "id",
      "sourceFixtures",
      "positiveDefiniteTreatment",
      "lineSearch",
      "convergence",
      "referenceGates",
      ...(isV2 ? ["gpuRuntime", "gpuTestSelectors"] : []),
      "caseIds",
      "restrictedObjectiveTerms",
      "runtimeStatus",
    ],
    "Phase 1 globalization manifest",
  );
  literal(
    root.id,
    isV2
      ? "phase1.globalization-material-inertia-cpu-gpu"
      : "phase1.globalization-material-inertia-reference",
    "id",
  );

  const sources = record(root.sourceFixtures, "sourceFixtures");
  exactKeys(
    sources,
    [
      ...Object.keys(EXPECTED_CPU_SOURCE_FIXTURES),
      ...(isV2 ? ["gpuTestFiles"] : []),
    ],
    "sourceFixtures",
  );
  for (const [key, expected] of Object.entries(
    EXPECTED_CPU_SOURCE_FIXTURES,
  )) {
    literal(sources[key], expected, `sourceFixtures.${key}`);
  }
  if (isV2) {
    if (
      !Array.isArray(sources.gpuTestFiles) ||
      sources.gpuTestFiles.length !== EXPECTED_GPU_TEST_FILES.length ||
      sources.gpuTestFiles.some(
        (entry, index) => entry !== EXPECTED_GPU_TEST_FILES[index],
      )
    ) {
      throw new Error(
        "sourceFixtures.gpuTestFiles must exactly match the frozen GPU test sources.",
      );
    }
  }

  const pd = record(root.positiveDefiniteTreatment, "positiveDefiniteTreatment");
  exactKeys(
    pd,
    [
      "f32UnitRoundoff",
      "eigenvalueFloorMultiplier",
      "relativeEigenvalueFloor",
      "maximumNormalizedShift",
      ...(isV2 ? ["positionResolutionMultiplier"] : []),
      "scales",
    ],
    "positiveDefiniteTreatment",
  );
  const unitRoundoff = positive(
    pd.f32UnitRoundoff,
    "positiveDefiniteTreatment.f32UnitRoundoff",
  );
  literal(unitRoundoff, 2 ** -24, "positiveDefiniteTreatment.f32UnitRoundoff");
  const multiplier = positive(
    pd.eigenvalueFloorMultiplier,
    "positiveDefiniteTreatment.eigenvalueFloorMultiplier",
  );
  if (!Number.isSafeInteger(multiplier)) {
    throw new Error("eigenvalueFloorMultiplier must be an integer.");
  }
  literal(multiplier, 256, "positiveDefiniteTreatment.eigenvalueFloorMultiplier");
  const relativeFloor = positive(
    pd.relativeEigenvalueFloor,
    "positiveDefiniteTreatment.relativeEigenvalueFloor",
  );
  if (relativeFloor !== multiplier * unitRoundoff) {
    throw new Error(
      "positiveDefiniteTreatment.relativeEigenvalueFloor must equal " +
        "eigenvalueFloorMultiplier * f32UnitRoundoff.",
    );
  }
  literal(
    positive(
      pd.maximumNormalizedShift,
      "positiveDefiniteTreatment.maximumNormalizedShift",
    ),
    1e-3,
    "positiveDefiniteTreatment.maximumNormalizedShift",
  );
  if (isV2) {
    literal(
      positive(
        pd.positionResolutionMultiplier,
        "positiveDefiniteTreatment.positionResolutionMultiplier",
      ),
      8,
      "positiveDefiniteTreatment.positionResolutionMultiplier",
    );
  }
  if (!Array.isArray(pd.scales) || pd.scales.length !== 3) {
    throw new Error("positiveDefiniteTreatment.scales must contain three values.");
  }
  const scales = pd.scales;
  [1e-12, 1, 1e12].forEach((expected, index) =>
    literal(
      positive(scales[index], `positiveDefiniteTreatment.scales[${index}]`),
      expected,
      `positiveDefiniteTreatment.scales[${index}]`,
    ),
  );

  const lineSearch = record(root.lineSearch, "lineSearch");
  exactKeys(
    lineSearch,
    [
      "armijoC1",
      "backtrackFactor",
      "maximumBacktracks",
      "determinantFloor",
      "feasibilityBeforeEnergy",
      ...(isV2 ? ["energyComparison"] : []),
      "failureUpdate",
    ],
    "lineSearch",
  );
  const armijoC1 = positive(lineSearch.armijoC1, "lineSearch.armijoC1");
  literal(armijoC1, 1e-4, "lineSearch.armijoC1");
  if (!(armijoC1 < 1)) throw new Error("lineSearch.armijoC1 must be below one.");
  const factor = positive(lineSearch.backtrackFactor, "lineSearch.backtrackFactor");
  if (!(factor < 1)) throw new Error("lineSearch.backtrackFactor must be below one.");
  literal(factor, 0.5, "lineSearch.backtrackFactor");
  const maximumBacktracks = finite(lineSearch.maximumBacktracks, "lineSearch.maximumBacktracks");
  if (!Number.isSafeInteger(maximumBacktracks) || maximumBacktracks < 0) {
    throw new Error("lineSearch.maximumBacktracks must be a nonnegative integer.");
  }
  literal(maximumBacktracks, 12, "lineSearch.maximumBacktracks");
  const determinantFloor = positive(
    lineSearch.determinantFloor,
    "lineSearch.determinantFloor",
  );
  literal(determinantFloor, 1e-4, "lineSearch.determinantFloor");
  literal(lineSearch.feasibilityBeforeEnergy, true, "lineSearch.feasibilityBeforeEnergy");
  if (isV2) {
    literal(
      lineSearch.energyComparison,
      "delta-from-frozen-source",
      "lineSearch.energyComparison",
    );
  }
  literal(lineSearch.failureUpdate, "zero", "lineSearch.failureUpdate");

  const convergence = record(root.convergence, "convergence");
  exactKeys(
    convergence,
    [
      "tinyReferenceResidualTolerance",
      "maximumRuntimeTolerance",
      "requiresResidualAndUpdate",
      "sceneScale",
      ...(isV2 ? ["gpuHistoryCapacity"] : []),
    ],
    "convergence",
  );
  const tinyTolerance = positive(
    convergence.tinyReferenceResidualTolerance,
    "convergence.tinyReferenceResidualTolerance",
  );
  const runtimeTolerance = positive(
    convergence.maximumRuntimeTolerance,
    "convergence.maximumRuntimeTolerance",
  );
  literal(tinyTolerance, 1e-5, "convergence.tinyReferenceResidualTolerance");
  literal(runtimeTolerance, 1e-3, "convergence.maximumRuntimeTolerance");
  if (!(tinyTolerance <= runtimeTolerance && runtimeTolerance <= 1e-3)) {
    throw new Error("Convergence tolerances violate the frozen ordering.");
  }
  literal(convergence.requiresResidualAndUpdate, true, "convergence.requiresResidualAndUpdate");
  literal(convergence.sceneScale, "initial-dynamic-aabb-diagonal", "convergence.sceneScale");
  if (isV2) {
    literal(
      positive(
        convergence.gpuHistoryCapacity,
        "convergence.gpuHistoryCapacity",
      ),
      64,
      "convergence.gpuHistoryCapacity",
    );
  }

  const gates = record(root.referenceGates, "referenceGates");
  exactKeys(
    gates,
    [
      "restrictedGradientRelativeError",
      "restrictedHessianRelativeError",
      "shiftedLinearResidual",
      ...(isV2 ? ["gpuShiftedLinearResidual"] : []),
      "canonicalPackedLocalSystemCount",
      "assembledRevertDeterminant",
    ],
    "referenceGates",
  );
  literal(
    positive(
      gates.restrictedGradientRelativeError,
      "referenceGates.restrictedGradientRelativeError",
    ),
    2e-6,
    "referenceGates.restrictedGradientRelativeError",
  );
  literal(
    positive(
      gates.restrictedHessianRelativeError,
      "referenceGates.restrictedHessianRelativeError",
    ),
    2e-4,
    "referenceGates.restrictedHessianRelativeError",
  );
  literal(
    positive(gates.shiftedLinearResidual, "referenceGates.shiftedLinearResidual"),
    1e-8,
    "referenceGates.shiftedLinearResidual",
  );
  if (isV2) {
    literal(
      positive(
        gates.gpuShiftedLinearResidual,
        "referenceGates.gpuShiftedLinearResidual",
      ),
      2e-5,
      "referenceGates.gpuShiftedLinearResidual",
    );
  }
  const systemCount = positive(
    gates.canonicalPackedLocalSystemCount,
    "referenceGates.canonicalPackedLocalSystemCount",
  );
  if (!Number.isSafeInteger(systemCount)) {
    throw new Error("canonicalPackedLocalSystemCount must be an integer.");
  }
  literal(systemCount, 240, "referenceGates.canonicalPackedLocalSystemCount");
  const revertDeterminant = finite(
    gates.assembledRevertDeterminant,
    "referenceGates.assembledRevertDeterminant",
  );
  literal(
    revertDeterminant,
    -0.2100000000000002,
    "referenceGates.assembledRevertDeterminant",
  );
  if (!(revertDeterminant <= determinantFloor)) {
    throw new Error("The assembled revert fixture must violate the determinant floor.");
  }

  if (isV2) {
    const gpuRuntime = record(root.gpuRuntime, "gpuRuntime");
    exactKeys(
      gpuRuntime,
      [
        "enabledMaterial",
        "mixedMaterialSolve",
        "uniformBytes",
        "localDiagnosticVec4s",
        "tetDiagnosticVec4s",
        "convergenceComponentVec4s",
        "controlVec4s",
        "historyVec4s",
        "historyCapacity",
        "steppingReadback",
        "initializationReadback",
        "explicitDiagnosticReadback",
        "assembledEnergyAcceptance",
        "pinnedTargetApplication",
      ],
      "gpuRuntime",
    );
    literal(
      gpuRuntime.enabledMaterial,
      "stable-neo-hookean",
      "gpuRuntime.enabledMaterial",
    );
    literal(
      gpuRuntime.mixedMaterialSolve,
      "rejected",
      "gpuRuntime.mixedMaterialSolve",
    );
    literal(
      positive(gpuRuntime.uniformBytes, "gpuRuntime.uniformBytes"),
      176,
      "gpuRuntime.uniformBytes",
    );
    literal(
      positive(
        gpuRuntime.localDiagnosticVec4s,
        "gpuRuntime.localDiagnosticVec4s",
      ),
      4,
      "gpuRuntime.localDiagnosticVec4s",
    );
    literal(
      positive(
        gpuRuntime.tetDiagnosticVec4s,
        "gpuRuntime.tetDiagnosticVec4s",
      ),
      2,
      "gpuRuntime.tetDiagnosticVec4s",
    );
    literal(
      positive(
        gpuRuntime.convergenceComponentVec4s,
        "gpuRuntime.convergenceComponentVec4s",
      ),
      5,
      "gpuRuntime.convergenceComponentVec4s",
    );
    literal(
      positive(gpuRuntime.controlVec4s, "gpuRuntime.controlVec4s"),
      4,
      "gpuRuntime.controlVec4s",
    );
    literal(
      positive(gpuRuntime.historyVec4s, "gpuRuntime.historyVec4s"),
      8,
      "gpuRuntime.historyVec4s",
    );
    literal(
      positive(gpuRuntime.historyCapacity, "gpuRuntime.historyCapacity"),
      64,
      "gpuRuntime.historyCapacity",
    );
    literal(
      gpuRuntime.steppingReadback,
      "none",
      "gpuRuntime.steppingReadback",
    );
    literal(
      gpuRuntime.initializationReadback,
      "gpu-source-feasibility-once",
      "gpuRuntime.initializationReadback",
    );
    literal(
      gpuRuntime.explicitDiagnosticReadback,
      "on-request",
      "gpuRuntime.explicitDiagnosticReadback",
    );
    literal(
      gpuRuntime.assembledEnergyAcceptance,
      "diagnostic-only",
      "gpuRuntime.assembledEnergyAcceptance",
    );
    literal(
      gpuRuntime.pinnedTargetApplication,
      "assembled-feasibility-gated",
      "gpuRuntime.pinnedTargetApplication",
    );
  }

  if (!Array.isArray(root.caseIds) || root.caseIds.length === 0) {
    throw new Error("caseIds must be a nonempty array.");
  }
  const caseIds = root.caseIds.map((entry, index) =>
    stableId(entry, `caseIds[${index}]`),
  );
  if (new Set(caseIds).size !== caseIds.length) {
    throw new Error("caseIds must be unique.");
  }
  if (
    caseIds.length !== EXPECTED_CASE_IDS.length ||
    caseIds.some((entry, index) => entry !== EXPECTED_CASE_IDS[index])
  ) {
    throw new Error("caseIds must exactly match the frozen case inventory.");
  }

  const expectedRestrictedObjectiveTerms = isV2
    ? EXPECTED_V2_RESTRICTED_OBJECTIVE_TERMS
    : EXPECTED_V1_RESTRICTED_OBJECTIVE_TERMS;
  if (
    !Array.isArray(root.restrictedObjectiveTerms) ||
    root.restrictedObjectiveTerms.length !==
      expectedRestrictedObjectiveTerms.length ||
    root.restrictedObjectiveTerms.some(
      (entry, index) => entry !== expectedRestrictedObjectiveTerms[index],
    )
  ) {
    throw new Error(
      "restrictedObjectiveTerms must exactly describe the partial reference scope.",
    );
  }

  if (isV2) {
    if (
      !Array.isArray(root.gpuTestSelectors) ||
      root.gpuTestSelectors.length !== EXPECTED_GPU_TEST_SELECTORS.length
    ) {
      throw new Error(
        "gpuTestSelectors must exactly match the frozen Phase 1 GPU selector inventory.",
      );
    }
    root.gpuTestSelectors.forEach((value, index) => {
      const selector = record(value, `gpuTestSelectors[${index}]`);
      exactKeys(selector, ["source", "selector"], `gpuTestSelectors[${index}]`);
      const expected = EXPECTED_GPU_TEST_SELECTORS[index]!;
      literal(
        selector.source,
        expected.source,
        `gpuTestSelectors[${index}].source`,
      );
      literal(
        selector.selector,
        expected.selector,
        `gpuTestSelectors[${index}].selector`,
      );
    });
  }

  const status = record(root.runtimeStatus, "runtimeStatus");
  exactKeys(
    status,
    [
      "cpuMaterialInertiaReference",
      "compositeForcesAndTargets",
      "gpuProduction",
      "qualifiesPhase1Exit",
    ],
    "runtimeStatus",
  );
  literal(
    status.cpuMaterialInertiaReference,
    "implemented",
    "runtimeStatus.cpuMaterialInertiaReference",
  );
  literal(
    status.compositeForcesAndTargets,
    "pending",
    "runtimeStatus.compositeForcesAndTargets",
  );
  literal(
    status.gpuProduction,
    isV2 ? "implemented-material-inertia-floor" : "pending",
    "runtimeStatus.gpuProduction",
  );
  literal(status.qualifiesPhase1Exit, false, "runtimeStatus.qualifiesPhase1Exit");
  return value as Phase1GlobalizationManifest;
}
