export const PHASE1_GLOBALIZATION_MANIFEST_SCHEMA =
  "org.jgs2.phase1-globalization" as const;
export const PHASE1_GLOBALIZATION_MANIFEST_VERSION = 1 as const;

const EXPECTED_SOURCE_FIXTURES = Object.freeze({
  materialManifest: "manifests/phase1-scenes.v1.json",
  cubatureManifest: "manifests/phase1-cubature.v1.json",
  cubatureFixtureId: "phase1.nonlinear-cubature-beam",
});
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
const EXPECTED_RESTRICTED_OBJECTIVE_TERMS = Object.freeze([
  "implicit-euler-inertia",
  "stable-neo-hookean-material",
] as const);

export interface Phase1GlobalizationManifest {
  readonly schema: typeof PHASE1_GLOBALIZATION_MANIFEST_SCHEMA;
  readonly schemaVersion: typeof PHASE1_GLOBALIZATION_MANIFEST_VERSION;
  readonly id: string;
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
  readonly caseIds: readonly string[];
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
      "caseIds",
      "restrictedObjectiveTerms",
      "runtimeStatus",
    ],
    "Phase 1 globalization manifest",
  );
  literal(root.schema, PHASE1_GLOBALIZATION_MANIFEST_SCHEMA, "schema");
  literal(
    root.schemaVersion,
    PHASE1_GLOBALIZATION_MANIFEST_VERSION,
    "schemaVersion",
  );
  literal(
    root.id,
    "phase1.globalization-material-inertia-reference",
    "id",
  );

  const sources = record(root.sourceFixtures, "sourceFixtures");
  exactKeys(sources, Object.keys(EXPECTED_SOURCE_FIXTURES), "sourceFixtures");
  for (const [key, expected] of Object.entries(EXPECTED_SOURCE_FIXTURES)) {
    literal(sources[key], expected, `sourceFixtures.${key}`);
  }

  const pd = record(root.positiveDefiniteTreatment, "positiveDefiniteTreatment");
  exactKeys(
    pd,
    [
      "f32UnitRoundoff",
      "eigenvalueFloorMultiplier",
      "relativeEigenvalueFloor",
      "maximumNormalizedShift",
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
  literal(lineSearch.failureUpdate, "zero", "lineSearch.failureUpdate");

  const convergence = record(root.convergence, "convergence");
  exactKeys(
    convergence,
    [
      "tinyReferenceResidualTolerance",
      "maximumRuntimeTolerance",
      "requiresResidualAndUpdate",
      "sceneScale",
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

  const gates = record(root.referenceGates, "referenceGates");
  exactKeys(
    gates,
    [
      "restrictedGradientRelativeError",
      "restrictedHessianRelativeError",
      "shiftedLinearResidual",
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

  if (
    !Array.isArray(root.restrictedObjectiveTerms) ||
    root.restrictedObjectiveTerms.length !==
      EXPECTED_RESTRICTED_OBJECTIVE_TERMS.length ||
    root.restrictedObjectiveTerms.some(
      (entry, index) => entry !== EXPECTED_RESTRICTED_OBJECTIVE_TERMS[index],
    )
  ) {
    throw new Error(
      "restrictedObjectiveTerms must exactly describe the partial reference scope.",
    );
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
  literal(status.gpuProduction, "pending", "runtimeStatus.gpuProduction");
  literal(status.qualifiesPhase1Exit, false, "runtimeStatus.qualifiesPhase1Exit");
  return value as Phase1GlobalizationManifest;
}
