export const PHASE1_CUBATURE_MANIFEST_SCHEMA =
  "org.jgs2.phase1-cubature" as const;
export const PHASE1_CUBATURE_MANIFEST_VERSION = 1 as const;

export interface Phase1CubatureManifest {
  readonly schema: typeof PHASE1_CUBATURE_MANIFEST_SCHEMA;
  readonly schemaVersion: typeof PHASE1_CUBATURE_MANIFEST_VERSION;
  readonly corpusVersion: string;
  readonly fixture: {
    readonly id: string;
    readonly generator: {
      readonly id: string;
      readonly version: string;
      readonly parameters: {
        readonly positions: readonly number[];
        readonly tetrahedra: readonly number[];
        readonly fixed: readonly number[];
        readonly materialIds: readonly number[];
        readonly bodyIds: readonly number[];
      };
    };
    readonly material: Readonly<Record<string, unknown>>;
    readonly simulation: Readonly<Record<string, unknown>>;
  };
  readonly basis: Readonly<Record<string, unknown>>;
  readonly training: Readonly<Record<string, unknown>>;
  readonly validation: Readonly<Record<string, unknown>>;
  readonly cubature: Readonly<Record<string, unknown>>;
  readonly updateValidation: Readonly<Record<string, unknown>>;
  readonly anchors: {
    readonly firstTraining: Phase1CubatureAnchor;
    readonly lastValidation: Phase1CubatureAnchor;
  };
  readonly expectedPackedSelections: readonly Phase1PackedCubatureSelection[];
}

export interface Phase1CubatureAnchor {
  readonly id: string;
  readonly minimumDeformationDeterminant: number;
  readonly positions: readonly number[];
  readonly predictedPositions: readonly number[];
}

export interface Phase1PackedCubatureSelection {
  readonly vertex: number;
  readonly tetrahedra: readonly number[];
  readonly weights: readonly number[];
  readonly nonzeroCandidateCount: number;
  readonly trainingColumnRank: number;
  readonly packedTrainingColumnRank: number;
  readonly packedResidual: number;
}

export function validatePhase1CubatureManifest(
  value: unknown,
): Phase1CubatureManifest {
  const root = record(value, "Phase 1 Cubature manifest");
  literal(root.schema, PHASE1_CUBATURE_MANIFEST_SCHEMA, "schema");
  literal(root.schemaVersion, PHASE1_CUBATURE_MANIFEST_VERSION, "schemaVersion");
  nonempty(root.corpusVersion, "corpusVersion");

  const fixture = record(root.fixture, "fixture");
  stableId(fixture.id, "fixture.id");
  const generator = record(fixture.generator, "fixture.generator");
  stableId(generator.id, "fixture.generator.id");
  nonempty(generator.version, "fixture.generator.version");
  const parameters = record(
    generator.parameters,
    "fixture.generator.parameters",
  );
  const positions = finiteArray(
    parameters.positions,
    undefined,
    "fixture.generator.parameters.positions",
  );
  if (positions.length === 0 || positions.length % 3 !== 0) {
    throw new Error("fixture positions must contain complete xyz vertices.");
  }
  const vertexCount = positions.length / 3;
  const tetrahedra = integerArray(
    parameters.tetrahedra,
    undefined,
    "fixture.generator.parameters.tetrahedra",
  );
  if (tetrahedra.length === 0 || tetrahedra.length % 4 !== 0) {
    throw new Error("fixture tetrahedra must contain complete index groups.");
  }
  for (const vertex of tetrahedra) {
    if (vertex >= vertexCount) {
      throw new Error("fixture tetrahedra reference an unknown vertex.");
    }
  }
  const tetrahedronCount = tetrahedra.length / 4;
  const fixed = integerArray(
    parameters.fixed,
    vertexCount,
    "fixture.generator.parameters.fixed",
  );
  if (fixed.some((entry) => entry !== 0 && entry !== 1)) {
    throw new Error("fixture fixed entries must be zero or one.");
  }
  integerArray(
    parameters.materialIds,
    tetrahedronCount,
    "fixture.generator.parameters.materialIds",
  );
  integerArray(
    parameters.bodyIds,
    vertexCount,
    "fixture.generator.parameters.bodyIds",
  );

  const material = record(fixture.material, "fixture.material");
  stableId(material.id, "fixture.material.id");
  literal(material.model, "stable-neo-hookean", "fixture.material.model");
  positive(material.density, "fixture.material.density");
  positive(material.youngModulus, "fixture.material.youngModulus");
  const poisson = finite(material.poissonRatio, "fixture.material.poissonRatio");
  if (poisson < 0 || poisson >= 0.5) {
    throw new Error("fixture.material.poissonRatio must be in [0, 0.5).");
  }
  const simulation = record(fixture.simulation, "fixture.simulation");
  positive(simulation.timestep, "fixture.simulation.timestep");
  finiteArray(simulation.gravity, 3, "fixture.simulation.gravity");
  finite(simulation.floorY, "fixture.simulation.floorY");
  positiveInteger(
    simulation.solverIterations,
    "fixture.simulation.solverIterations",
  );

  const basis = record(root.basis, "basis");
  positiveInteger(basis.modeCount, "basis.modeCount");
  positiveInteger(
    basis.validationOnlyModeCount,
    "basis.validationOnlyModeCount",
  );
  positiveInteger(basis.inverseIterations, "basis.inverseIterations");
  literal(
    basis.coRotation,
    "R_v * Ubar_vi * transpose(R_i)",
    "basis.coRotation",
  );
  literal(
    basis.vertexFrame,
    "polar(rest-volume-weighted-average-F)",
    "basis.vertexFrame",
  );

  const training = record(root.training, "training");
  const trainingCount = positiveInteger(training.count, "training.count");
  const signs = finiteArray(training.signsPerMode, 2, "training.signsPerMode");
  if (signs[0] !== 1 || signs[1] !== -1) {
    throw new Error("training.signsPerMode must freeze +1 and -1.");
  }
  if (trainingCount !== (basis.modeCount as number) * signs.length) {
    throw new Error("training.count must equal modeCount times sign count.");
  }
  positive(training.amplitude, "training.amplitude");
  literal(
    training.normalization,
    "max(max-tet-displacement-gradient-frobenius,max-vertex-displacement-over-scene-scale)",
    "training.normalization",
  );
  literal(
    training.targetNormalization,
    "per-pose-l2-complementary-gradient",
    "training.targetNormalization",
  );
  literal(training.validCount, trainingCount, "training.validCount");
  literal(training.trivialCount, 0, "training.trivialCount");
  literal(training.predictedBlend, 0, "training.predictedBlend");

  const validation = record(root.validation, "validation");
  const validationCount = positiveInteger(validation.count, "validation.count");
  literal(validation.mixtureCount, validationCount, "validation.mixtureCount");
  positive(validation.lowAmplitude, "validation.lowAmplitude");
  positive(validation.highAmplitude, "validation.highAmplitude");
  const highAmplitudeStart = nonnegativeInteger(
    validation.highAmplitudeStart,
    "validation.highAmplitudeStart",
  );
  if (highAmplitudeStart >= validationCount) {
    throw new Error("validation.highAmplitudeStart must be inside the corpus.");
  }
  const predictedBlend = finite(
    validation.predictedBlend,
    "validation.predictedBlend",
  );
  if (predictedBlend < 0 || predictedBlend > 1) {
    throw new Error("validation.predictedBlend must be in [0, 1].");
  }
  literal(
    validation.coefficientFormula,
    "sin((r+1)(k+1)0.754877666)+0.5cos((r+1)(k+3)0.569840291)",
    "validation.coefficientFormula",
  );
  nonnegativeInteger(
    validation.rotatedCasesWhenUnconstrained,
    "validation.rotatedCasesWhenUnconstrained",
  );
  positive(
    validation.minimumDeformationDeterminant,
    "validation.minimumDeformationDeterminant",
  );

  const cubature = record(root.cubature, "cubature");
  const maximumSamples = positiveInteger(
    cubature.maximumSamples,
    "cubature.maximumSamples",
  );
  literal(cubature.candidateCount, tetrahedronCount, "cubature.candidateCount");
  if (maximumSamples >= tetrahedronCount) {
    throw new Error("cubature.maximumSamples must be smaller than candidateCount.");
  }
  literal(
    cubature.requireRankAboveSampleBudget,
    true,
    "cubature.requireRankAboveSampleBudget",
  );
  const rankTolerance = positive(
    cubature.columnRankRelativeTolerance,
    "cubature.columnRankRelativeTolerance",
  );
  if (rankTolerance > 1e-3) {
    throw new Error("cubature.columnRankRelativeTolerance is too large.");
  }
  unitIntervalExclusiveZero(
    cubature.normalizedTrainingResidual,
    "cubature.normalizedTrainingResidual",
  );
  literal(cubature.nonnegativeWeights, true, "cubature.nonnegativeWeights");
  literal(cubature.runtimePacking, "f32", "cubature.runtimePacking");
  literal(
    cubature.selection,
    "deterministic-greedy-nnls-with-small-fixture-subset-audit",
    "cubature.selection",
  );
  literal(
    cubature.residualMetric,
    "l2(packed-columns*packed-weights-f64-target)/l2(f64-target)",
    "cubature.residualMetric",
  );

  const update = record(root.updateValidation, "updateValidation");
  unitIntervalExclusiveZero(
    update.trainingRelativeRms,
    "updateValidation.trainingRelativeRms",
  );
  unitIntervalExclusiveZero(
    update.validationRelativeRms,
    "updateValidation.validationRelativeRms",
  );
  unitIntervalExclusiveZero(
    update.combinedRelativeRms,
    "updateValidation.combinedRelativeRms",
  );
  unitIntervalExclusiveZero(
    update.productionGpuRelativeError,
    "updateValidation.productionGpuRelativeError",
  );
  literal(update.regularization, "none", "updateValidation.regularization");
  literal(
    update.metric,
    "sqrt(sum(l2(selected-update-exact-update)^2)/sum(l2(exact-update)^2))",
    "updateValidation.metric",
  );
  literal(
    update.exactReference,
    "independent-dense-current-projection",
    "updateValidation.exactReference",
  );
  literal(
    update.gpuPrediction,
    "production-f32-implicit-euler-predictor",
    "updateValidation.gpuPrediction",
  );
  literal(
    update.productionIterations,
    2,
    "updateValidation.productionIterations",
  );

  const anchors = record(root.anchors, "anchors");
  validateAnchor(
    anchors.firstTraining,
    "anchors.firstTraining",
    vertexCount * 3,
  );
  validateAnchor(
    anchors.lastValidation,
    "anchors.lastValidation",
    vertexCount * 3,
  );

  const selections = array(root.expectedPackedSelections, "expectedPackedSelections");
  const vertices = new Set<number>();
  for (const [index, entryValue] of selections.entries()) {
    const label = `expectedPackedSelections[${index}]`;
    const entry = record(entryValue, label);
    const vertex = nonnegativeInteger(entry.vertex, `${label}.vertex`);
    if (vertex >= vertexCount || vertices.has(vertex)) {
      throw new Error(`${label}.vertex must be unique and in range.`);
    }
    vertices.add(vertex);
    const selectedTetrahedra = integerArray(
      entry.tetrahedra,
      undefined,
      `${label}.tetrahedra`,
    );
    const weights = finiteArray(
      entry.weights,
      selectedTetrahedra.length,
      `${label}.weights`,
    );
    if (selectedTetrahedra.length !== maximumSamples) {
      throw new Error(`${label} violates the sample budget.`);
    }
    const ids = new Set<number>();
    for (const tetrahedron of selectedTetrahedra) {
      if (tetrahedron >= tetrahedronCount || ids.has(tetrahedron)) {
        throw new Error(`${label}.tetrahedra must be unique and in range.`);
      }
      ids.add(tetrahedron);
    }
    for (const weight of weights) {
      if (!(weight > 0) || Math.fround(weight) !== weight) {
        throw new Error(`${label}.weights must be positive f32 values.`);
      }
    }
    literal(
      entry.nonzeroCandidateCount,
      tetrahedronCount,
      `${label}.nonzeroCandidateCount`,
    );
    const trainingColumnRank = positiveInteger(
      entry.trainingColumnRank,
      `${label}.trainingColumnRank`,
    );
    const packedTrainingColumnRank = positiveInteger(
      entry.packedTrainingColumnRank,
      `${label}.packedTrainingColumnRank`,
    );
    if (
      trainingColumnRank <= maximumSamples ||
      trainingColumnRank > tetrahedronCount ||
      packedTrainingColumnRank <= maximumSamples ||
      packedTrainingColumnRank > tetrahedronCount
    ) {
      throw new Error(`${label} must retain rank above the sample budget.`);
    }
    const packedResidual = positive(
      entry.packedResidual,
      `${label}.packedResidual`,
    );
    if (packedResidual > (cubature.normalizedTrainingResidual as number)) {
      throw new Error(`${label}.packedResidual exceeds the frozen gate.`);
    }
  }
  const expectedActiveVertices = fixed
    .map((entry, vertex) => ({ entry, vertex }))
    .filter(({ entry }) => entry === 0)
    .map(({ vertex }) => vertex);
  if (
    selections.length !== expectedActiveVertices.length ||
    expectedActiveVertices.some((vertex) => !vertices.has(vertex))
  ) {
    throw new Error(
      "expectedPackedSelections must cover every active fixture vertex.",
    );
  }
  return value as Phase1CubatureManifest;
}

function validateAnchor(value: unknown, label: string, length: number): void {
  const anchor = record(value, label);
  nonempty(anchor.id, `${label}.id`);
  positive(
    anchor.minimumDeformationDeterminant,
    `${label}.minimumDeformationDeterminant`,
  );
  finiteArray(anchor.positions, length, `${label}.positions`);
  finiteArray(anchor.predictedPositions, length, `${label}.predictedPositions`);
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): readonly unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value;
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

function unitIntervalExclusiveZero(value: unknown, label: string): number {
  const result = positive(value, label);
  if (result > 1) throw new Error(`${label} must not exceed one.`);
  return result;
}

function nonnegativeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${label} must be a nonnegative safe integer.`);
  }
  return value as number;
}

function positiveInteger(value: unknown, label: string): number {
  const result = nonnegativeInteger(value, label);
  if (result === 0) throw new Error(`${label} must be positive.`);
  return result;
}

function finiteArray(
  value: unknown,
  expectedLength: number | undefined,
  label: string,
): readonly number[] {
  const entries = array(value, label);
  if (expectedLength !== undefined && entries.length !== expectedLength) {
    throw new Error(`${label} must contain ${expectedLength} entries.`);
  }
  return entries.map((entry, index) => finite(entry, `${label}[${index}]`));
}

function integerArray(
  value: unknown,
  expectedLength: number | undefined,
  label: string,
): readonly number[] {
  const entries = array(value, label);
  if (expectedLength !== undefined && entries.length !== expectedLength) {
    throw new Error(`${label} must contain ${expectedLength} entries.`);
  }
  return entries.map((entry, index) =>
    nonnegativeInteger(entry, `${label}[${index}]`),
  );
}

function nonempty(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a nonempty string.`);
  }
  return value;
}

function stableId(value: unknown, label: string): string {
  const id = nonempty(value, label);
  if (!/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/.test(id)) {
    throw new Error(`${label} must be a stable lowercase ID.`);
  }
  return id;
}

function literal<T>(value: unknown, expected: T, label: string): void {
  if (value !== expected) {
    throw new Error(`${label} must equal ${String(expected)}.`);
  }
}
