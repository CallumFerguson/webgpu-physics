export const BASELINE_TEST_MANIFEST_SCHEMA =
  "org.jgs2.baseline-tests" as const;
export const BASELINE_TEST_MANIFEST_VERSION = 1 as const;
export const CANONICAL_SCENE_MANIFEST_SCHEMA =
  "org.jgs2.canonical-scenes" as const;
export const CANONICAL_SCENE_MANIFEST_VERSION = 1 as const;

export interface BaselineTestManifest {
  readonly schema: typeof BASELINE_TEST_MANIFEST_SCHEMA;
  readonly schemaVersion: typeof BASELINE_TEST_MANIFEST_VERSION;
  readonly frozenOn: string;
  readonly baselineDescription: string;
  readonly suites: readonly {
    readonly id: "unit" | "e2e";
    readonly command: string;
    readonly expectedTestCount: number;
    readonly zeroUnexpectedSkips: true;
  }[];
  readonly requiredCommands: readonly {
    readonly id: string;
    readonly command: string;
  }[];
  readonly expectedTests: readonly {
    readonly id: string;
    readonly suite: "unit" | "e2e";
    readonly source: string;
    readonly selector: string;
  }[];
  readonly allowedSkips: readonly {
    readonly testId: string;
    readonly condition: string;
    readonly reason: string;
  }[];
}

export type CanonicalFixtureKind =
  | "baseline-scene"
  | "force-free"
  | "oracle";

export interface CanonicalSceneManifest {
  readonly schema: typeof CANONICAL_SCENE_MANIFEST_SCHEMA;
  readonly schemaVersion: typeof CANONICAL_SCENE_MANIFEST_VERSION;
  readonly corpusVersion: string;
  readonly fixtures: readonly CanonicalFixture[];
  readonly corpora: readonly CanonicalCorpus[];
}

export interface CanonicalFixture {
  readonly id: string;
  readonly phase: 0;
  readonly kind: CanonicalFixtureKind;
  readonly purpose: string;
  readonly runtimeSceneId: string | null;
  readonly generator: {
    readonly id: string;
    readonly version: string;
    readonly seed: number;
    readonly parameters: Readonly<Record<string, unknown>>;
  };
  readonly materials: readonly CanonicalMaterial[];
  readonly simulation: {
    readonly timestep: number;
    readonly gravity: readonly [number, number, number];
    readonly floorY: number | null;
    readonly parityMode: boolean;
  };
  readonly solver: {
    readonly iterations: number;
    readonly schedule: "jacobi" | "graph-colored-gauss-seidel";
    readonly cubature: {
      readonly mode: "selected" | "exact-all-elements";
      readonly samplesPerSubproblem: 4 | 6 | null;
    };
  };
  readonly initialState: Readonly<Record<string, unknown>>;
  readonly sampledFrames: readonly CanonicalSampledFrame[];
  readonly camera: CanonicalCamera;
  readonly expectedMetrics: readonly CanonicalMetric[];
}

export interface CanonicalMaterial {
  readonly id: string;
  readonly density: number;
  readonly youngModulus: number;
  readonly poissonRatio: number;
  readonly color: readonly [number, number, number, number];
}

export interface CanonicalSampledFrame {
  readonly id: string;
  readonly frame: number;
  readonly simulatedSeconds: number;
  readonly artifacts: readonly ("diagnostics" | "screenshot")[];
}

export interface CanonicalCamera {
  readonly eye: readonly [number, number, number];
  readonly target: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  readonly fovYRadians: number;
}

export interface CanonicalMetric {
  readonly id: string;
  readonly operator: "<" | "<=" | ">" | ">=" | "equals";
  readonly value: number | boolean;
  readonly normalization: string;
}

export interface CanonicalCorpus {
  readonly id: string;
  readonly fixtureId: string;
  readonly generator: {
    readonly id: string;
    readonly version: string;
    readonly seed: number;
    readonly caseCount: number;
    readonly parameters: Readonly<Record<string, unknown>>;
  };
}

export function validateBaselineTestManifest(
  value: unknown,
): BaselineTestManifest {
  const root = requireRecord(value, "baseline test manifest");
  requireLiteral(
    root.schema,
    BASELINE_TEST_MANIFEST_SCHEMA,
    "baseline schema",
  );
  requireLiteral(
    root.schemaVersion,
    BASELINE_TEST_MANIFEST_VERSION,
    "baseline schemaVersion",
  );
  const frozenOn = requireNonemptyString(root.frozenOn, "frozenOn");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(frozenOn)) {
    throw new Error("frozenOn must use YYYY-MM-DD.");
  }
  requireNonemptyString(root.baselineDescription, "baselineDescription");

  const suites = requireArray(root.suites, "suites");
  const suiteIds = new Set<string>();
  const suiteCounts = new Map<string, number>();
  for (const [index, entry] of suites.entries()) {
    const suite = requireRecord(entry, `suites[${index}]`);
    const id = requireOneOf(suite.id, ["unit", "e2e"] as const, `suites[${index}].id`);
    requireUnique(suiteIds, id, "suite ID");
    requireNonemptyString(suite.command, `suites[${index}].command`);
    suiteCounts.set(
      id,
      requireNonnegativeSafeInteger(
        suite.expectedTestCount,
        `suites[${index}].expectedTestCount`,
      ),
    );
    requireLiteral(
      suite.zeroUnexpectedSkips,
      true,
      `suites[${index}].zeroUnexpectedSkips`,
    );
  }
  if (suiteIds.size !== 2 || !suiteIds.has("unit") || !suiteIds.has("e2e")) {
    throw new Error("suites must define exactly unit and e2e.");
  }

  const commands = requireArray(root.requiredCommands, "requiredCommands");
  const commandIds = new Set<string>();
  for (const [index, entry] of commands.entries()) {
    const command = requireRecord(entry, `requiredCommands[${index}]`);
    const id = requireNonemptyString(command.id, `requiredCommands[${index}].id`);
    requireUnique(commandIds, id, "required command ID");
    requireNonemptyString(
      command.command,
      `requiredCommands[${index}].command`,
    );
  }
  if (!commandIds.has("build")) {
    throw new Error("requiredCommands must include the production build.");
  }

  const expectedTests = requireArray(root.expectedTests, "expectedTests");
  const testIds = new Set<string>();
  const actualCounts = new Map<string, number>([
    ["unit", 0],
    ["e2e", 0],
  ]);
  for (const [index, entry] of expectedTests.entries()) {
    const test = requireRecord(entry, `expectedTests[${index}]`);
    const id = requireStableId(test.id, `expectedTests[${index}].id`);
    requireUnique(testIds, id, "test ID");
    const suite = requireOneOf(
      test.suite,
      ["unit", "e2e"] as const,
      `expectedTests[${index}].suite`,
    );
    actualCounts.set(suite, actualCounts.get(suite)! + 1);
    requireSourcePath(test.source, suite, `expectedTests[${index}].source`);
    requireNonemptyString(test.selector, `expectedTests[${index}].selector`);
  }
  for (const suite of ["unit", "e2e"] as const) {
    if (suiteCounts.get(suite) !== actualCounts.get(suite)) {
      throw new Error(
        `${suite} expectedTestCount does not match expectedTests entries.`,
      );
    }
  }

  const allowedSkips = requireArray(root.allowedSkips, "allowedSkips");
  const skipIds = new Set<string>();
  for (const [index, entry] of allowedSkips.entries()) {
    const skip = requireRecord(entry, `allowedSkips[${index}]`);
    const testId = requireStableId(
      skip.testId,
      `allowedSkips[${index}].testId`,
    );
    requireUnique(skipIds, testId, "allowed-skip test ID");
    if (!testIds.has(testId)) {
      throw new Error(`Allowed skip ${testId} is not an expected test.`);
    }
    requireNonemptyString(skip.condition, `allowedSkips[${index}].condition`);
    requireNonemptyString(skip.reason, `allowedSkips[${index}].reason`);
  }

  return value as BaselineTestManifest;
}

export function validateCanonicalSceneManifest(
  value: unknown,
): CanonicalSceneManifest {
  const root = requireRecord(value, "canonical scene manifest");
  requireLiteral(
    root.schema,
    CANONICAL_SCENE_MANIFEST_SCHEMA,
    "canonical schema",
  );
  requireLiteral(
    root.schemaVersion,
    CANONICAL_SCENE_MANIFEST_VERSION,
    "canonical schemaVersion",
  );
  requireNonemptyString(root.corpusVersion, "corpusVersion");

  const fixtures = requireArray(root.fixtures, "fixtures");
  const fixtureIds = new Set<string>();
  for (const [index, entry] of fixtures.entries()) {
    validateFixture(entry, index, fixtureIds);
  }
  if (fixtures.length === 0) {
    throw new Error("fixtures must not be empty.");
  }

  const corpora = requireArray(root.corpora, "corpora");
  const corpusIds = new Set<string>();
  for (const [index, entry] of corpora.entries()) {
    const corpus = requireRecord(entry, `corpora[${index}]`);
    const id = requireStableId(corpus.id, `corpora[${index}].id`);
    requireUnique(corpusIds, id, "corpus ID");
    const fixtureId = requireStableId(
      corpus.fixtureId,
      `corpora[${index}].fixtureId`,
    );
    if (!fixtureIds.has(fixtureId)) {
      throw new Error(`Corpus ${id} refers to unknown fixture ${fixtureId}.`);
    }
    const generator = requireRecord(
      corpus.generator,
      `corpora[${index}].generator`,
    );
    validateGenerator(generator, `corpora[${index}].generator`);
    requirePositiveSafeInteger(
      generator.caseCount,
      `corpora[${index}].generator.caseCount`,
    );
  }
  if (corpora.length === 0) {
    throw new Error("corpora must not be empty.");
  }

  return value as CanonicalSceneManifest;
}

function validateFixture(
  entry: unknown,
  index: number,
  fixtureIds: Set<string>,
): void {
  const label = `fixtures[${index}]`;
  const fixture = requireRecord(entry, label);
  const id = requireStableId(fixture.id, `${label}.id`);
  requireUnique(fixtureIds, id, "fixture ID");
  requireLiteral(fixture.phase, 0, `${label}.phase`);
  const kind = requireOneOf(
    fixture.kind,
    ["baseline-scene", "force-free", "oracle"] as const,
    `${label}.kind`,
  );
  requireNonemptyString(fixture.purpose, `${label}.purpose`);
  if (kind === "baseline-scene") {
    requireNonemptyString(fixture.runtimeSceneId, `${label}.runtimeSceneId`);
  } else if (fixture.runtimeSceneId !== null) {
    throw new Error(`${label}.runtimeSceneId must be null for ${kind}.`);
  }

  validateGenerator(requireRecord(fixture.generator, `${label}.generator`), `${label}.generator`);

  const materials = requireArray(fixture.materials, `${label}.materials`);
  const materialIds = new Set<string>();
  if (materials.length === 0) {
    throw new Error(`${label}.materials must not be empty.`);
  }
  for (const [materialIndex, entryValue] of materials.entries()) {
    const materialLabel = `${label}.materials[${materialIndex}]`;
    const material = requireRecord(entryValue, materialLabel);
    const materialId = requireStableId(material.id, `${materialLabel}.id`);
    requireUnique(materialIds, materialId, "material ID");
    requirePositiveFinite(material.density, `${materialLabel}.density`);
    requirePositiveFinite(
      material.youngModulus,
      `${materialLabel}.youngModulus`,
    );
    const poissonRatio = requireFiniteNumber(
      material.poissonRatio,
      `${materialLabel}.poissonRatio`,
    );
    if (poissonRatio <= -1 || poissonRatio >= 0.5) {
      throw new Error(`${materialLabel}.poissonRatio must be in (-1, 0.5).`);
    }
    requireFiniteTuple(material.color, 4, `${materialLabel}.color`);
  }

  const simulation = requireRecord(fixture.simulation, `${label}.simulation`);
  const timestep = requirePositiveFinite(
    simulation.timestep,
    `${label}.simulation.timestep`,
  );
  requireFiniteTuple(simulation.gravity, 3, `${label}.simulation.gravity`);
  if (simulation.floorY !== null) {
    requireFiniteNumber(simulation.floorY, `${label}.simulation.floorY`);
  }
  requireBoolean(simulation.parityMode, `${label}.simulation.parityMode`);

  const solver = requireRecord(fixture.solver, `${label}.solver`);
  requirePositiveSafeInteger(solver.iterations, `${label}.solver.iterations`);
  requireOneOf(
    solver.schedule,
    ["jacobi", "graph-colored-gauss-seidel"] as const,
    `${label}.solver.schedule`,
  );
  const cubature = requireRecord(solver.cubature, `${label}.solver.cubature`);
  const cubatureMode = requireOneOf(
    cubature.mode,
    ["selected", "exact-all-elements"] as const,
    `${label}.solver.cubature.mode`,
  );
  if (cubatureMode === "selected") {
    requireOneOf(
      cubature.samplesPerSubproblem,
      [4, 6] as const,
      `${label}.solver.cubature.samplesPerSubproblem`,
    );
  } else if (cubature.samplesPerSubproblem !== null) {
    throw new Error(
      `${label}.solver.cubature.samplesPerSubproblem must be null in exact mode.`,
    );
  }

  requireRecord(fixture.initialState, `${label}.initialState`);
  const sampledFrames = requireArray(
    fixture.sampledFrames,
    `${label}.sampledFrames`,
  );
  const frameIds = new Set<string>();
  let previousFrame = -1;
  for (const [frameIndex, frameValue] of sampledFrames.entries()) {
    const frameLabel = `${label}.sampledFrames[${frameIndex}]`;
    const frame = requireRecord(frameValue, frameLabel);
    const frameId = requireStableId(frame.id, `${frameLabel}.id`);
    requireUnique(frameIds, frameId, "sampled-frame ID");
    const frameNumber = requireNonnegativeSafeInteger(
      frame.frame,
      `${frameLabel}.frame`,
    );
    if (frameNumber <= previousFrame) {
      throw new Error(`${label}.sampledFrames must be strictly frame-sorted.`);
    }
    previousFrame = frameNumber;
    const seconds = requireNonnegativeFinite(
      frame.simulatedSeconds,
      `${frameLabel}.simulatedSeconds`,
    );
    if (Math.abs(seconds - frameNumber * timestep) > 1e-10) {
      throw new Error(
        `${frameLabel}.simulatedSeconds does not equal frame * timestep.`,
      );
    }
    const artifacts = requireArray(frame.artifacts, `${frameLabel}.artifacts`);
    if (artifacts.length === 0) {
      throw new Error(`${frameLabel}.artifacts must not be empty.`);
    }
    for (const artifact of artifacts) {
      requireOneOf(
        artifact,
        ["diagnostics", "screenshot"] as const,
        `${frameLabel}.artifacts entry`,
      );
    }
  }
  if (sampledFrames.length === 0 || previousFrame < 0) {
    throw new Error(`${label}.sampledFrames must not be empty.`);
  }
  const firstFrame = requireRecord(sampledFrames[0], `${label}.sampledFrames[0]`);
  if (firstFrame.frame !== 0) {
    throw new Error(`${label}.sampledFrames must begin at frame 0.`);
  }

  const camera = requireRecord(fixture.camera, `${label}.camera`);
  requireFiniteTuple(camera.eye, 3, `${label}.camera.eye`);
  requireFiniteTuple(camera.target, 3, `${label}.camera.target`);
  requireFiniteTuple(camera.up, 3, `${label}.camera.up`);
  requirePositiveFinite(camera.fovYRadians, `${label}.camera.fovYRadians`);

  const metrics = requireArray(fixture.expectedMetrics, `${label}.expectedMetrics`);
  const metricIds = new Set<string>();
  if (metrics.length === 0) {
    throw new Error(`${label}.expectedMetrics must not be empty.`);
  }
  for (const [metricIndex, metricValue] of metrics.entries()) {
    const metricLabel = `${label}.expectedMetrics[${metricIndex}]`;
    const metric = requireRecord(metricValue, metricLabel);
    const metricId = requireStableId(metric.id, `${metricLabel}.id`);
    requireUnique(metricIds, metricId, "metric ID");
    requireOneOf(
      metric.operator,
      ["<", "<=", ">", ">=", "equals"] as const,
      `${metricLabel}.operator`,
    );
    if (typeof metric.value === "number") {
      requireFiniteNumber(metric.value, `${metricLabel}.value`);
    } else {
      requireBoolean(metric.value, `${metricLabel}.value`);
    }
    requireNonemptyString(metric.normalization, `${metricLabel}.normalization`);
  }
}

function validateGenerator(
  generator: Record<string, unknown>,
  label: string,
): void {
  requireStableId(generator.id, `${label}.id`);
  requireNonemptyString(generator.version, `${label}.version`);
  requireNonnegativeSafeInteger(generator.seed, `${label}.seed`);
  requireRecord(generator.parameters, `${label}.parameters`);
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, label: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array.`);
  }
  return value;
}

function requireNonemptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a nonempty string.`);
  }
  return value;
}

function requireStableId(value: unknown, label: string): string {
  const id = requireNonemptyString(value, label);
  if (!/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/.test(id)) {
    throw new Error(`${label} must be a stable lowercase ID.`);
  }
  return id;
}

function requireSourcePath(
  value: unknown,
  suite: "unit" | "e2e",
  label: string,
): void {
  const source = requireNonemptyString(value, label);
  const expectedSuffix = suite === "unit" ? ".test.ts" : ".spec.ts";
  if (!source.endsWith(expectedSuffix) || source.includes("\\")) {
    throw new Error(
      `${label} must be a repository-relative POSIX ${expectedSuffix} path.`,
    );
  }
}

function requireUnique(values: Set<string>, value: string, label: string): void {
  if (values.has(value)) {
    throw new Error(`Duplicate ${label}: ${value}.`);
  }
  values.add(value);
}

function requireLiteral<T extends string | number | boolean>(
  value: unknown,
  expected: T,
  label: string,
): T {
  if (value !== expected) {
    throw new Error(`${label} must equal ${String(expected)}.`);
  }
  return expected;
}

function requireOneOf<const T extends readonly (string | number)[]>(
  value: unknown,
  expected: T,
  label: string,
): T[number] {
  if (!expected.includes(value as never)) {
    throw new Error(`${label} must be one of ${expected.join(", ")}.`);
  }
  return value as T[number];
}

function requireFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function requirePositiveFinite(value: unknown, label: string): number {
  const number = requireFiniteNumber(value, label);
  if (number <= 0) {
    throw new Error(`${label} must be positive.`);
  }
  return number;
}

function requireNonnegativeFinite(value: unknown, label: string): number {
  const number = requireFiniteNumber(value, label);
  if (number < 0) {
    throw new Error(`${label} must be nonnegative.`);
  }
  return number;
}

function requireNonnegativeSafeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${label} must be a nonnegative safe integer.`);
  }
  return value as number;
}

function requirePositiveSafeInteger(value: unknown, label: string): number {
  const number = requireNonnegativeSafeInteger(value, label);
  if (number === 0) {
    throw new Error(`${label} must be positive.`);
  }
  return number;
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be boolean.`);
  }
  return value;
}

function requireFiniteTuple(
  value: unknown,
  length: number,
  label: string,
): void {
  const entries = requireArray(value, label);
  if (entries.length !== length) {
    throw new Error(`${label} must contain ${length} entries.`);
  }
  for (let index = 0; index < length; index += 1) {
    requireFiniteNumber(entries[index], `${label}[${index}]`);
  }
}
