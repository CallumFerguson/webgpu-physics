export const PHASE1_SCENE_MANIFEST_SCHEMA =
  "org.jgs2.phase1-scenes" as const;
export const PHASE1_SCENE_MANIFEST_VERSION = 1 as const;

export type Phase1OraclePoseKind =
  | "rest"
  | "rigid"
  | "compression"
  | "shear"
  | "stretch"
  | "seeded";

export interface Phase1SceneManifest {
  readonly schema: typeof PHASE1_SCENE_MANIFEST_SCHEMA;
  readonly schemaVersion: typeof PHASE1_SCENE_MANIFEST_VERSION;
  readonly corpusVersion: string;
  readonly fixtures: readonly Phase1CanonicalFixture[];
  readonly corpora: readonly Phase1CanonicalCorpus[];
}

export interface Phase1CanonicalFixture {
  readonly id: string;
  readonly phase: 1;
  readonly kind: "stable-neo-hookean-oracle";
  readonly purpose: string;
  readonly runtimeSceneId: null;
  readonly generator: Phase1Generator;
  readonly material: {
    readonly id: string;
    readonly model: "stable-neo-hookean";
    readonly density: number;
    readonly youngModulus: number;
    readonly poissonRatio: number;
    readonly color: readonly [number, number, number, number];
  };
  readonly simulation: {
    readonly timestep: number;
    readonly gravity: readonly [number, number, number];
    readonly floorY: number;
    readonly floorStiffness: number;
    readonly parityMode: true;
  };
  readonly solver: {
    readonly iterations: number;
    readonly schedule: "jacobi";
    readonly cubature: {
      readonly mode: "exact-all-elements";
      readonly packedSamplesPerSubproblem: number;
    };
  };
  readonly expectedTolerances: {
    readonly cpu: Readonly<Record<string, number>>;
    readonly gpu: Readonly<Record<string, number>>;
    readonly acceptedPoseMinimumDeterminant: number;
  };
}

export interface Phase1Generator {
  readonly id: string;
  readonly version: string;
  readonly seed: number;
  readonly parameters: Readonly<Record<string, unknown>>;
}

export interface Phase1AnchorState {
  readonly id: string;
  readonly index: number;
  readonly kind: Phase1OraclePoseKind;
  readonly determinant: number;
  readonly deformationGradient: readonly number[];
  readonly positions: readonly number[];
}

export interface Phase1CanonicalCorpus {
  readonly id: string;
  readonly fixtureId: string;
  readonly generator: Phase1Generator & { readonly caseCount: number };
  readonly requiredCaseLayout: readonly {
    readonly index: number;
    readonly kind: Exclude<Phase1OraclePoseKind, "seeded">;
    readonly determinant: number;
  }[];
  readonly anchors: {
    readonly first: Phase1AnchorState;
    readonly last: Phase1AnchorState;
  };
}

export function validatePhase1SceneManifest(
  value: unknown,
): Phase1SceneManifest {
  const root = record(value, "Phase 1 scene manifest");
  literal(root.schema, PHASE1_SCENE_MANIFEST_SCHEMA, "schema");
  literal(root.schemaVersion, PHASE1_SCENE_MANIFEST_VERSION, "schemaVersion");
  nonempty(root.corpusVersion, "corpusVersion");

  const fixtures = array(root.fixtures, "fixtures");
  if (fixtures.length === 0) throw new Error("fixtures must not be empty.");
  const fixtureIds = new Set<string>();
  for (const [index, value] of fixtures.entries()) {
    const label = `fixtures[${index}]`;
    const fixture = record(value, label);
    const id = stableId(fixture.id, `${label}.id`);
    unique(fixtureIds, id, "fixture ID");
    literal(fixture.phase, 1, `${label}.phase`);
    literal(
      fixture.kind,
      "stable-neo-hookean-oracle",
      `${label}.kind`,
    );
    nonempty(fixture.purpose, `${label}.purpose`);
    literal(fixture.runtimeSceneId, null, `${label}.runtimeSceneId`);
    validateGenerator(fixture.generator, `${label}.generator`, false);

    const material = record(fixture.material, `${label}.material`);
    stableId(material.id, `${label}.material.id`);
    literal(
      material.model,
      "stable-neo-hookean",
      `${label}.material.model`,
    );
    positive(material.density, `${label}.material.density`);
    positive(material.youngModulus, `${label}.material.youngModulus`);
    const poisson = finite(
      material.poissonRatio,
      `${label}.material.poissonRatio`,
    );
    if (poisson < 0 || poisson >= 0.5) {
      throw new Error(`${label}.material.poissonRatio must be in [0, 0.5).`);
    }
    finiteArray(material.color, 4, `${label}.material.color`);

    const simulation = record(fixture.simulation, `${label}.simulation`);
    positive(simulation.timestep, `${label}.simulation.timestep`);
    finiteArray(simulation.gravity, 3, `${label}.simulation.gravity`);
    finite(simulation.floorY, `${label}.simulation.floorY`);
    nonnegative(
      simulation.floorStiffness,
      `${label}.simulation.floorStiffness`,
    );
    literal(simulation.parityMode, true, `${label}.simulation.parityMode`);

    const solver = record(fixture.solver, `${label}.solver`);
    positiveInteger(solver.iterations, `${label}.solver.iterations`);
    literal(solver.schedule, "jacobi", `${label}.solver.schedule`);
    const cubature = record(solver.cubature, `${label}.solver.cubature`);
    literal(
      cubature.mode,
      "exact-all-elements",
      `${label}.solver.cubature.mode`,
    );
    positiveInteger(
      cubature.packedSamplesPerSubproblem,
      `${label}.solver.cubature.packedSamplesPerSubproblem`,
    );

    const tolerances = record(
      fixture.expectedTolerances,
      `${label}.expectedTolerances`,
    );
    positiveRecord(tolerances.cpu, `${label}.expectedTolerances.cpu`);
    positiveRecord(tolerances.gpu, `${label}.expectedTolerances.gpu`);
    positive(
      tolerances.acceptedPoseMinimumDeterminant,
      `${label}.expectedTolerances.acceptedPoseMinimumDeterminant`,
    );
  }

  const corpora = array(root.corpora, "corpora");
  if (corpora.length === 0) throw new Error("corpora must not be empty.");
  const corpusIds = new Set<string>();
  for (const [index, value] of corpora.entries()) {
    const label = `corpora[${index}]`;
    const corpus = record(value, label);
    unique(corpusIds, stableId(corpus.id, `${label}.id`), "corpus ID");
    const fixtureId = stableId(corpus.fixtureId, `${label}.fixtureId`);
    if (!fixtureIds.has(fixtureId)) {
      throw new Error(`${label}.fixtureId references an unknown fixture.`);
    }
    const generator = validateGenerator(
      corpus.generator,
      `${label}.generator`,
      true,
    );
    const caseCount = positiveInteger(
      generator.caseCount,
      `${label}.generator.caseCount`,
    );
    const layout = array(corpus.requiredCaseLayout, `${label}.requiredCaseLayout`);
    if (layout.length === 0) {
      throw new Error(`${label}.requiredCaseLayout must not be empty.`);
    }
    let previousIndex = -1;
    for (const [caseIndex, value] of layout.entries()) {
      const caseLabel = `${label}.requiredCaseLayout[${caseIndex}]`;
      const entry = record(value, caseLabel);
      const stateIndex = nonnegativeInteger(entry.index, `${caseLabel}.index`);
      if (stateIndex <= previousIndex || stateIndex >= caseCount) {
        throw new Error(`${caseLabel}.index must be sorted and in range.`);
      }
      previousIndex = stateIndex;
      oneOf(
        entry.kind,
        ["rest", "rigid", "compression", "shear", "stretch"] as const,
        `${caseLabel}.kind`,
      );
      positive(entry.determinant, `${caseLabel}.determinant`);
    }
    const anchors = record(corpus.anchors, `${label}.anchors`);
    const first = validateAnchor(anchors.first, `${label}.anchors.first`);
    const last = validateAnchor(anchors.last, `${label}.anchors.last`);
    if (first.index !== 0 || last.index !== caseCount - 1) {
      throw new Error(`${label}.anchors must freeze the first and last states.`);
    }
  }
  return value as Phase1SceneManifest;
}

function validateGenerator(
  value: unknown,
  label: string,
  withCaseCount: boolean,
): Record<string, unknown> {
  const generator = record(value, label);
  stableId(generator.id, `${label}.id`);
  nonempty(generator.version, `${label}.version`);
  nonnegativeInteger(generator.seed, `${label}.seed`);
  record(generator.parameters, `${label}.parameters`);
  if (withCaseCount) positiveInteger(generator.caseCount, `${label}.caseCount`);
  return generator;
}

function validateAnchor(value: unknown, label: string): Phase1AnchorState {
  const anchor = record(value, label);
  stateId(anchor.id, `${label}.id`);
  nonnegativeInteger(anchor.index, `${label}.index`);
  oneOf(
    anchor.kind,
    ["rest", "rigid", "compression", "shear", "stretch", "seeded"] as const,
    `${label}.kind`,
  );
  positive(anchor.determinant, `${label}.determinant`);
  finiteArray(anchor.deformationGradient, 9, `${label}.deformationGradient`);
  const positions = array(anchor.positions, `${label}.positions`);
  if (positions.length !== 12) {
    throw new Error(`${label}.positions must contain 12 entries.`);
  }
  for (const [index, entry] of positions.entries()) {
    finite(entry, `${label}.positions[${index}]`);
  }
  return anchor as unknown as Phase1AnchorState;
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

function stateId(value: unknown, label: string): string {
  const id = nonempty(value, label);
  if (!/^[a-z0-9]+(?:[._-][a-z0-9]+)*\/[0-9]+$/.test(id)) {
    throw new Error(`${label} must be a stable corpus-state ID.`);
  }
  return id;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function positive(value: unknown, label: string): number {
  const result = finite(value, label);
  if (result <= 0) throw new Error(`${label} must be positive.`);
  return result;
}

function nonnegative(value: unknown, label: string): number {
  const result = finite(value, label);
  if (result < 0) throw new Error(`${label} must be nonnegative.`);
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

function finiteArray(value: unknown, length: number, label: string): void {
  const entries = array(value, label);
  if (entries.length !== length) {
    throw new Error(`${label} must contain ${length} entries.`);
  }
  for (const [index, entry] of entries.entries()) {
    finite(entry, `${label}[${index}]`);
  }
}

function positiveRecord(value: unknown, label: string): void {
  const entries = Object.entries(record(value, label));
  if (entries.length === 0) throw new Error(`${label} must not be empty.`);
  for (const [key, entry] of entries) positive(entry, `${label}.${key}`);
}

function literal<T>(value: unknown, expected: T, label: string): void {
  if (value !== expected) {
    throw new Error(`${label} must equal ${String(expected)}.`);
  }
}

function oneOf<const T extends readonly string[]>(
  value: unknown,
  expected: T,
  label: string,
): T[number] {
  if (!expected.includes(value as never)) {
    throw new Error(`${label} must be one of ${expected.join(", ")}.`);
  }
  return value as T[number];
}

function unique(values: Set<string>, value: string, label: string): void {
  if (values.has(value)) throw new Error(`Duplicate ${label}: ${value}.`);
  values.add(value);
}
