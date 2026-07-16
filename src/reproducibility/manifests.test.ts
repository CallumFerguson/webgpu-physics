import { describe, expect, it } from "vitest";

import baselineManifestJson from "../../manifests/baseline-tests.v1.json?raw";
import canonicalManifestJson from "../../manifests/canonical-scenes.v1.json?raw";
import {
  PHASE0_FORCE_FREE_BASE_STATE,
  PHASE0_FORCE_FREE_CORPUS_CASE_COUNT,
  PHASE0_FORCE_FREE_CORPUS_GENERATOR_VERSION,
  PHASE0_FORCE_FREE_CORPUS_ID,
  PHASE0_FORCE_FREE_CORPUS_SEED,
  PHASE0_FORCE_FREE_FIXTURE_ID,
  PHASE0_FORCE_FREE_FRAME_COUNT,
  PHASE0_FORCE_FREE_INITIAL_EULER,
  PHASE0_FORCE_FREE_ITERATIONS,
  PHASE0_FORCE_FREE_TIMESTEP,
  SCENE_IDS,
  buildPhase0ForceFreeDefinition,
  buildSceneDefinition,
} from "../scenes";
import {
  PHASE0_ORACLE_CORPUS_CASE_COUNT,
  PHASE0_ORACLE_CORPUS_GENERATOR_VERSION,
  PHASE0_ORACLE_CORPUS_ID,
  PHASE0_ORACLE_CORPUS_SEED,
  PHASE0_ORACLE_DISPLACEMENT_SCALE,
  PHASE0_ORACLE_FIXTURE_ID,
  PHASE0_ORACLE_MINIMUM_DETERMINANT,
  PHASE0_ORACLE_REST_POSITIONS,
  PHASE0_ORACLE_TIMESTEP,
  buildPhase0OracleDefinition,
} from "../simulation/cpu";
import {
  validateBaselineTestManifest,
  validateCanonicalSceneManifest,
} from "./manifests";

function baselineValue(): unknown {
  return JSON.parse(baselineManifestJson) as unknown;
}

function canonicalValue(): unknown {
  return JSON.parse(canonicalManifestJson) as unknown;
}

describe("checked-in reproducibility manifests", () => {
  it("P0-MANIFEST-001 freezes stable baseline IDs with no implicit skips", () => {
    const manifest = validateBaselineTestManifest(baselineValue());
    const unit = manifest.suites.find((suite) => suite.id === "unit")!;
    const e2e = manifest.suites.find((suite) => suite.id === "e2e")!;

    expect(unit.expectedTestCount).toBe(26);
    expect(e2e.expectedTestCount).toBe(7);
    expect(manifest.expectedTests).toHaveLength(33);
    expect(manifest.allowedSkips).toEqual([]);
    expect(manifest.requiredCommands).toContainEqual({
      id: "build",
      command: "npm.cmd run build",
    });
  });

  it("P0-MANIFEST-002 matches every frozen runtime scene parameter", () => {
    const manifest = validateCanonicalSceneManifest(canonicalValue());
    const runtimeFixtures = manifest.fixtures.filter(
      (fixture) => fixture.kind === "baseline-scene",
    );
    expect(runtimeFixtures.map((fixture) => fixture.runtimeSceneId)).toEqual([
      ...SCENE_IDS,
    ]);

    for (const sceneId of SCENE_IDS) {
      const definition = buildSceneDefinition(sceneId);
      const fixture = runtimeFixtures.find(
        (candidate) => candidate.runtimeSceneId === sceneId,
      )!;

      expect(fixture.simulation.timestep).toBe(definition.settings.timestep);
      expect(fixture.simulation.gravity).toEqual(definition.settings.gravity);
      expect(fixture.simulation.floorY).toBe(definition.settings.floorY);
      expect(fixture.solver.iterations).toBe(
        definition.settings.solverIterations,
      );
      expect(fixture.solver.cubature.samplesPerSubproblem).toBe(
        definition.settings.cubatureSamples,
      );
      expect(fixture.camera).toEqual(definition.camera);
      expect(fixture.materials).toHaveLength(definition.materials.length);
      for (let material = 0; material < definition.materials.length; material += 1) {
        const recorded = fixture.materials[material]!;
        const runtime = definition.materials[material]!;
        expect(recorded.density).toBe(runtime.density);
        expect(recorded.youngModulus).toBe(runtime.youngModulus);
        expect(recorded.poissonRatio).toBe(runtime.poissonRatio);
        expect(recorded.color).toEqual(runtime.color);
      }
    }
  });

  it("P0-MANIFEST-003 records force-free and oracle fixtures and corpora", () => {
    const manifest = validateCanonicalSceneManifest(canonicalValue());
    const forceFree = manifest.fixtures.find(
      (fixture) => fixture.id === PHASE0_FORCE_FREE_FIXTURE_ID,
    )!;
    const oracle = manifest.fixtures.find(
      (fixture) => fixture.id === PHASE0_ORACLE_FIXTURE_ID,
    )!;
    const forceFreeDefinition = buildPhase0ForceFreeDefinition();
    const oracleDefinition = buildPhase0OracleDefinition();
    const forceFreeCorpus = manifest.corpora.find(
      (corpus) => corpus.id === PHASE0_FORCE_FREE_CORPUS_ID,
    )!;
    const oracleCorpus = manifest.corpora.find(
      (corpus) => corpus.id === PHASE0_ORACLE_CORPUS_ID,
    )!;

    expect(forceFree.kind).toBe("force-free");
    expect(forceFree.generator).toMatchObject({
      id: "regular-cuboid-tetrahedra",
      version: "1",
      seed: PHASE0_FORCE_FREE_CORPUS_SEED,
      parameters: {
        cells: [2, 2, 2],
        origin: [-0.5, -0.5, -0.5],
        size: [1, 1, 1],
      },
    });
    expect(forceFree.simulation).toMatchObject({
      timestep: PHASE0_FORCE_FREE_TIMESTEP,
      gravity: [0, 0, 0],
      floorY: null,
      parityMode: true,
    });
    expect(forceFree.solver).toMatchObject({
      iterations: PHASE0_FORCE_FREE_ITERATIONS,
      schedule: "jacobi",
      cubature: {
        mode: "exact-all-elements",
        samplesPerSubproblem: null,
      },
    });
    expect(forceFree.initialState).toMatchObject({
      position:
        `generator-rest-pose rotated by [` +
        `${PHASE0_FORCE_FREE_INITIAL_EULER.join(", ")}] radians`,
      linearVelocity: PHASE0_FORCE_FREE_BASE_STATE.linearVelocity,
      angularVelocity: PHASE0_FORCE_FREE_BASE_STATE.angularVelocity,
    });
    expect(forceFree.sampledFrames.at(-1)).toMatchObject({
      frame: PHASE0_FORCE_FREE_FRAME_COUNT,
      simulatedSeconds: 10,
    });
    expect(forceFree.camera).toEqual(forceFreeDefinition.camera);
    expect(forceFree.materials[0]).toMatchObject({
      density: forceFreeDefinition.materials[0]!.density,
      youngModulus: forceFreeDefinition.materials[0]!.youngModulus,
      poissonRatio: forceFreeDefinition.materials[0]!.poissonRatio,
      color: forceFreeDefinition.materials[0]!.color,
    });
    expect(forceFreeDefinition.settings).toMatchObject({
      timestep: PHASE0_FORCE_FREE_TIMESTEP,
      gravity: [0, 0, 0],
      solverIterations: PHASE0_FORCE_FREE_ITERATIONS,
    });

    expect(oracle.kind).toBe("oracle");
    expect(oracle.generator).toMatchObject({
      id: "explicit-positive-tetrahedron",
      version: "1",
      seed: PHASE0_ORACLE_CORPUS_SEED,
      parameters: {
        positions: Array.from(
          { length: PHASE0_ORACLE_REST_POSITIONS.length / 3 },
          (_unused, vertex) =>
            Array.from(
              PHASE0_ORACLE_REST_POSITIONS.subarray(
                vertex * 3,
                vertex * 3 + 3,
              ),
            ),
        ),
        tetrahedra: [[0, 1, 2, 3]],
        fixedVertices: [0],
      },
    });
    expect(oracle.simulation.timestep).toBe(PHASE0_ORACLE_TIMESTEP);
    expect(oracle.solver.cubature.mode).toBe("exact-all-elements");
    expect(oracle.camera).toEqual(oracleDefinition.camera);
    expect(oracle.materials[0]).toMatchObject({
      density: oracleDefinition.materials[0]!.density,
      youngModulus: oracleDefinition.materials[0]!.youngModulus,
      poissonRatio: oracleDefinition.materials[0]!.poissonRatio,
      color: oracleDefinition.materials[0]!.color,
    });

    expect(forceFreeCorpus).toMatchObject({
      fixtureId: PHASE0_FORCE_FREE_FIXTURE_ID,
      generator: {
        id: "seeded-rigid-velocity-corpus",
        version: PHASE0_FORCE_FREE_CORPUS_GENERATOR_VERSION,
        seed: PHASE0_FORCE_FREE_CORPUS_SEED,
        caseCount: PHASE0_FORCE_FREE_CORPUS_CASE_COUNT,
        parameters: {
          linearSpeedRange: [0.1, 1],
          angularSpeedRange: [0.1, 1],
          rejectNearZeroAngularMomentum: true,
        },
      },
    });
    expect(oracleCorpus).toMatchObject({
      fixtureId: PHASE0_ORACLE_FIXTURE_ID,
      generator: {
        id: "seeded-positive-determinant-tetrahedron-poses",
        version: PHASE0_ORACLE_CORPUS_GENERATOR_VERSION,
        seed: PHASE0_ORACLE_CORPUS_SEED,
        caseCount: PHASE0_ORACLE_CORPUS_CASE_COUNT,
        parameters: {
          displacementScale: PHASE0_ORACLE_DISPLACEMENT_SCALE,
          minimumDeterminant: PHASE0_ORACLE_MINIMUM_DETERMINANT,
          includeRestPose: true,
          includeRigidRotations: true,
        },
      },
    });
    expect(manifest.corpora.map((corpus) => corpus.fixtureId)).toEqual([
      forceFree.id,
      oracle.id,
    ]);
    expect(manifest.corpora.every((corpus) => corpus.generator.seed > 0)).toBe(
      true,
    );
  });

  it("P0-MANIFEST-004 rejects count, reference, and frame-time drift", () => {
    const badCount = structuredClone(baselineValue()) as {
      suites: { id: string; expectedTestCount: number }[];
    };
    badCount.suites.find((suite) => suite.id === "unit")!.expectedTestCount = 25;
    expect(() => validateBaselineTestManifest(badCount)).toThrow(
      /expectedTestCount/,
    );

    const badReference = structuredClone(canonicalValue()) as {
      corpora: { fixtureId: string }[];
    };
    badReference.corpora[0]!.fixtureId = "phase0.missing";
    expect(() => validateCanonicalSceneManifest(badReference)).toThrow(
      /unknown fixture/,
    );

    const badFrameTime = structuredClone(canonicalValue()) as {
      fixtures: { sampledFrames: { simulatedSeconds: number }[] }[];
    };
    badFrameTime.fixtures[0]!.sampledFrames[1]!.simulatedSeconds = 1.25;
    expect(() => validateCanonicalSceneManifest(badFrameTime)).toThrow(
      /frame \* timestep/,
    );
  });
});
