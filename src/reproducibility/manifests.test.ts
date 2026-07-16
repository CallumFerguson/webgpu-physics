import { describe, expect, it } from "vitest";

import baselineManifestJson from "../../manifests/baseline-tests.v1.json?raw";
import canonicalManifestJson from "../../manifests/canonical-scenes.v1.json?raw";
import { buildSceneDefinition, SCENE_IDS } from "../scenes";
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
      (fixture) => fixture.id === "phase0.force-free-cuboid",
    )!;
    const oracle = manifest.fixtures.find(
      (fixture) => fixture.id === "phase0.oracle-single-tetrahedron",
    )!;

    expect(forceFree.kind).toBe("force-free");
    expect(forceFree.simulation).toMatchObject({
      gravity: [0, 0, 0],
      floorY: null,
      parityMode: true,
    });
    expect(forceFree.sampledFrames.at(-1)).toMatchObject({
      frame: 1200,
      simulatedSeconds: 10,
    });
    expect(oracle.kind).toBe("oracle");
    expect(oracle.solver.cubature.mode).toBe("exact-all-elements");
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
