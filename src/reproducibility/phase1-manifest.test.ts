import { describe, expect, it } from "vitest";

import phase1ManifestJson from "../../manifests/phase1-scenes.v1.json?raw";
import {
  PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_VERSION,
  PHASE1_STABLE_NEO_HOOKEAN_CPU_TOLERANCES,
  PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID,
  PHASE1_STABLE_NEO_HOOKEAN_GENERATOR_VERSION,
  PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES,
  PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
  PHASE1_STABLE_NEO_HOOKEAN_MAXIMUM_SEEDED_DETERMINANT,
  PHASE1_STABLE_NEO_HOOKEAN_MINIMUM_SEEDED_DETERMINANT,
  PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS,
  PHASE1_STABLE_NEO_HOOKEAN_RIGID_EULER_RADIANS,
  PHASE1_STABLE_NEO_HOOKEAN_SEEDED_ENTRY_SCALE,
  PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX,
  PHASE1_STABLE_NEO_HOOKEAN_SHEAR_GRADIENT,
  PHASE1_STABLE_NEO_HOOKEAN_STRETCH_GRADIENT,
  buildPhase1StableNeoHookeanOracleDefinition,
  generatePhase1StableNeoHookeanPoseCorpus,
} from "../scenes/phase1";
import { validatePhase1SceneManifest } from "./phase1-manifest";

function manifestValue(): unknown {
  return JSON.parse(phase1ManifestJson) as unknown;
}

describe("versioned Phase 1 scene manifest", () => {
  it("P1-MANIFEST-003 binds executable fixture and generator fields", () => {
    const manifest = validatePhase1SceneManifest(manifestValue());
    const fixture = manifest.fixtures[0]!;
    const corpus = manifest.corpora[0]!;
    const definition = buildPhase1StableNeoHookeanOracleDefinition();
    const definitionPositions = Array.from(
      { length: PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS.length / 3 },
      (_unused, vertex) =>
        Array.from(
          PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS.subarray(
            vertex * 3,
            vertex * 3 + 3,
          ),
        ),
    );
    const definitionTetrahedra = Array.from(
      { length: definition.mesh.tetrahedra.length / 4 },
      (_unused, tetrahedron) =>
        Array.from(
          definition.mesh.tetrahedra.subarray(
            tetrahedron * 4,
            tetrahedron * 4 + 4,
          ),
        ),
    );
    const definitionFixedVertices = Array.from(definition.mesh.fixed.entries())
      .filter(([_vertex, fixed]) => fixed !== 0)
      .map(([vertex]) => vertex);
    expect(new Set(definition.mesh.bodyIds)).toEqual(new Set([0]));

    expect(manifest.corpusVersion).toBe(
      PHASE1_STABLE_NEO_HOOKEAN_CORPUS_VERSION,
    );
    expect(manifest.fixtures).toHaveLength(1);
    expect(manifest.corpora).toHaveLength(1);
    expect(fixture).toMatchObject({
      id: PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID,
      phase: 1,
      kind: "stable-neo-hookean-oracle",
      runtimeSceneId: null,
      generator: {
        id: "explicit-unit-tetrahedron",
        version: "1",
        seed: 0,
        parameters: {
          positions: definitionPositions,
          tetrahedra: definitionTetrahedra,
          fixedVertices: definitionFixedVertices,
          materialId: definition.mesh.materialIds[0],
          bodyId: definition.mesh.bodyIds[0],
        },
      },
      material: {
        id: PHASE1_STABLE_NEO_HOOKEAN_MATERIAL.name,
        model: PHASE1_STABLE_NEO_HOOKEAN_MATERIAL.model,
        density: PHASE1_STABLE_NEO_HOOKEAN_MATERIAL.density,
        youngModulus: PHASE1_STABLE_NEO_HOOKEAN_MATERIAL.youngModulus,
        poissonRatio: PHASE1_STABLE_NEO_HOOKEAN_MATERIAL.poissonRatio,
        color: PHASE1_STABLE_NEO_HOOKEAN_MATERIAL.color,
      },
      simulation: {
        timestep: definition.settings.timestep,
        gravity: definition.settings.gravity,
        floorY: definition.settings.floorY,
        floorStiffness: 0,
        parityMode: true,
      },
      solver: {
        iterations: definition.settings.solverIterations,
        schedule: "jacobi",
        cubature: {
          mode: "exact-all-elements",
          packedSamplesPerSubproblem: definition.settings.cubatureSamples,
        },
      },
      expectedTolerances: {
        cpu: PHASE1_STABLE_NEO_HOOKEAN_CPU_TOLERANCES,
        gpu: PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES,
        acceptedPoseMinimumDeterminant: 1e-4,
      },
    });
    expect(corpus).toMatchObject({
      id: PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID,
      fixtureId: PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID,
      generator: {
        id: "seeded-affine-positive-determinant-poses",
        version: PHASE1_STABLE_NEO_HOOKEAN_GENERATOR_VERSION,
        seed: PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED,
        caseCount: PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT,
        parameters: {
          rigidEulerRadians: PHASE1_STABLE_NEO_HOOKEAN_RIGID_EULER_RADIANS,
          compressionDeterminants:
            PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS,
          shearGradient: [...PHASE1_STABLE_NEO_HOOKEAN_SHEAR_GRADIENT],
          stretchGradient: [...PHASE1_STABLE_NEO_HOOKEAN_STRETCH_GRADIENT],
          seededStartIndex: PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX,
          seededEntryScale: PHASE1_STABLE_NEO_HOOKEAN_SEEDED_ENTRY_SCALE,
          minimumSeededDeterminant:
            PHASE1_STABLE_NEO_HOOKEAN_MINIMUM_SEEDED_DETERMINANT,
          maximumSeededDeterminant:
            PHASE1_STABLE_NEO_HOOKEAN_MAXIMUM_SEEDED_DETERMINANT,
        },
      },
    });
  });

  it("P1-MANIFEST-004 freezes required edge cases and exact first/last states", () => {
    const manifest = validatePhase1SceneManifest(manifestValue());
    const corpus = manifest.corpora[0]!;
    const generated = generatePhase1StableNeoHookeanPoseCorpus();
    const first = generated[0]!;
    const last = generated.at(-1)!;

    expect(corpus.requiredCaseLayout).toEqual(
      generated.slice(0, PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX).map(
        ({ index, kind, determinant }) => ({ index, kind, determinant }),
      ),
    );
    expect(corpus.anchors.first).toEqual({
      id: first.id,
      index: first.index,
      kind: first.kind,
      determinant: first.determinant,
      deformationGradient: [...first.deformationGradient],
      positions: [...first.positions],
    });
    expect(corpus.anchors.last).toEqual({
      id: last.id,
      index: last.index,
      kind: last.kind,
      determinant: last.determinant,
      deformationGradient: [...last.deformationGradient],
      positions: [...last.positions],
    });
  });

  it("P1-MANIFEST-005 rejects schema, reference, and anchor drift", () => {
    const badSchema = structuredClone(manifestValue()) as { schemaVersion: number };
    badSchema.schemaVersion = 2;
    expect(() => validatePhase1SceneManifest(badSchema)).toThrow(/schemaVersion/);

    const badReference = structuredClone(manifestValue()) as {
      corpora: { fixtureId: string }[];
    };
    badReference.corpora[0]!.fixtureId = "phase1.missing";
    expect(() => validatePhase1SceneManifest(badReference)).toThrow(
      /unknown fixture/,
    );

    const badAnchor = structuredClone(manifestValue()) as {
      corpora: { anchors: { last: { index: number } } }[];
    };
    badAnchor.corpora[0]!.anchors.last.index = 62;
    expect(() => validatePhase1SceneManifest(badAnchor)).toThrow(
      /first and last states/,
    );
  });
});
