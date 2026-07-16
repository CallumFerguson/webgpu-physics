import { describe, expect, it } from "vitest";

import { getVertexCount } from "../simulation/cpu";
import {
  DEFAULT_SCENE_ID,
  SCENE_IDS,
  buildScene,
  buildSceneDefinition,
  toJGS2GpuInput,
  type SceneId,
} from ".";

describe("procedural demo scenes", () => {
  it("uses the UI scene IDs and defaults to the minimal cantilever", () => {
    expect(SCENE_IDS).toEqual(["minimal", "stiffness", "drop", "stress"]);
    expect(DEFAULT_SCENE_ID).toBe("minimal");
    expect(buildSceneDefinition(DEFAULT_SCENE_ID).id).toBe("minimal");
  });

  it("builds every scene within the browser preprocessing budget", () => {
    for (const id of SCENE_IDS) {
      const start = performance.now();
      const scene = buildScene(id);
      const elapsed = performance.now() - start;
      const gpuInput = toJGS2GpuInput(scene);
      const maximumResidual = Math.max(
        ...scene.vertexPrecomputations.map((entry) => entry.trainingResidual),
      );
      const worst = scene.vertexPrecomputations.reduce((left, right) =>
        left.trainingResidual >= right.trainingResidual ? left : right,
      );

      console.log(
        `scene ${id}: ${getVertexCount(scene.mesh)} vertices, ` +
          `${scene.mesh.tetrahedra.length / 4} tets, ${elapsed.toFixed(1)} ms, ` +
          `max cubature residual ${maximumResidual.toFixed(6)} ` +
          `(vertex ${worst.vertex}, ${worst.cubature.length} samples)`,
      );
      expect(getVertexCount(scene.mesh)).toBeLessThanOrEqual(120);
      expect(scene.vertexPrecomputations.every((entry) => !entry.exactBasis)).toBe(
        true,
      );
      expect(gpuInput.vertexCount).toBe(getVertexCount(scene.mesh));
      expect(gpuInput.cubatureK).toBe(scene.settings.cubatureSamples);
      expect(scene.surface.triangles.length).toBeGreaterThan(0);
      expect(scene.surface.edges.length).toBeGreaterThan(0);
      expect(Number.isFinite(scene.settings.floorY)).toBe(true);
      expect(maximumResidual).toBeLessThan(0.3);
      if (id === "stress") {
        expect(elapsed).toBeLessThan(3_000);
      }
    }
  }, 20_000);

  it("is byte-for-byte deterministic", () => {
    const first = buildScene("minimal");
    const second = buildScene("minimal");
    const firstGpu = toJGS2GpuInput(first);
    const secondGpu = toJGS2GpuInput(second);

    expect([...first.mesh.positions]).toEqual([...second.mesh.positions]);
    expect([...first.mesh.tetrahedra]).toEqual([...second.mesh.tetrahedra]);
    expect([...first.surface.triangles]).toEqual([...second.surface.triangles]);
    expect([...firstGpu.cubatureTetIds]).toEqual([
      ...secondGpu.cubatureTetIds,
    ]);
    expect([...firstGpu.cubatureWeights]).toEqual([
      ...secondGpu.cubatureWeights,
    ]);
    expect([...firstGpu.cubatureBasis]).toEqual([...secondGpu.cubatureBasis]);
  });

  it("packs dense body metadata and valid mass inputs for COM correction", () => {
    const expectedBodyCounts: Record<SceneId, number> = {
      minimal: 1,
      stiffness: 2,
      drop: 1,
      stress: 6,
    };
    const expectedPinnedBodyCounts: Record<SceneId, number> = {
      minimal: 1,
      stiffness: 2,
      drop: 0,
      stress: 0,
    };

    for (const id of SCENE_IDS) {
      const input = toJGS2GpuInput(buildScene(id));
      const bodyIds = new Set<number>();
      const bodyHasPinnedVertex: boolean[] = [];
      const bodyMasses: number[] = [];
      const weightedRestPositions: [number, number, number][] = [];

      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        const bodyId = input.vertexInfo[vertex * 4 + 3]!;
        const mass = input.vertexRest[vertex * 4 + 3]!;
        bodyIds.add(bodyId);
        bodyHasPinnedVertex[bodyId] ||=
          input.vertexInfo[vertex * 4 + 2] !== 0;
        bodyMasses[bodyId] = (bodyMasses[bodyId] ?? 0) + mass;
        const weighted = (weightedRestPositions[bodyId] ??= [0, 0, 0]);
        for (let axis = 0; axis < 3; axis += 1) {
          weighted[axis] += input.vertexRest[vertex * 4 + axis]! * mass;
        }
      }

      const bodyCount = Math.max(...bodyIds) + 1;
      expect(bodyCount).toBe(expectedBodyCounts[id]);
      expect([...bodyIds].sort((left, right) => left - right)).toEqual(
        Array.from({ length: bodyCount }, (_unused, bodyId) => bodyId),
      );
      expect(bodyHasPinnedVertex.filter(Boolean)).toHaveLength(
        expectedPinnedBodyCounts[id],
      );

      for (let bodyId = 0; bodyId < bodyCount; bodyId += 1) {
        const mass = bodyMasses[bodyId]!;
        expect(Number.isFinite(mass)).toBe(true);
        expect(mass).toBeGreaterThan(0);
        for (const weightedCoordinate of weightedRestPositions[bodyId]!) {
          expect(Number.isFinite(weightedCoordinate / mass)).toBe(true);
        }
      }
    }
  }, 20_000);
});
