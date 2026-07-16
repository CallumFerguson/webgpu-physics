import { describe, expect, it } from "vitest";

import {
  getTetrahedronCount,
  getVertexCount,
  transformTetrahedralMesh,
} from "../simulation/cpu";
import {
  DEFAULT_SCENE_ID,
  FORCE_FREE_CONSERVATION_FIXTURE_ID,
  PHASE0_FORCE_FREE_BASE_STATE,
  PHASE0_FORCE_FREE_CORPUS_CASE_COUNT,
  PHASE0_FORCE_FREE_CORPUS_GENERATOR_VERSION,
  PHASE0_FORCE_FREE_FIXTURE_ID,
  PHASE0_FORCE_FREE_INITIAL_EULER,
  PHASE0_FORCE_FREE_ITERATIONS,
  PHASE0_FORCE_FREE_TIMESTEP,
  SCENE_IDS,
  buildForceFreeConservationScene,
  buildScene,
  buildSceneDefinition,
  generatePhase0ForceFreeInitialStateCorpus,
  toForceFreeConservationGpuInput,
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

  it("implements the frozen exact-all-element force-free fixture", () => {
    expect(SCENE_IDS).not.toContain(FORCE_FREE_CONSERVATION_FIXTURE_ID);

    const firstScene = buildForceFreeConservationScene();
    const secondScene = buildForceFreeConservationScene();
    const firstInput = toForceFreeConservationGpuInput(firstScene);
    const secondInput = toForceFreeConservationGpuInput(secondScene);
    const velocities = firstInput.velocities;

    expect(firstScene.id).toBe(PHASE0_FORCE_FREE_FIXTURE_ID);
    expect(firstScene.mesh.positions.length / 3).toBe(27);
    expect(getTetrahedronCount(firstScene.mesh)).toBe(48);
    expect(firstScene.mesh.positions[0]).toBe(-0.5);
    expect(firstScene.mesh.positions[1]).toBe(-0.5);
    expect(firstScene.mesh.positions[2]).toBe(-0.5);
    expect(firstScene.settings.timestep).toBe(PHASE0_FORCE_FREE_TIMESTEP);
    expect(firstScene.settings.solverIterations).toBe(
      PHASE0_FORCE_FREE_ITERATIONS,
    );
    expect(firstScene.settings.gravity).toEqual([0, 0, 0]);
    expect(firstScene.materials[0]).toMatchObject({
      density: 1_000,
      youngModulus: 80_000,
      poissonRatio: 0.3,
    });
    expect([...firstScene.mesh.fixed]).toEqual(
      Array.from({ length: firstInput.vertexCount }, () => 0),
    );
    expect(new Set(firstScene.mesh.bodyIds)).toEqual(new Set([0]));
    expect(velocities).toBeDefined();
    expect([...secondInput.velocities!]).toEqual([...velocities!]);
    expect([...secondInput.positions]).toEqual([...firstInput.positions]);
    expect(firstInput.cubatureK).toBe(firstInput.tetCount);
    expect(firstInput.cubatureK).toBe(48);
    expect(PHASE0_FORCE_FREE_INITIAL_EULER).toEqual([0.17, -0.23, 0.31]);
    const expectedInitialPose = transformTetrahedralMesh(firstScene.mesh, {
      rotationEuler: PHASE0_FORCE_FREE_INITIAL_EULER,
    });
    for (let vertex = 0; vertex < firstInput.vertexCount; vertex += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        expect(firstInput.positions[vertex * 4 + axis]).toBeCloseTo(
          expectedInitialPose.positions[vertex * 3 + axis]!,
          6,
        );
      }
    }

    for (let vertex = 0; vertex < firstInput.vertexCount; vertex += 1) {
      const precomputation = firstScene.vertexPrecomputations[vertex]!;
      expect(precomputation.vertex).toBe(vertex);
      expect(precomputation.exactBasis).toBeDefined();
      expect(precomputation.trainingResidual).toBe(0);
      expect(precomputation.cubature).toHaveLength(firstInput.tetCount);
      const recordBase = vertex * firstInput.cubatureK;
      expect(
        [...firstInput.cubatureTetIds.subarray(recordBase, recordBase + 48)],
      ).toEqual(Array.from({ length: 48 }, (_unused, tet) => tet));
      expect(
        [...firstInput.cubatureWeights.subarray(recordBase, recordBase + 48)],
      ).toEqual(Array.from({ length: 48 }, () => 1));
      for (let tetrahedron = 0; tetrahedron < firstInput.tetCount; tetrahedron += 1) {
        const record = recordBase + tetrahedron;
        for (let localVertex = 0; localVertex < 4; localVertex += 1) {
          const targetVertex = firstScene.mesh.tetrahedra[
            tetrahedron * 4 + localVertex
          ]!;
          for (let entry = 0; entry < 9; entry += 1) {
            expect(
              firstInput.cubatureBasis[
                record * 36 + localVertex * 9 + entry
              ],
            ).toBeCloseTo(
              precomputation.exactBasis![targetVertex * 9 + entry]!,
              5,
            );
          }
        }
      }
    }

    const distinctVelocities = new Set(
      Array.from({ length: firstInput.vertexCount }, (_unused, vertex) =>
        [
          velocities![vertex * 4],
          velocities![vertex * 4 + 1],
          velocities![vertex * 4 + 2],
        ].join(","),
      ),
    );
    expect(distinctVelocities.size).toBeGreaterThan(1);

    let totalMass = 0;
    const totalMomentum = [0, 0, 0];
    for (let vertex = 0; vertex < firstInput.vertexCount; vertex += 1) {
      const mass = firstScene.lumpedMasses[vertex]!;
      totalMass += mass;
      for (let axis = 0; axis < 3; axis += 1) {
        totalMomentum[axis] += mass * velocities![vertex * 4 + axis]!;
      }
      expect(velocities![vertex * 4 + 3]).toBe(0);
    }
    expect(totalMomentum[0] / totalMass).toBeCloseTo(
      PHASE0_FORCE_FREE_BASE_STATE.linearVelocity[0],
      6,
    );
    expect(totalMomentum[1] / totalMass).toBeCloseTo(
      PHASE0_FORCE_FREE_BASE_STATE.linearVelocity[1],
      6,
    );
    expect(totalMomentum[2] / totalMass).toBeCloseTo(
      PHASE0_FORCE_FREE_BASE_STATE.linearVelocity[2],
      6,
    );
  }, 30_000);

  it("generates and packs every frozen force-free corpus state", () => {
    const first = generatePhase0ForceFreeInitialStateCorpus();
    const second = generatePhase0ForceFreeInitialStateCorpus();
    expect(first).toEqual(second);
    expect(PHASE0_FORCE_FREE_CORPUS_GENERATOR_VERSION).toBe("1");
    expect(first).toHaveLength(PHASE0_FORCE_FREE_CORPUS_CASE_COUNT);
    expect(new Set(first.map((entry) => entry.id)).size).toBe(
      PHASE0_FORCE_FREE_CORPUS_CASE_COUNT,
    );
    expect(first[0]).toEqual({
      id: "phase0.force-free-initial-states/00",
      linearVelocity: [
        -0.45846957506529673,
        0.33551068797554123,
        -0.3184711985400522,
      ],
      angularVelocity: [
        -0.9724775707987993,
        0.21396913055014707,
        0.05338255748009366,
      ],
    });
    expect(first[31]).toEqual({
      id: "phase0.force-free-initial-states/31",
      linearVelocity: [
        -0.1613361445196155,
        -0.5490207886538739,
        -0.4162380237814731,
      ],
      angularVelocity: [
        -0.2369681366893712,
        -0.3062079393113022,
        -0.32347516329834597,
      ],
    });

    const scene = buildForceFreeConservationScene();
    for (const state of first) {
      const linearSpeed = Math.hypot(...state.linearVelocity);
      const angularSpeed = Math.hypot(...state.angularVelocity);
      expect(linearSpeed).toBeGreaterThanOrEqual(0.1);
      expect(linearSpeed).toBeLessThanOrEqual(1);
      expect(angularSpeed).toBeGreaterThanOrEqual(0.1);
      expect(angularSpeed).toBeLessThanOrEqual(1);

      const input = toForceFreeConservationGpuInput(scene, state);
      let totalMass = 0;
      const linearMomentum = [0, 0, 0];
      const centerOfMass = [0, 0, 0];
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        const mass = scene.lumpedMasses[vertex]!;
        totalMass += mass;
        for (let axis = 0; axis < 3; axis += 1) {
          centerOfMass[axis] += mass * input.positions[vertex * 4 + axis]!;
        }
        for (let axis = 0; axis < 3; axis += 1) {
          linearMomentum[axis] += mass * input.velocities![vertex * 4 + axis]!;
        }
      }
      for (let axis = 0; axis < 3; axis += 1) {
        centerOfMass[axis] /= totalMass;
      }
      for (let axis = 0; axis < 3; axis += 1) {
        expect(linearMomentum[axis]! / totalMass).toBeCloseTo(
          state.linearVelocity[axis],
          5,
        );
      }
      const angularMomentum = [0, 0, 0];
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        const mass = scene.lumpedMasses[vertex]!;
        const rx = input.positions[vertex * 4]! - centerOfMass[0]!;
        const ry = input.positions[vertex * 4 + 1]! - centerOfMass[1]!;
        const rz = input.positions[vertex * 4 + 2]! - centerOfMass[2]!;
        const px = mass * (input.velocities![vertex * 4]! - state.linearVelocity[0]);
        const py = mass * (input.velocities![vertex * 4 + 1]! - state.linearVelocity[1]);
        const pz = mass * (input.velocities![vertex * 4 + 2]! - state.linearVelocity[2]);
        angularMomentum[0] += ry * pz - rz * py;
        angularMomentum[1] += rz * px - rx * pz;
        angularMomentum[2] += rx * py - ry * px;
      }
      expect(Math.hypot(...angularMomentum)).toBeGreaterThan(1e-6);
    }
  }, 30_000);

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
