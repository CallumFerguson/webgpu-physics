import { expect, test } from "@playwright/test";

import {
  NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS,
  NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE,
} from "../../src/simulation/cpu/nonlinear-cubature-training";

interface NonlinearCubatureGpuCase {
  readonly id: string;
  readonly predictionRelativeError: number;
  readonly updateRelativeError: number;
  readonly minimumSolvedDeterminant: number;
  readonly finite: boolean;
  readonly explicitReadbacks: number;
}

interface NonlinearCubatureGpuResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly iterations: number;
  readonly cases: readonly NonlinearCubatureGpuCase[];
}

test("P1-EC-12-GPU: production nonlinear Cubature updates match the packed CPU reference", async ({
  page,
}) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  await page.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });

  const result = await page.evaluate<NonlinearCubatureGpuResult>(async () => {
    const [cpuApi, scenes, phase1, gpuApi] = await Promise.all([
      import("/src/simulation/cpu/index.ts"),
      import("/src/scenes/index.ts"),
      import("/src/scenes/phase1.ts"),
      import("/src/simulation/gpu/index.ts"),
    ]);
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("Nonlinear Cubature parity requires hardware WebGPU.");
    }
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    const device = await adapter.requestDevice();
    const uncapturedErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      uncapturedErrors.push(event.error.message);
    });
    device.pushErrorScope("validation");
    const parityIterations =
      cpuApi.NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS;

    const definition = phase1.buildPhase1NonlinearCubatureDefinition();
    const scene = cpuApi.buildPrecomputedScene(definition, {
      retainExactBases: true,
    });
    const corpus = cpuApi.buildNonlinearCubaturePoseCorpus({
      mesh: scene.mesh,
      restData: scene.restTetraData,
      materials: scene.materials,
      lumpedMasses: scene.lumpedMasses,
      restSystem: scene.restSystem,
    });
    const input = scenes.toJGS2GpuInput(scene);
    const selectedCases = [
      { pose: corpus.training[0]!, useFrozenTarget: false },
      { pose: corpus.training[7]!, useFrozenTarget: false },
      { pose: corpus.training.at(-1)!, useFrozenTarget: false },
      { pose: corpus.validation[0]!, useFrozenTarget: true },
      { pose: corpus.validation[4]!, useFrozenTarget: true },
      { pose: corpus.validation.at(-1)!, useFrozenTarget: true },
    ];

    const mirrorRestPositions = new Float64Array(input.vertexCount * 3);
    const mirrorMasses = new Float64Array(input.vertexCount);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        mirrorRestPositions[vertex * 3 + coordinate] =
          input.vertexRest[vertex * 4 + coordinate]!;
      }
      mirrorMasses[vertex] = input.vertexRest[vertex * 4 + 3]!;
    }
    const mirrorInverseRest = new Float64Array(input.tetCount * 9);
    const mirrorVolumes = new Float64Array(input.tetCount);
    for (let tetrahedron = 0; tetrahedron < input.tetCount; tetrahedron += 1) {
      mirrorVolumes[tetrahedron] = input.tetMeta[tetrahedron * 4]!;
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 3; column += 1) {
          mirrorInverseRest[tetrahedron * 9 + row * 3 + column] =
            input.tetInverseDm[tetrahedron * 12 + column * 4 + row]!;
        }
      }
    }
    const mirrorMesh = {
      ...scene.mesh,
      positions: mirrorRestPositions,
    };
    const mirrorRestData = {
      volumes: mirrorVolumes,
      inverseRestMatrices: mirrorInverseRest,
      stiffnessMatrices: Float64Array.from(input.tetRestStiffness),
    };
    const precomputationByVertex = new Map(
      scene.vertexPrecomputations.map((entry) => [entry.vertex, entry]),
    );
    const cases: NonlinearCubatureGpuCase[] = [];

    for (const { pose, useFrozenTarget } of selectedCases) {
      const packedPositions = input.positions.slice();
      const packedVelocities = new Float32Array(input.vertexCount * 4);
      const requestedPredictedPositions = new Float64Array(
        input.vertexCount * 3,
      );
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          const value = Math.fround(pose.positions[vertex * 3 + coordinate]!);
          const predicted = Math.fround(
            (useFrozenTarget ? pose.predictedPositions : pose.positions)[
              vertex * 3 + coordinate
            ]!,
          );
          packedPositions[vertex * 4 + coordinate] = value;
          requestedPredictedPositions[vertex * 3 + coordinate] = predicted;
          packedVelocities[vertex * 4 + coordinate] = Math.fround(
            (predicted - value) / definition.settings.timestep,
          );
        }
        packedPositions[vertex * 4 + 3] = 1;
      }
      const solver = await gpuApi.JGS2GpuSolver.create(
        device,
        {
          ...input,
          positions: packedPositions,
          velocities: packedVelocities,
        },
        {
          timestep: definition.settings.timestep,
          gravity: [0, 0, 0],
          iterations: 1,
          floorHeight: definition.settings.floorY,
          floorStiffness: 0,
          velocityDamping: 1,
          regularization: 1e-12,
          rotationEpsilon: 1e-7,
          maxStep: 0,
          contactTangentialDamping: 0,
          contactMargin: 0,
          horizontalBodyCorrection: false,
          parityMode: true,
        },
      );
      solver.stepExactIterations(parityIterations);
      await solver.awaitIdle();
      const predicted4 = await solver.readPredictedPositions();
      const solved4 = await solver.readPositions();
      const snappedPredictedPositions = new Float64Array(
        input.vertexCount * 3,
      );
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          snappedPredictedPositions[vertex * 3 + coordinate] =
            predicted4[vertex * 4 + coordinate]!;
        }
      }
      const gpuUpdates: number[] = [];
      const cpuUpdates: number[] = [];
      let cpuPositions = snappedPredictedPositions.slice();
      for (
        let iteration = 0;
        iteration < parityIterations;
        iteration += 1
      ) {
        const nextCpuPositions = cpuPositions.slice();
        for (const vertex of scene.restSystem.activeVertices) {
          const precomputation = precomputationByVertex.get(vertex)!;
          if (!precomputation.exactBasis) {
            throw new Error(`Missing retained basis for vertex ${vertex}.`);
          }
          const packedExactBasis = Float64Array.from(
            Float32Array.from(precomputation.exactBasis),
          );
          const selected = cpuApi.assembleStableNeoHookeanJGS2LocalSystem(
            {
              mesh: mirrorMesh,
              restData: mirrorRestData,
              materials: scene.materials,
              lumpedMasses: mirrorMasses,
              restSystem: scene.restSystem,
              timestep: definition.settings.timestep,
              predictedPositions: snappedPredictedPositions,
              sourceVertex: vertex,
              exactBasis: packedExactBasis,
            },
            cpuPositions,
            precomputation.cubature,
          );
          if (!selected.newtonUpdate) {
            throw new Error(
              `${pose.id} iteration ${iteration} vertex ${vertex} has a ` +
                `singular CPU update.`,
            );
          }
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            nextCpuPositions[vertex * 3 + coordinate] = Math.fround(
              cpuPositions[vertex * 3 + coordinate]! +
                selected.newtonUpdate[coordinate]!,
            );
          }
        }
        cpuPositions = nextCpuPositions;
      }
      for (const vertex of scene.restSystem.activeVertices) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          gpuUpdates.push(
            solved4[vertex * 4 + coordinate]! -
              snappedPredictedPositions[vertex * 3 + coordinate]!,
          );
          cpuUpdates.push(
            cpuPositions[vertex * 3 + coordinate]! -
              snappedPredictedPositions[vertex * 3 + coordinate]!,
          );
        }
      }
      const solvedPositions = new Float64Array(input.vertexCount * 3);
      let finite = true;
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          const value = solved4[vertex * 4 + coordinate]!;
          solvedPositions[vertex * 3 + coordinate] = value;
          finite &&= Number.isFinite(value);
        }
      }
      const solvedMaterial = cpuApi.evaluateStableNeoHookeanMesh(
        mirrorMesh,
        mirrorRestData,
        scene.materials,
        solvedPositions,
      );
      cases.push({
        id: pose.id,
        predictionRelativeError: cpuApi.relativeError(
          snappedPredictedPositions,
          requestedPredictedPositions,
        ),
        updateRelativeError: cpuApi.relativeError(gpuUpdates, cpuUpdates),
        minimumSolvedDeterminant: Math.min(
          ...solvedMaterial.deformationDeterminants,
        ),
        finite,
        explicitReadbacks: solver.explicitDiagnosticReadbackCount,
      });
      solver.destroy();
    }

    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    device.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      iterations: parityIterations,
      cases,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(result.iterations).toBe(NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS);
  expect(browserErrors).toEqual([]);
  expect(result.cases).toHaveLength(6);
  let maximumError = 0;
  let maximumPredictionError = 0;
  for (const entry of result.cases) {
    expect(entry.finite, entry.id).toBe(true);
    expect(entry.minimumSolvedDeterminant, entry.id).toBeGreaterThan(0);
    expect(entry.explicitReadbacks, entry.id).toBe(2);
    expect(entry.predictionRelativeError, entry.id).toBeLessThanOrEqual(1e-6);
    expect(entry.updateRelativeError, entry.id).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE,
    );
    maximumError = Math.max(maximumError, entry.updateRelativeError);
    maximumPredictionError = Math.max(
      maximumPredictionError,
      entry.predictionRelativeError,
    );
  }
  console.log(
    `P1 production nonlinear Cubature GPU update parity: ${result.cases.length} ` +
      `poses, maximum relative error ${maximumError.toExponential(3)}, ` +
      `prediction error ${maximumPredictionError.toExponential(3)}`,
  );
});
