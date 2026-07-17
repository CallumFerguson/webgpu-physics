import { expect, test } from "@playwright/test";

import {
  NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS,
  NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE,
} from "../../src/simulation/cpu/nonlinear-cubature-training";

const FULL_E2E_QUALIFICATION =
  process.env.JGS2_FULL_E2E === "1" ||
  process.env.npm_lifecycle_event === "test:e2e:full";

interface NonlinearCubatureGpuCase {
  readonly id: string;
  readonly iterations: number;
  readonly activeLocalCount: number;
  readonly predictionRelativeError: number;
  readonly updateRelativeError: number;
  readonly minimumSolvedDeterminant: number;
  readonly finite: boolean;
  readonly explicitReadbacksBeforeDiagnostics: number;
  readonly explicitReadbacks: number;
  readonly globalization: {
    readonly historyCount: number;
    readonly localFinite: boolean;
    readonly localStatuses: readonly string[];
    readonly localDiagnostics: readonly {
      readonly status: string;
      readonly directionMagnitude: number;
      readonly alpha: number;
      readonly minimumEigenvalue: number;
      readonly scale: number;
      readonly diagonalShift: number;
      readonly normalizedShift: number;
      readonly acceptedEnergyDelta: number;
      readonly armijoDeltaBound: number;
      readonly gradientDotDirection: number;
      readonly shiftedLinearResidual: number;
      readonly minimumTrialDeformationDeterminant: number;
      readonly backtrackCount: number;
      readonly energyEvaluationCount: number;
    }[];
    readonly maximumNormalizedShift: number;
    readonly maximumBacktrackCount: number;
    readonly historyFinite: boolean;
    readonly assembledAccepted: boolean;
    readonly assembledReverted: boolean;
    readonly minimumAcceptedDeterminant: number;
    readonly localFailureCount: number;
    readonly finalRelativeResidual: number;
    readonly finalNormalizedMaximumUpdate: number;
    readonly oracleResidualRelativeError: number;
    readonly oracleMaximumUpdateRelativeError: number;
    readonly oracleMinimumDeterminantRelativeError: number;
    readonly oracleAcceptedEnergyRelativeError: number;
  };
}

interface NonlinearCubatureGpuResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly uniquePoseCount: number;
  readonly activeLocalCount: number;
  readonly twoIterationCaseCount: number;
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

  const result = await page.evaluate<NonlinearCubatureGpuResult, boolean>(async (
    fullQualification,
  ) => {
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
    const fullCases = [...corpus.training, ...corpus.validation];
    const selectedCases = fullQualification
      ? fullCases.map((pose, index) => ({
          pose,
          useFrozenTarget: true,
          // Preserve nonlinear depth at both ends of the frozen corpus while
          // keeping the exhaustive qualification pass inexpensive.
          iterations: index === 0 || index === fullCases.length - 1 ? 2 : 1,
        }))
      : [
          { pose: corpus.training[0]!, useFrozenTarget: false, iterations: 2 },
          { pose: corpus.training[7]!, useFrozenTarget: false, iterations: 2 },
          { pose: corpus.training.at(-1)!, useFrozenTarget: false, iterations: 2 },
          { pose: corpus.validation[0]!, useFrozenTarget: true, iterations: 2 },
          { pose: corpus.validation[4]!, useFrozenTarget: true, iterations: 2 },
          { pose: corpus.validation.at(-1)!, useFrozenTarget: true, iterations: 2 },
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

    for (const { pose, useFrozenTarget, iterations } of selectedCases) {
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
      solver.stepExactIterations(iterations);
      await solver.awaitIdle();
      const explicitReadbacksBeforeDiagnostics =
        solver.explicitDiagnosticReadbackCount;
      const predicted4 = await solver.readPredictedPositions();
      const solved4 = await solver.readPositions();
      const globalization = await solver.readGlobalizationDiagnostics();
      const oracle = await solver.readOracleDiagnostics();
      const snappedPredictedPositions = new Float64Array(
        input.vertexCount * 3,
      );
      const snappedSourcePositions = new Float64Array(input.vertexCount * 3);
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          snappedPredictedPositions[vertex * 3 + coordinate] =
            predicted4[vertex * 4 + coordinate]!;
          snappedSourcePositions[vertex * 3 + coordinate] =
            packedPositions[vertex * 4 + coordinate]!;
        }
      }
      const gpuUpdates: number[] = [];
      const cpuUpdates: number[] = [];
      let cpuPositions = snappedSourcePositions.slice();
      for (
        let iteration = 0;
        iteration < iterations;
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
              snappedSourcePositions[vertex * 3 + coordinate]!,
          );
          cpuUpdates.push(
            cpuPositions[vertex * 3 + coordinate]! -
              snappedSourcePositions[vertex * 3 + coordinate]!,
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
        iterations,
        activeLocalCount: scene.restSystem.activeVertices.length,
        predictionRelativeError: cpuApi.relativeError(
          snappedPredictedPositions,
          requestedPredictedPositions,
        ),
        updateRelativeError: cpuApi.relativeError(gpuUpdates, cpuUpdates),
        minimumSolvedDeterminant: Math.min(
          ...solvedMaterial.deformationDeterminants,
        ),
        finite,
        explicitReadbacksBeforeDiagnostics,
        explicitReadbacks: solver.explicitDiagnosticReadbackCount,
        globalization: {
          historyCount: globalization.historyCount,
          localFinite: scene.restSystem.activeVertices.every(
            (vertex) => globalization.local[vertex]!.finite,
          ),
          localStatuses: Array.from(
            scene.restSystem.activeVertices,
            (vertex) => globalization.local[vertex]!.status,
          ),
          localDiagnostics: Array.from(
            scene.restSystem.activeVertices,
            (vertex) => {
              const diagnostic = globalization.local[vertex]!;
              return {
                status: diagnostic.status,
                directionMagnitude: Math.hypot(...diagnostic.direction),
                alpha: diagnostic.alpha,
                minimumEigenvalue: diagnostic.minimumEigenvalue,
                scale: diagnostic.scale,
                diagonalShift: diagnostic.diagonalShift,
                normalizedShift: diagnostic.normalizedShift,
                acceptedEnergyDelta: diagnostic.acceptedEnergyDelta,
                armijoDeltaBound: diagnostic.armijoDeltaBound,
                gradientDotDirection: diagnostic.gradientDotDirection,
                shiftedLinearResidual: diagnostic.shiftedLinearResidual,
                minimumTrialDeformationDeterminant:
                  diagnostic.minimumTrialDeformationDeterminant,
                backtrackCount: diagnostic.backtrackCount,
                energyEvaluationCount: diagnostic.energyEvaluationCount,
              };
            },
          ),
          maximumNormalizedShift: Math.max(
            ...Array.from(
              scene.restSystem.activeVertices,
              (vertex) => globalization.local[vertex]!.normalizedShift,
            ),
          ),
          maximumBacktrackCount: Math.max(
            ...globalization.history.map(
              (record) => record.maximumBacktrackCount,
            ),
          ),
          historyFinite: globalization.history.every(
            (record) => record.finite,
          ),
          assembledAccepted: globalization.history.every(
            (record) => record.assembledAccepted,
          ),
          assembledReverted: globalization.history.some(
            (record) => record.assembledReverted,
          ),
          minimumAcceptedDeterminant: Math.min(
            ...globalization.history.map(
              (record) => record.acceptedMinimumDeformationDeterminant,
            ),
          ),
          localFailureCount: globalization.history.reduce(
            (sum, record) => sum + record.localFailureCount,
            0,
          ),
          finalRelativeResidual:
            globalization.history.at(-1)!.relativeResidual,
          finalNormalizedMaximumUpdate:
            globalization.history.at(-1)!.normalizedMaximumUpdate,
          oracleResidualRelativeError: gpuApi.jgs2DiagnosticRelativeError(
            globalization.history.at(-1)!.relativeResidual,
            oracle.relativeResidual,
          ),
          oracleMaximumUpdateRelativeError: gpuApi.jgs2DiagnosticRelativeError(
            globalization.history.at(-1)!.maximumUpdate,
            oracle.maximumUpdate,
          ),
          oracleMinimumDeterminantRelativeError:
            gpuApi.jgs2DiagnosticRelativeError(
              globalization.history.at(-1)!
                .acceptedMinimumDeformationDeterminant,
              oracle.minimumDeformationDeterminant,
            ),
          oracleAcceptedEnergyRelativeError: gpuApi.jgs2DiagnosticRelativeError(
            globalization.history.at(-1)!.acceptedEnergy,
            oracle.energy,
          ),
        },
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
      uniquePoseCount: new Set(cases.map((entry) => entry.id)).size,
      activeLocalCount: cases.reduce(
        (sum, entry) => sum + entry.activeLocalCount,
        0,
      ),
      twoIterationCaseCount: cases.filter((entry) => entry.iterations === 2)
        .length,
      cases,
    };
  }, FULL_E2E_QUALIFICATION);

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(browserErrors).toEqual([]);
  expect(result.cases).toHaveLength(FULL_E2E_QUALIFICATION ? 24 : 6);
  expect(result.uniquePoseCount).toBe(FULL_E2E_QUALIFICATION ? 24 : 6);
  expect(result.activeLocalCount).toBe(FULL_E2E_QUALIFICATION ? 240 : 60);
  expect(result.twoIterationCaseCount).toBe(
    FULL_E2E_QUALIFICATION ? 2 : 6,
  );
  let maximumError = 0;
  let maximumPredictionError = 0;
  for (const entry of result.cases) {
    expect(entry.iterations, entry.id).toBe(
      FULL_E2E_QUALIFICATION &&
        entry !== result.cases[0] &&
        entry !== result.cases.at(-1)
        ? 1
        : NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS,
    );
    expect(entry.activeLocalCount, entry.id).toBe(10);
    expect(entry.finite, entry.id).toBe(true);
    expect(entry.minimumSolvedDeterminant, entry.id).toBeGreaterThan(0);
    expect(entry.explicitReadbacksBeforeDiagnostics, entry.id).toBe(0);
    expect(entry.explicitReadbacks, entry.id).toBe(4);
    expect(entry.predictionRelativeError, entry.id).toBeLessThanOrEqual(1e-6);
    expect(entry.updateRelativeError, entry.id).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE,
    );
    expect(entry.globalization.historyCount, entry.id).toBe(
      entry.iterations,
    );
    expect(entry.globalization.localFinite, entry.id).toBe(true);
    expect(
      entry.globalization.localStatuses.every(
        (status) => status === "accepted" || status === "zero-gradient",
      ),
      `${entry.id} local diagnostics ${JSON.stringify(entry.globalization.localDiagnostics)}`,
    ).toBe(true);
    expect(entry.globalization.localDiagnostics).toHaveLength(10);
    for (const [localIndex, local] of entry.globalization.localDiagnostics.entries()) {
      const label = `${entry.id} local ${localIndex}`;
      expect(Number.isFinite(local.minimumEigenvalue), label).toBe(true);
      expect(local.scale, label).toBeGreaterThan(0);
      expect(local.diagonalShift, label).toBeGreaterThanOrEqual(0);
      expect(local.normalizedShift, label).toBeGreaterThanOrEqual(0);
      expect(local.normalizedShift, label).toBeLessThanOrEqual(1e-3);
      expect(local.shiftedLinearResidual, label).toBeLessThan(2e-5);
      if (local.status === "accepted") {
        expect(local.directionMagnitude, label).toBeGreaterThan(0);
        expect(local.alpha, label).toBeGreaterThan(0);
        expect(local.alpha, label).toBeLessThanOrEqual(1);
        expect(local.gradientDotDirection, label).toBeLessThan(0);
        expect(local.acceptedEnergyDelta, label).toBeLessThanOrEqual(
          local.armijoDeltaBound,
        );
        expect(local.minimumTrialDeformationDeterminant, label).toBeGreaterThan(
          1e-4,
        );
      } else {
        expect(local.status, label).toBe("zero-gradient");
        expect(local.alpha, label).toBe(0);
        expect(local.acceptedEnergyDelta, label).toBe(0);
        expect(local.armijoDeltaBound, label).toBe(0);
      }
    }
    expect(entry.globalization.maximumNormalizedShift, entry.id).toBeLessThanOrEqual(
      1e-3,
    );
    expect(entry.globalization.historyFinite, entry.id).toBe(true);
    expect(entry.globalization.assembledAccepted, entry.id).toBe(true);
    expect(entry.globalization.assembledReverted, entry.id).toBe(false);
    expect(entry.globalization.minimumAcceptedDeterminant, entry.id).toBeGreaterThan(
      1e-4,
    );
    expect(entry.globalization.localFailureCount, entry.id).toBe(0);
    expect(
      entry.globalization.oracleResidualRelativeError,
      `${entry.id} convergence residual reduction`,
    ).toBeLessThan(2e-5);
    expect(
      entry.globalization.oracleMaximumUpdateRelativeError,
      `${entry.id} accepted update reduction`,
    ).toBeLessThan(2e-5);
    expect(
      entry.globalization.oracleMinimumDeterminantRelativeError,
      `${entry.id} accepted determinant reduction`,
    ).toBeLessThan(2e-5);
    expect(
      entry.globalization.oracleAcceptedEnergyRelativeError,
      `${entry.id} accepted energy diagnostic`,
    ).toBeLessThan(2e-5);
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
