import { expect, test } from "@playwright/test";
import type { BrowserContext, Page } from "@playwright/test";

const FULL_E2E_QUALIFICATION =
  process.env.JGS2_FULL_E2E === "1" ||
  process.env.npm_lifecycle_event === "test:e2e:full";

test.use({ trace: "off" });

let sharedContext: BrowserContext;
let sharedPage: Page;
const sharedBrowserErrors: string[] = [];

test.beforeAll(async ({ browser }) => {
  sharedContext = await browser.newContext({
    baseURL: "http://127.0.0.1:4173",
    viewport: { width: 800, height: 600 },
    deviceScaleFactor: 1,
    colorScheme: "dark",
    locale: "en-US",
    timezoneId: "UTC",
    reducedMotion: "reduce",
  });
  sharedPage = await sharedContext.newPage();
  sharedPage.on("pageerror", (error) => sharedBrowserErrors.push(error.message));
  sharedPage.on("console", (message) => {
    if (message.type() === "error") sharedBrowserErrors.push(message.text());
  });
  await sharedPage.goto("/tests/e2e/numeric-harness.html", {
    waitUntil: "domcontentloaded",
  });
  await sharedPage.waitForFunction(
    () =>
      (
        globalThis as typeof globalThis & {
          __jgs2NumericHarnessReady?: boolean;
        }
      ).__jgs2NumericHarnessReady === true,
  );
  await sharedPage.evaluate(async () => {
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
      throw new Error("Nonlinear objective tests require hardware WebGPU.");
    }
    const device = await adapter.requestDevice();
    const uncapturedErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      uncapturedErrors.push(event.error.message);
    });
    const stableDefinition =
      phase1.buildPhase1StableNeoHookeanOracleDefinition();
    const stableScene = cpuApi.buildPrecomputedScene(stableDefinition, {
      retainExactBases: true,
    });
    const beamDefinition = phase1.buildPhase1NonlinearCubatureDefinition();
    const beamScene = cpuApi.buildPrecomputedScene(beamDefinition, {
      retainExactBases: true,
    });
    Object.assign(globalThis, {
      __jgs2ObjectivesE2E: {
        cpuApi,
        scenes,
        phase1,
        gpuApi,
        adapter,
        device,
        uncapturedErrors,
        stableDefinition,
        stableScene,
        stableInput: scenes.toJGS2GpuInput(stableScene),
        beamDefinition,
        beamScene,
        beamInput: scenes.toJGS2GpuInput(beamScene),
      },
    });
  });
});

test.beforeEach(async () => {
  sharedBrowserErrors.length = 0;
  await sharedPage.evaluate(() => {
    const shared = (
      globalThis as typeof globalThis & {
        __jgs2ObjectivesE2E: { uncapturedErrors: string[] };
      }
    ).__jgs2ObjectivesE2E;
    shared.uncapturedErrors.length = 0;
  });
});

test.afterAll(async () => {
  if (sharedPage) {
    await sharedPage.evaluate(async () => {
      const shared = (
        globalThis as typeof globalThis & {
          __jgs2ObjectivesE2E: { device: GPUDevice };
        }
      ).__jgs2ObjectivesE2E;
      await shared.device.queue.onSubmittedWorkDone();
      shared.device.destroy();
      Reflect.deleteProperty(globalThis, "__jgs2ObjectivesE2E");
    });
  }
  await sharedContext?.close();
});

interface ObjectiveParityResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly explicitReadbacksBeforeDiagnostics: number;
  readonly explicitReadbacks: number;
  readonly local: readonly {
    readonly vertex: number;
    readonly status: string;
    readonly referenceDirectionMagnitude: number;
    readonly directionRelativeError: number;
    readonly alphaError: number;
    readonly acceptedEnergyDeltaError: number;
  }[];
  readonly externalForceNormRelativeError: number;
  readonly targetNormRelativeError: number;
  readonly oracleExternalForceEnergyRelativeError: number;
  readonly oracleTargetEnergyRelativeError: number;
  readonly oracleExternalForceNormRelativeError: number;
  readonly oracleTargetNormRelativeError: number;
  readonly oracleExternalForceGradientRelativeError: number;
  readonly oracleTargetGradientRelativeError: number;
  readonly oracleTotalGradientRelativeError: number;
  readonly oracleTargetHessianRelativeError: number;
  readonly oracleFinite: boolean;
  readonly baselineOracleFinite: boolean;
  readonly baselineExternalForceEnergy: number;
  readonly baselineTargetEnergy: number;
  readonly baselineExternalForceGradientNorm: number;
  readonly baselineTargetGradientNorm: number;
  readonly historyFinite: boolean;
  readonly assembledAccepted: boolean;
  readonly acceptedMinimumDeterminant: number;
  readonly sparseProbe: {
    readonly driverVertex: number;
    readonly projectionVertex: number;
    readonly sourceShareLeverage: number;
    readonly projectionDriverSampleCount: number;
    readonly driverObjectiveEffect: number;
    readonly projectionObjectiveEffect: number;
    readonly driverEffectRelativeError: number;
    readonly projectionEffectRelativeError: number;
    readonly driverStatus: string;
    readonly projectionStatus: string;
    readonly explicitReadbacksBeforeDiagnostics: number;
    readonly explicitReadbacks: number;
    readonly historyFinite: boolean;
    readonly assembledAccepted: boolean;
  };
}

test("P1-COMPOSITE-GPU: force and quadratic-target derivatives match the independent CPU reference", async () => {
  const result = await sharedPage.evaluate<ObjectiveParityResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2ObjectivesE2E: any }
    ).__jgs2ObjectivesE2E;
    const {
      cpuApi,
      gpuApi,
      adapter,
      device,
      uncapturedErrors,
      stableDefinition: definition,
      stableScene: scene,
      stableInput: input,
      beamDefinition,
      beamScene,
      beamInput,
    } = shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    const externalForces = new Float32Array(input.vertexCount * 3);
    const targetPositions = new Float32Array(input.vertexCount * 3);
    const targetStiffnesses = new Float32Array(input.vertexCount);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        targetPositions[vertex * 3 + coordinate] =
          input.positions[vertex * 4 + coordinate]!;
      }
    }
    for (const vertex of scene.restSystem.activeVertices) {
      const inertiaScale =
        input.vertexRest[vertex * 4 + 3]! /
        (definition.settings.timestep * definition.settings.timestep);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const phase = vertex * 0.71 + coordinate * 1.13;
        externalForces[vertex * 3 + coordinate] =
          0.008 * inertiaScale * Math.cos(phase);
        targetPositions[vertex * 3 + coordinate] +=
          0.03 * Math.sin(phase + 0.23);
      }
      targetStiffnesses[vertex] = 0.35 * inertiaScale;
    }
    const objective = {
      externalForces,
      targetPositions,
      targetStiffnesses,
    };
    const packedPositions = new Float64Array(input.vertexCount * 3);
    const mirrorMasses = new Float64Array(input.vertexCount);
    const mirrorRestPositions = new Float64Array(input.vertexCount * 3);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      mirrorMasses[vertex] = input.vertexRest[vertex * 4 + 3]!;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        packedPositions[vertex * 3 + coordinate] =
          input.positions[vertex * 4 + coordinate]!;
        mirrorRestPositions[vertex * 3 + coordinate] =
          input.vertexRest[vertex * 4 + coordinate]!;
      }
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
    const mirrorMesh = { ...scene.mesh, positions: mirrorRestPositions };
    const mirrorRestData = {
      volumes: mirrorVolumes,
      inverseRestMatrices: mirrorInverseRest,
      stiffnessMatrices: Float64Array.from(input.tetRestStiffness),
    };
    const objective64 = {
      externalForces: Float64Array.from(externalForces),
      targetPositions: Float64Array.from(targetPositions),
      targetStiffnesses: Float64Array.from(targetStiffnesses),
    };
    const cpuReferences = new Map<number, any>();
    for (const vertex of scene.restSystem.activeVertices) {
      const precomputation = scene.vertexPrecomputations.find(
        (entry: { vertex: number }) => entry.vertex === vertex,
      );
      if (!precomputation?.exactBasis) {
        throw new Error(`Missing retained objective basis for vertex ${vertex}.`);
      }
      cpuReferences.set(
        vertex,
        cpuApi.globalizeSelectedCubatureRestrictedLocal({
          context: {
            mesh: mirrorMesh,
            restData: mirrorRestData,
            materials: scene.materials,
            lumpedMasses: mirrorMasses,
            restSystem: scene.restSystem,
            timestep: definition.settings.timestep,
            predictedPositions: packedPositions,
            sourceVertex: vertex,
            exactBasis: Float64Array.from(
              Float32Array.from(precomputation.exactBasis),
            ),
          },
          positions: packedPositions,
          samples: precomputation.cubature.map((sample: any) => ({
            tetrahedron: sample.tetrahedron,
            weight: Math.fround(sample.weight),
            basisBlocks: Float64Array.from(
              Float32Array.from(sample.basisBlocks),
            ),
          })),
          objective: objective64,
        }),
      );
    }
    device.pushErrorScope("validation");
    const stepSettings = {
      timestep: definition.settings.timestep,
      gravity: [0, 0, 0],
      iterations: 1,
      floorStiffness: 0,
      velocityDamping: 1,
      regularization: 1e-12,
      maxStep: 0,
      contactTangentialDamping: 0,
      contactMargin: 0,
      horizontalBodyCorrection: false,
      parityMode: true,
    };
    const solver = await gpuApi.JGS2GpuSolver.create(
      device,
      { ...input, objectives: objective },
      stepSettings,
    );
    solver.stepExactIterations(1);
    await solver.awaitIdle();
    const explicitReadbacksBeforeDiagnostics =
      solver.explicitDiagnosticReadbackCount;
    const diagnostics = await solver.readGlobalizationDiagnostics();
    const positions = await solver.readPositions();
    const oracle = await solver.readOracleDiagnostics();
    solver.clearExternalForces();
    solver.releaseAllQuadraticTargets();
    const baselineOracle = await solver.readOracleDiagnostics();
    const relativeVectorError = (
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
    ) => {
      let squaredError = 0;
      let squaredReference = 0;
      for (let index = 0; index < expected.length; index += 1) {
        squaredError += (actual[index]! - expected[index]!) ** 2;
        squaredReference += expected[index]! ** 2;
      }
      return Math.sqrt(squaredError) / Math.max(1, Math.sqrt(squaredReference));
    };
    const local = Array.from(
      scene.restSystem.activeVertices,
      (vertex: number) => {
        const gpu = diagnostics.local[vertex]!;
        const cpu = cpuReferences.get(vertex)!;
        const cpuLineSearch = cpu.lineSearch;
        if (!cpuLineSearch) {
          throw new Error(`CPU objective direction failed for vertex ${vertex}.`);
        }
        const cpuEnergyDelta =
          cpuLineSearch.acceptedEnergy - cpuLineSearch.initialEnergy;
        return {
          vertex,
          status: gpu.status,
          referenceDirectionMagnitude: Math.hypot(
            ...cpu.direction.direction,
          ),
          directionRelativeError: relativeVectorError(
            gpu.direction,
            cpu.direction.direction,
          ),
          alphaError: Math.abs(gpu.alpha - cpuLineSearch.alpha),
          acceptedEnergyDeltaError: Math.abs(
            gpu.acceptedEnergyDelta - cpuEnergyDelta,
          ) / Math.max(1, Math.abs(cpuEnergyDelta)),
        };
      },
    );
    let expectedExternalSquared = 0;
    let expectedTargetSquared = 0;
    let expectedExternalEnergy = 0;
    let expectedTargetEnergy = 0;
    let oracleExternalForceGradientRelativeError = 0;
    let oracleTargetGradientRelativeError = 0;
    let oracleTotalGradientRelativeError = 0;
    let oracleTargetHessianRelativeError = 0;
    for (const vertex of scene.restSystem.activeVertices) {
      const stiffness = targetStiffnesses[vertex]!;
      const oracleVertex = oracle.vertices[vertex]!;
      const baselineVertex = baselineOracle.vertices[vertex]!;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const position = positions[vertex * 4 + coordinate]!;
        const force = externalForces[vertex * 3 + coordinate]!;
        const targetResidual =
          position - targetPositions[vertex * 3 + coordinate]!;
        expectedExternalSquared +=
          force ** 2;
        expectedExternalEnergy -= force * position;
        expectedTargetSquared +=
          (stiffness * targetResidual) ** 2;
        expectedTargetEnergy +=
          0.5 * stiffness * targetResidual * targetResidual;
        oracleExternalForceGradientRelativeError = Math.max(
          oracleExternalForceGradientRelativeError,
          Math.abs(oracleVertex.externalForceGradient[coordinate]! + force) /
            Math.max(1, Math.abs(force)),
        );
        const expectedTargetGradient = stiffness * targetResidual;
        oracleTargetGradientRelativeError = Math.max(
          oracleTargetGradientRelativeError,
          Math.abs(
            oracleVertex.targetGradient[coordinate]! -
              expectedTargetGradient,
          ) / Math.max(1, Math.abs(expectedTargetGradient)),
        );
        const expectedObjectiveGradient = -force + expectedTargetGradient;
        const actualObjectiveGradient =
          oracleVertex.gradient[coordinate]! -
          baselineVertex.gradient[coordinate]!;
        oracleTotalGradientRelativeError = Math.max(
          oracleTotalGradientRelativeError,
          Math.abs(actualObjectiveGradient - expectedObjectiveGradient) /
            Math.max(1, Math.abs(expectedObjectiveGradient)),
        );
      }
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 3; column += 1) {
          const index = row * 3 + column;
          const expected = row === column ? stiffness : 0;
          const actual =
            oracleVertex.localHessian[index]! -
            baselineVertex.localHessian[index]!;
          oracleTargetHessianRelativeError = Math.max(
            oracleTargetHessianRelativeError,
            Math.abs(actual - expected) / Math.max(1, Math.abs(expected)),
          );
        }
      }
    }
    const history = diagnostics.history[0]!;
    const expectedExternal = Math.sqrt(expectedExternalSquared);
    const expectedTarget = Math.sqrt(expectedTargetSquared);
    const explicitReadbacks = solver.explicitDiagnosticReadbackCount;

    const driverVertex = 11;
    const projectionVertex = 1;
    const beamActiveVertices = Array.from(
      beamScene.restSystem.activeVertices,
    );
    if (
      beamInput.vertexCount <= driverVertex ||
      !beamActiveVertices.includes(driverVertex) ||
      !beamActiveVertices.includes(projectionVertex)
    ) {
      throw new Error(
        "Sparse objective probe requires active beam vertices 11 (driver) " +
          "and 1 (projection); the canonical topology changed.",
      );
    }
    const driverPrecomputation = beamScene.vertexPrecomputations.find(
      (entry: { vertex: number }) => entry.vertex === driverVertex,
    );
    const projectionPrecomputation = beamScene.vertexPrecomputations.find(
      (entry: { vertex: number }) => entry.vertex === projectionVertex,
    );
    if (
      !driverPrecomputation?.exactBasis ||
      !projectionPrecomputation?.exactBasis
    ) {
      throw new Error(
        "Sparse objective probe requires retained exact bases for beam " +
          "vertices 11 and 1.",
      );
    }
    const sampleContainsVertex = (sample: any, vertex: number) => {
      const start = sample.tetrahedron * 4;
      return (
        sample.weight > 0 &&
        (beamScene.mesh.tetrahedra[start] === vertex ||
          beamScene.mesh.tetrahedra[start + 1] === vertex ||
          beamScene.mesh.tetrahedra[start + 2] === vertex ||
          beamScene.mesh.tetrahedra[start + 3] === vertex)
      );
    };
    const projectionDriverSampleCount =
      projectionPrecomputation.cubature.filter((sample: any) =>
        sampleContainsVertex(sample, driverVertex),
      ).length;
    if (projectionDriverSampleCount === 0) {
      throw new Error(
        "Sparse objective probe requires vertex 1 Cubature samples to " +
          "include a tetrahedron containing driver vertex 11.",
      );
    }
    const driverIncidentCount = beamInput.vertexInfo[driverVertex * 4 + 1]!;
    if (!(driverIncidentCount > 0)) {
      throw new Error("Sparse objective driver vertex 11 has no incident tet.");
    }
    const sourceShareLeverage =
      driverPrecomputation.cubature
        .filter((sample: any) => sampleContainsVertex(sample, driverVertex))
        .reduce(
          (sum: number, sample: any) => sum + Math.fround(sample.weight),
          0,
        ) / driverIncidentCount;

    const sparseExternalForces = new Float32Array(
      beamInput.vertexCount * 3,
    );
    const driverInertiaScale =
      beamInput.vertexRest[driverVertex * 4 + 3]! /
      (beamDefinition.settings.timestep *
        beamDefinition.settings.timestep);
    sparseExternalForces.set(
      [
        0.08 * driverInertiaScale,
        -0.06 * driverInertiaScale,
        0.04 * driverInertiaScale,
      ],
      driverVertex * 3,
    );
    const sparseSolver = await gpuApi.JGS2GpuSolver.create(
      device,
      {
        ...beamInput,
        objectives: { externalForces: sparseExternalForces },
      },
      {
        ...stepSettings,
        timestep: beamDefinition.settings.timestep,
      },
    );
    sparseSolver.stepExactIterations(1);
    await sparseSolver.awaitIdle();
    const sparseReadbacksBeforeDiagnostics =
      sparseSolver.explicitDiagnosticReadbackCount;
    const sparseDiagnostics =
      await sparseSolver.readGlobalizationDiagnostics();
    const sparsePredicted4 = await sparseSolver.readPredictedPositions();

    const beamPositions = new Float64Array(beamInput.vertexCount * 3);
    const beamPredicted = new Float64Array(beamInput.vertexCount * 3);
    const beamRestPositions = new Float64Array(beamInput.vertexCount * 3);
    const beamMasses = new Float64Array(beamInput.vertexCount);
    for (let vertex = 0; vertex < beamInput.vertexCount; vertex += 1) {
      beamMasses[vertex] = beamInput.vertexRest[vertex * 4 + 3]!;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        beamPositions[vertex * 3 + coordinate] =
          beamInput.positions[vertex * 4 + coordinate]!;
        beamPredicted[vertex * 3 + coordinate] =
          sparsePredicted4[vertex * 4 + coordinate]!;
        beamRestPositions[vertex * 3 + coordinate] =
          beamInput.vertexRest[vertex * 4 + coordinate]!;
      }
    }
    const beamInverseRest = new Float64Array(beamInput.tetCount * 9);
    const beamVolumes = new Float64Array(beamInput.tetCount);
    for (
      let tetrahedron = 0;
      tetrahedron < beamInput.tetCount;
      tetrahedron += 1
    ) {
      beamVolumes[tetrahedron] = beamInput.tetMeta[tetrahedron * 4]!;
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 3; column += 1) {
          beamInverseRest[tetrahedron * 9 + row * 3 + column] =
            beamInput.tetInverseDm[
              tetrahedron * 12 + column * 4 + row
            ]!;
        }
      }
    }
    const beamMirrorMesh = {
      ...beamScene.mesh,
      positions: beamRestPositions,
    };
    const beamMirrorRestData = {
      volumes: beamVolumes,
      inverseRestMatrices: beamInverseRest,
      stiffnessMatrices: Float64Array.from(beamInput.tetRestStiffness),
    };
    const sparseObjective = {
      externalForces: Float64Array.from(sparseExternalForces),
      targetPositions: beamPositions.slice(),
      targetStiffnesses: new Float64Array(beamInput.vertexCount),
    };
    const vectorDistance = (
      left: ArrayLike<number>,
      right: ArrayLike<number>,
    ) => {
      let squared = 0;
      for (let index = 0; index < left.length; index += 1) {
        squared += (left[index]! - right[index]!) ** 2;
      }
      return Math.sqrt(squared);
    };
    const sparseLocal = (sourceVertex: number, precomputation: any) => {
      const cpuContext = {
        mesh: beamMirrorMesh,
        restData: beamMirrorRestData,
        materials: beamScene.materials,
        lumpedMasses: beamMasses,
        restSystem: beamScene.restSystem,
        timestep: beamDefinition.settings.timestep,
        predictedPositions: beamPredicted,
        sourceVertex,
        exactBasis: Float64Array.from(
          Float32Array.from(precomputation.exactBasis),
        ),
      };
      const samples = precomputation.cubature.map((sample: any) => ({
        tetrahedron: sample.tetrahedron,
        weight: Math.fround(sample.weight),
        basisBlocks: Float64Array.from(
          Float32Array.from(sample.basisBlocks),
        ),
      }));
      const active = cpuApi.globalizeSelectedCubatureRestrictedLocal({
        context: cpuContext,
        positions: beamPositions,
        samples,
        objective: sparseObjective,
      });
      const absent = cpuApi.globalizeSelectedCubatureRestrictedLocal({
        context: cpuContext,
        positions: beamPositions,
        samples,
      });
      if (!active.direction.accepted) {
        throw new Error(
          `Sparse objective CPU direction failed for vertex ${sourceVertex}.`,
        );
      }
      if (!absent.direction.accepted) {
        throw new Error(
          `Sparse objective CPU baseline direction failed for vertex ${sourceVertex}.`,
        );
      }
      const effect = vectorDistance(
        active.direction.direction,
        absent.direction.direction,
      );
      const gpu = sparseDiagnostics.local[sourceVertex]!;
      return {
        status: gpu.status,
        effect,
        effectRelativeError:
          vectorDistance(gpu.direction, active.direction.direction) /
          Math.max(1e-12, effect),
      };
    };
    const driverLocal = sparseLocal(driverVertex, driverPrecomputation);
    const projectionLocal = sparseLocal(
      projectionVertex,
      projectionPrecomputation,
    );
    const sparseHistory = sparseDiagnostics.history[0]!;
    const sparseProbe = {
      driverVertex,
      projectionVertex,
      sourceShareLeverage,
      projectionDriverSampleCount,
      driverObjectiveEffect: driverLocal.effect,
      projectionObjectiveEffect: projectionLocal.effect,
      driverEffectRelativeError: driverLocal.effectRelativeError,
      projectionEffectRelativeError: projectionLocal.effectRelativeError,
      driverStatus: driverLocal.status,
      projectionStatus: projectionLocal.status,
      explicitReadbacksBeforeDiagnostics: sparseReadbacksBeforeDiagnostics,
      explicitReadbacks: sparseSolver.explicitDiagnosticReadbackCount,
      historyFinite: sparseHistory.finite,
      assembledAccepted: sparseHistory.assembledAccepted,
    };
    sparseSolver.destroy();
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    solver.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      explicitReadbacksBeforeDiagnostics,
      explicitReadbacks,
      local,
      externalForceNormRelativeError:
        Math.abs(history.componentGradientNorms.externalForce - expectedExternal) /
        Math.max(1, expectedExternal),
      targetNormRelativeError:
        Math.abs(history.componentGradientNorms.target - expectedTarget) /
        Math.max(1, expectedTarget),
      oracleExternalForceEnergyRelativeError:
        Math.abs(
          oracle.components.externalForce - expectedExternalEnergy,
        ) / Math.max(1, Math.abs(expectedExternalEnergy)),
      oracleTargetEnergyRelativeError:
        Math.abs(oracle.components.quadraticTarget - expectedTargetEnergy) /
        Math.max(1, Math.abs(expectedTargetEnergy)),
      oracleExternalForceNormRelativeError:
        Math.abs(
          oracle.componentGradientNorms.externalForce - expectedExternal,
        ) / Math.max(1, expectedExternal),
      oracleTargetNormRelativeError:
        Math.abs(oracle.componentGradientNorms.target - expectedTarget) /
        Math.max(1, expectedTarget),
      oracleExternalForceGradientRelativeError,
      oracleTargetGradientRelativeError,
      oracleTotalGradientRelativeError,
      oracleTargetHessianRelativeError,
      oracleFinite: oracle.finite,
      baselineOracleFinite: baselineOracle.finite,
      baselineExternalForceEnergy:
        baselineOracle.components.externalForce,
      baselineTargetEnergy: baselineOracle.components.quadraticTarget,
      baselineExternalForceGradientNorm:
        baselineOracle.componentGradientNorms.externalForce,
      baselineTargetGradientNorm:
        baselineOracle.componentGradientNorms.target,
      historyFinite: history.finite,
      assembledAccepted: history.assembledAccepted,
      acceptedMinimumDeterminant:
        history.acceptedMinimumDeformationDeterminant,
      sparseProbe,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.explicitReadbacksBeforeDiagnostics).toBe(0);
  expect(result.explicitReadbacks).toBe(4);
  expect(result.oracleFinite).toBe(true);
  expect(result.baselineOracleFinite).toBe(true);
  expect(result.baselineExternalForceEnergy).toBe(0);
  expect(result.baselineTargetEnergy).toBe(0);
  expect(result.baselineExternalForceGradientNorm).toBe(0);
  expect(result.baselineTargetGradientNorm).toBe(0);
  expect(result.historyFinite).toBe(true);
  expect(result.assembledAccepted).toBe(true);
  expect(result.acceptedMinimumDeterminant).toBeGreaterThan(1e-4);
  expect(result.externalForceNormRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.targetNormRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.oracleExternalForceEnergyRelativeError).toBeLessThanOrEqual(
    1e-3,
  );
  expect(result.oracleTargetEnergyRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.oracleExternalForceNormRelativeError).toBeLessThanOrEqual(
    1e-3,
  );
  expect(result.oracleTargetNormRelativeError).toBeLessThanOrEqual(1e-3);
  expect(
    result.oracleExternalForceGradientRelativeError,
  ).toBeLessThanOrEqual(1e-3);
  expect(result.oracleTargetGradientRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.oracleTotalGradientRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.oracleTargetHessianRelativeError).toBeLessThanOrEqual(1e-3);
  console.log(
    `P1 sparse objective probe: source leverage=${result.sparseProbe.sourceShareLeverage.toExponential(6)}, ` +
      `driver effect=${result.sparseProbe.driverObjectiveEffect.toExponential(6)}, ` +
      `driver error/effect=${result.sparseProbe.driverEffectRelativeError.toExponential(6)}, ` +
      `projection effect=${result.sparseProbe.projectionObjectiveEffect.toExponential(6)}, ` +
      `projection error/effect=${result.sparseProbe.projectionEffectRelativeError.toExponential(6)}, ` +
      `total-gradient error=${result.oracleTotalGradientRelativeError.toExponential(6)}.`,
  );
  expect(
    result.sparseProbe.sourceShareLeverage,
    "driver vertex 11 selected source-share leverage",
  ).toBeGreaterThan(0.5);
  expect(result.sparseProbe.projectionDriverSampleCount).toBeGreaterThan(0);
  expect(result.sparseProbe.driverStatus).toBe("accepted");
  expect(result.sparseProbe.projectionStatus).toBe("accepted");
  expect(result.sparseProbe.driverObjectiveEffect).toBeGreaterThan(1e-4);
  expect(result.sparseProbe.projectionObjectiveEffect).toBeGreaterThan(1e-4);
  expect(result.sparseProbe.driverEffectRelativeError).toBeLessThanOrEqual(
    1e-3,
  );
  expect(
    result.sparseProbe.projectionEffectRelativeError,
  ).toBeLessThanOrEqual(1e-3);
  expect(result.sparseProbe.explicitReadbacksBeforeDiagnostics).toBe(0);
  expect(result.sparseProbe.explicitReadbacks).toBe(2);
  expect(result.sparseProbe.historyFinite).toBe(true);
  expect(result.sparseProbe.assembledAccepted).toBe(true);
  for (const local of result.local) {
    expect(local.status, `vertex ${local.vertex}`).toBe("accepted");
    expect(
      local.referenceDirectionMagnitude,
      `vertex ${local.vertex} objective effect`,
    ).toBeGreaterThan(1e-3);
    expect(local.directionRelativeError, `vertex ${local.vertex}`).toBeLessThanOrEqual(1e-3);
    expect(local.alphaError, `vertex ${local.vertex}`).toBeLessThanOrEqual(1e-3);
    expect(local.acceptedEnergyDeltaError, `vertex ${local.vertex}`).toBeLessThanOrEqual(1e-3);
  }
});

test("P1-COMPOSITE-GPU: complete-objective Armijo, shift, and convergence policies hold", async () => {
  const result = await sharedPage.evaluate(async (fullQualification) => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2ObjectivesE2E: any }
    ).__jgs2ObjectivesE2E;
    const {
      cpuApi,
      gpuApi,
      adapter,
      device,
      uncapturedErrors,
      beamDefinition: definition,
      beamScene: scene,
      beamInput: input,
    } = shared;
    const corpus = cpuApi.buildNonlinearCubaturePoseCorpus({
      mesh: scene.mesh,
      restData: scene.restTetraData,
      materials: scene.materials,
      lumpedMasses: scene.lumpedMasses,
      restSystem: scene.restSystem,
    });
    const allPoses = [...corpus.training, ...corpus.validation];
    const poses = fullQualification
      ? allPoses
      : [
          corpus.training[0]!,
          corpus.training[7]!,
          corpus.training.at(-1)!,
          corpus.validation[0]!,
          corpus.validation[4]!,
          corpus.validation.at(-1)!,
        ];
    const mirrorRestPositions = new Float64Array(input.vertexCount * 3);
    const mirrorMasses = new Float64Array(input.vertexCount);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      mirrorMasses[vertex] = input.vertexRest[vertex * 4 + 3]!;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        mirrorRestPositions[vertex * 3 + coordinate] =
          input.vertexRest[vertex * 4 + coordinate]!;
      }
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
    const mirrorMesh = { ...scene.mesh, positions: mirrorRestPositions };
    const mirrorRestData = {
      volumes: mirrorVolumes,
      inverseRestMatrices: mirrorInverseRest,
      stiffnessMatrices: Float64Array.from(input.tetRestStiffness),
    };
    const packedPrecomputationByVertex = new Map(
      scene.vertexPrecomputations.map((entry: any) => [
        entry.vertex,
        {
          exactBasis: Float64Array.from(Float32Array.from(entry.exactBasis)),
          samples: entry.cubature.map((sample: any) => ({
            tetrahedron: sample.tetrahedron,
            weight: Math.fround(sample.weight),
            basisBlocks: Float64Array.from(
              Float32Array.from(sample.basisBlocks),
            ),
          })),
        },
      ]),
    );
    const relativeVectorError = (
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
    ) => {
      let squaredError = 0;
      let squaredReference = 0;
      for (let index = 0; index < expected.length; index += 1) {
        squaredError += (actual[index]! - expected[index]!) ** 2;
        squaredReference += expected[index]! ** 2;
      }
      return Math.sqrt(squaredError) / Math.max(1, Math.sqrt(squaredReference));
    };
    const vectorDistance = (
      left: ArrayLike<number>,
      right: ArrayLike<number>,
    ) => {
      let squared = 0;
      for (let index = 0; index < left.length; index += 1) {
        squared += (left[index]! - right[index]!) ** 2;
      }
      return Math.sqrt(squared);
    };
    const started = performance.now();
    const cases: any[] = [];
    device.pushErrorScope("validation");
    for (const [caseIndex, pose] of poses.entries()) {
      const positions = input.positions.slice();
      const velocities = new Float32Array(input.vertexCount * 4);
      const sourcePositions = new Float64Array(input.vertexCount * 3);
      const externalForces = new Float32Array(input.vertexCount * 3);
      const targetPositions = new Float32Array(input.vertexCount * 3);
      const targetStiffnesses = new Float32Array(input.vertexCount);
      const mode = ["force", "target", "combined"][caseIndex % 3]!;
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          const value = Math.fround(pose.positions[vertex * 3 + coordinate]!);
          const predicted = Math.fround(
            pose.predictedPositions[vertex * 3 + coordinate]!,
          );
          positions[vertex * 4 + coordinate] = value;
          sourcePositions[vertex * 3 + coordinate] = value;
          velocities[vertex * 4 + coordinate] = Math.fround(
            (predicted - value) / definition.settings.timestep,
          );
          targetPositions[vertex * 3 + coordinate] = value;
        }
        positions[vertex * 4 + 3] = 1;
      }
      for (const vertex of scene.restSystem.activeVertices) {
        const inertiaScale =
          input.vertexRest[vertex * 4 + 3]! /
          (definition.settings.timestep * definition.settings.timestep);
        if (mode !== "target") {
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            const phase =
              caseIndex * 0.37 + vertex * 0.71 + coordinate * 1.13;
            externalForces[vertex * 3 + coordinate] =
              0.008 * inertiaScale * Math.cos(phase);
          }
        }
        if (mode !== "force") {
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            const phase =
              caseIndex * 0.37 + vertex * 0.71 + coordinate * 1.13;
            targetPositions[vertex * 3 + coordinate] +=
              0.03 * Math.sin(phase + 0.23);
          }
          targetStiffnesses[vertex] = 0.35 * inertiaScale;
        }
      }
      const solver = await gpuApi.JGS2GpuSolver.create(
        device,
        {
          ...input,
          positions,
          velocities,
          objectives: { externalForces, targetPositions, targetStiffnesses },
        },
        {
          timestep: definition.settings.timestep,
          gravity: [0, 0, 0],
          iterations: 1,
          floorStiffness: 0,
          velocityDamping: 1,
          regularization: 1e-12,
          maxStep: 0,
          contactTangentialDamping: 0,
          contactMargin: 0,
          horizontalBodyCorrection: false,
          parityMode: true,
        },
      );
      solver.stepExactIterations(1);
      await solver.awaitIdle();
      const before = solver.explicitDiagnosticReadbackCount;
      const diagnostics = await solver.readGlobalizationDiagnostics();
      const predicted4 = await solver.readPredictedPositions();
      const snappedPredictedPositions = new Float64Array(
        input.vertexCount * 3,
      );
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          snappedPredictedPositions[vertex * 3 + coordinate] =
            predicted4[vertex * 4 + coordinate]!;
        }
      }
      const history = diagnostics.history[0]!;
      const objective64 = {
        externalForces: Float64Array.from(externalForces),
        targetPositions: Float64Array.from(targetPositions),
        targetStiffnesses: Float64Array.from(targetStiffnesses),
      };
      const local = Array.from(
        scene.restSystem.activeVertices,
        (vertex: number) => {
          const gpu = diagnostics.local[vertex]!;
          const precomputation = packedPrecomputationByVertex.get(vertex);
          if (!precomputation) {
            throw new Error(`Missing packed objective basis for vertex ${vertex}.`);
          }
          const cpuContext = {
            mesh: mirrorMesh,
            restData: mirrorRestData,
            materials: scene.materials,
            lumpedMasses: mirrorMasses,
            restSystem: scene.restSystem,
            timestep: definition.settings.timestep,
            predictedPositions: snappedPredictedPositions,
            sourceVertex: vertex,
            exactBasis: precomputation.exactBasis,
          };
          const cpu = cpuApi.globalizeSelectedCubatureRestrictedLocal({
            context: cpuContext,
            positions: sourcePositions,
            samples: precomputation.samples,
            objective: objective64,
          });
          if (!cpu.direction.accepted || !cpu.lineSearch?.accepted) {
            throw new Error(
              `${pose.id} vertex ${vertex} CPU objective globalization failed.`,
            );
          }
          const baseline = cpuApi.globalizeSelectedCubatureRestrictedLocal({
            context: cpuContext,
            positions: sourcePositions,
            samples: precomputation.samples,
          });
          if (!baseline.direction.accepted || !baseline.lineSearch?.accepted) {
            throw new Error(
              `${pose.id} vertex ${vertex} CPU baseline globalization failed.`,
            );
          }
          const cpuEnergyDelta =
            cpu.lineSearch.acceptedEnergy - cpu.lineSearch.initialEnergy;
          const objectiveDirectionEffect = vectorDistance(
            cpu.direction.direction,
            baseline.direction.direction,
          );
          return {
            ...gpu,
            vertex,
            directionRelativeError: relativeVectorError(
              gpu.direction,
              cpu.direction.direction,
            ),
            objectiveDirectionEffect,
            directionEffectRelativeError:
              vectorDistance(gpu.direction, cpu.direction.direction) /
              Math.max(1e-12, objectiveDirectionEffect),
            alphaRelativeError:
              Math.abs(gpu.alpha - cpu.lineSearch.alpha) /
              Math.max(1, Math.abs(cpu.lineSearch.alpha)),
            acceptedEnergyDeltaRelativeError:
              Math.abs(gpu.acceptedEnergyDelta - cpuEnergyDelta) /
              Math.max(1, Math.abs(cpuEnergyDelta)),
          };
        },
      );
      cases.push({
        id: pose.id,
        mode,
        before,
        after: solver.explicitDiagnosticReadbackCount,
        history,
        local,
      });
      solver.destroy();
    }
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    const local = cases.flatMap((entry) => entry.local);
    return {
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      elapsedMilliseconds: performance.now() - started,
      activeLocalCount: local.length,
      maximumDirectionRelativeError: Math.max(
        ...local.map((entry) => entry.directionRelativeError),
      ),
      maximumAlphaRelativeError: Math.max(
        ...local.map((entry) => entry.alphaRelativeError),
      ),
      maximumAcceptedEnergyDeltaRelativeError: Math.max(
        ...local.map((entry) => entry.acceptedEnergyDeltaRelativeError),
      ),
      minimumObjectiveDirectionEffect: Math.min(
        ...local.map((entry) => entry.objectiveDirectionEffect),
      ),
      maximumDirectionEffectRelativeError: Math.max(
        ...local.map((entry) => entry.directionEffectRelativeError),
      ),
      cases,
    };
  }, FULL_E2E_QUALIFICATION);

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.cases).toHaveLength(FULL_E2E_QUALIFICATION ? 24 : 6);
  expect(result.activeLocalCount).toBe(FULL_E2E_QUALIFICATION ? 240 : 60);
  expect(result.maximumDirectionRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.maximumAlphaRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.maximumAcceptedEnergyDeltaRelativeError).toBeLessThanOrEqual(
    1e-3,
  );
  expect(result.minimumObjectiveDirectionEffect).toBeGreaterThan(1e-4);
  expect(result.maximumDirectionEffectRelativeError).toBeLessThanOrEqual(
    1e-3,
  );
  console.log(
    `P1 composite GPU corpus: poses=${result.cases.length}, ` +
      `locals=${result.activeLocalCount}, ` +
      `modes=${[...new Set(result.cases.map((entry) => entry.mode))].join("/")}, ` +
      `max direction error=${result.maximumDirectionRelativeError.toExponential(6)}, ` +
      `max alpha error=${result.maximumAlphaRelativeError.toExponential(6)}, ` +
      `max energy-delta error=${result.maximumAcceptedEnergyDeltaRelativeError.toExponential(6)}, ` +
      `min objective effect=${result.minimumObjectiveDirectionEffect.toExponential(6)}, ` +
      `max effect-relative direction error=${result.maximumDirectionEffectRelativeError.toExponential(6)}.`,
  );
  for (const entry of result.cases) {
    expect(entry.before, entry.id).toBe(0);
    expect(entry.after, entry.id).toBe(2);
    expect(entry.history.finite, entry.id).toBe(true);
    expect(entry.history.energyValid, entry.id).toBe(true);
    expect(entry.history.assembledAccepted, entry.id).toBe(true);
    expect(entry.history.assembledReverted, entry.id).toBe(false);
    expect(entry.history.localFailureCount, entry.id).toBe(0);
    expect(entry.history.acceptedMinimumDeformationDeterminant, entry.id).toBeGreaterThan(1e-4);
    if (entry.mode !== "target") {
      expect(entry.history.componentGradientNorms.externalForce, entry.id).toBeGreaterThan(0);
    }
    if (entry.mode !== "force") {
      expect(entry.history.componentGradientNorms.target, entry.id).toBeGreaterThan(0);
    }
    for (const local of entry.local) {
      expect(local.finite, entry.id).toBe(true);
      expect(local.status, `${entry.id} vertex ${local.vertex}`).toBe("accepted");
      expect(local.normalizedShift, entry.id).toBeLessThanOrEqual(1e-3);
      expect(local.gradientDotDirection, entry.id).toBeLessThan(0);
      expect(local.acceptedEnergyDelta, entry.id).toBeLessThanOrEqual(local.armijoDeltaBound);
      expect(local.minimumTrialDeformationDeterminant, entry.id).toBeGreaterThan(1e-4);
      expect(local.directionRelativeError, entry.id).toBeLessThanOrEqual(1e-3);
      expect(local.objectiveDirectionEffect, entry.id).toBeGreaterThan(1e-4);
      expect(local.directionEffectRelativeError, entry.id).toBeLessThanOrEqual(1e-3);
      expect(local.alphaRelativeError, entry.id).toBeLessThanOrEqual(1e-3);
      expect(local.acceptedEnergyDeltaRelativeError, entry.id).toBeLessThanOrEqual(1e-3);
    }
  }
});

test("P1-COMPOSITE-GPU: mutable objective inputs fail closed and release without hidden state", async () => {
  const result = await sharedPage.evaluate(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2ObjectivesE2E: any }
    ).__jgs2ObjectivesE2E;
    const {
      cpuApi,
      gpuApi,
      adapter,
      device,
      uncapturedErrors,
      stableDefinition: definition,
      stableScene: scene,
      stableInput: input,
    } = shared;
    const settings = {
      timestep: definition.settings.timestep,
      gravity: [0, 0, 0] as [number, number, number],
      iterations: 1,
      floorStiffness: 0,
      velocityDamping: 1,
      regularization: 1e-12,
      maxStep: 0,
      contactTangentialDamping: 0,
      contactMargin: 0,
      horizontalBodyCorrection: false,
      parityMode: true,
    };
    device.pushErrorScope("validation");
    const solver = await gpuApi.JGS2GpuSolver.create(device, input, settings);
    const initialRevision = solver.objectiveRevision;
    const initialActivity = solver.objectiveActivity;
    const errors: string[] = [];
    const attempt = (operation: () => void) => {
      try {
        operation();
      } catch (error) {
        errors.push(error instanceof Error ? error.message : String(error));
      }
    };
    attempt(() => solver.setExternalForce(-1, [1, 0, 0]));
    attempt(() => solver.setExternalForce(0, [1, 0, 0]));
    attempt(() => solver.setExternalForce(1, [Number.NaN, 0, 0]));
    attempt(() => solver.setQuadraticTarget(1, [0, 0, 0], -1));
    attempt(() => solver.setQuadraticTarget(0, [0, 0, 0], 1));
    const invalidRevision = solver.objectiveRevision;
    const invalidActivity = solver.objectiveActivity;

    const source = 1;
    const targetA: [number, number, number] = [0.92, 0.08, 0.03];
    const targetB: [number, number, number] = [0.88, -0.06, 0.05];
    const stiffness = 600;
    solver.setQuadraticTarget(source, targetA, stiffness);
    solver.setQuadraticTarget(source, targetB, stiffness);
    const batchedRevisionBeforeStep = solver.objectiveRevision;
    const batchedActivityBeforeStep = solver.objectiveActivity;
    const batchedReadbacksBeforeStep = solver.explicitDiagnosticReadbackCount;
    solver.stepFrames(2, { iterations: 1 });
    await solver.awaitIdle();
    const batchedRevisionAfterStep = solver.objectiveRevision;
    const batchedActivityAfterStep = solver.objectiveActivity;
    const batchedReadbacksBeforeDiagnostics =
      solver.explicitDiagnosticReadbackCount;
    const targetedDiagnostics = await solver.readGlobalizationDiagnostics();
    const targetedPositions = await solver.readPositions();
    let expectedTargetSquared = 0;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      expectedTargetSquared +=
        (stiffness *
          (targetedPositions[source * 4 + coordinate]! - targetB[coordinate]!)) ** 2;
    }
    const expectedTargetNorm = Math.sqrt(expectedTargetSquared);
    const queuedTargetNorm =
      targetedDiagnostics.history[0]!.componentGradientNorms.target;
    const positionsBeforeRelease = await solver.readPositions();
    const velocitiesBeforeRelease = await solver.readVelocities();
    const readbacksBeforeRelease = solver.explicitDiagnosticReadbackCount;
    solver.releaseQuadraticTarget(source);
    const readbacksAfterRelease = solver.explicitDiagnosticReadbackCount;
    const releasedActivity = solver.objectiveActivity;
    const releasedRevision = solver.objectiveRevision;
    const positionsAfterRelease = await solver.readPositions();
    const velocitiesAfterRelease = await solver.readVelocities();
    const exactArrayEqual = (left: Float32Array, right: Float32Array) =>
      left.length === right.length && left.every((value, index) => Object.is(value, right[index]));

    const staleSource = 1;
    const activeSource = 2;
    solver.setQuadraticTarget(staleSource, [3e38, -3e38, 3e38], 0);
    solver.setQuadraticTarget(activeSource, [0.08, 0.9, -0.04], 125);
    solver.stepExactIterations(1);
    await solver.awaitIdle();
    const staleDiagnostics = await solver.readGlobalizationDiagnostics();
    const checkpointPositions = await solver.readPositions();
    const checkpointVelocities = await solver.readVelocities();
    const readbacksBeforeReleaseAll = solver.explicitDiagnosticReadbackCount;
    solver.releaseAllQuadraticTargets();
    const readbacksAfterReleaseAll = solver.explicitDiagnosticReadbackCount;
    const positionsAfterReleaseAll = await solver.readPositions();
    const velocitiesAfterReleaseAll = await solver.readVelocities();
    const finalActivity = solver.objectiveActivity;
    const finalRevision = solver.objectiveRevision;
    const control = await gpuApi.JGS2GpuSolver.create(
      device,
      {
        ...input,
        positions: checkpointPositions,
        velocities: checkpointVelocities,
      },
      settings,
    );
    solver.stepExactIterations(1);
    control.stepExactIterations(1);
    await Promise.all([solver.awaitIdle(), control.awaitIdle()]);
    const releasedDiagnostics = await solver.readGlobalizationDiagnostics();
    const [releasedPositions, releasedVelocities, controlPositions, controlVelocities] =
      await Promise.all([
        solver.readPositions(),
        solver.readVelocities(),
        control.readPositions(),
        control.readVelocities(),
      ]);

    const backtrackSource = 1;
    const precomputation = scene.vertexPrecomputations.find(
      (entry: { vertex: number }) => entry.vertex === backtrackSource,
    );
    if (!precomputation?.exactBasis) {
      throw new Error("Missing retained source basis for target backtrack.");
    }
    const sourcePositions = new Float64Array(input.vertexCount * 3);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        sourcePositions[vertex * 3 + coordinate] =
          input.positions[vertex * 4 + coordinate]!;
      }
    }
    const baseSystem = cpuApi.assembleStableNeoHookeanJGS2LocalSystem(
      {
        mesh: scene.mesh,
        restData: scene.restTetraData,
        materials: scene.materials,
        lumpedMasses: scene.lumpedMasses,
        restSystem: scene.restSystem,
        timestep: definition.settings.timestep,
        predictedPositions: sourcePositions,
        sourceVertex: backtrackSource,
        exactBasis: precomputation.exactBasis,
      },
      sourcePositions,
      [],
    );
    const desiredDirection = new Float64Array([-(1 - 5e-5), 0, 0]);
    const targetStiffness = 1e7;
    const backtrackTarget: [number, number, number] = [0, 0, 0];
    for (let row = 0; row < 3; row += 1) {
      let hessianProduct = targetStiffness * desiredDirection[row]!;
      for (let column = 0; column < 3; column += 1) {
        hessianProduct +=
          baseSystem.hessian[row * 3 + column]! * desiredDirection[column]!;
      }
      backtrackTarget[row] =
        sourcePositions[backtrackSource * 3 + row]! +
        hessianProduct / targetStiffness;
    }
    const vertexInfo = input.vertexInfo.slice();
    for (const vertex of [0, 2, 3]) {
      vertexInfo[vertex * 4 + 2] = 1;
    }
    const cubatureTetIds = new Uint32Array(input.cubatureTetIds.length);
    cubatureTetIds.fill(0xffffffff);
    const backtrackSolver = await gpuApi.JGS2GpuSolver.create(
      device,
      {
        ...input,
        vertexInfo,
        cubatureTetIds,
        cubatureWeights: new Float32Array(input.cubatureWeights.length),
        cubatureBasis: new Float32Array(input.cubatureBasis.length),
      },
      settings,
    );
    backtrackSolver.setQuadraticTarget(
      backtrackSource,
      backtrackTarget,
      targetStiffness,
    );
    backtrackSolver.stepExactIterations(1);
    await backtrackSolver.awaitIdle();
    const backtrackDiagnostics =
      await backtrackSolver.readGlobalizationDiagnostics();
    const backtrackLocal = backtrackDiagnostics.local[backtrackSource]!;
    const backtrackHistory = backtrackDiagnostics.history[0]!;
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    const result = {
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      initialRevision,
      initialActivity,
      initialActivityFrozen: Object.isFrozen(initialActivity),
      errors,
      invalidRevision,
      invalidActivity,
      batchedRevisionBeforeStep,
      batchedRevisionAfterStep,
      batchedActivityBeforeStep,
      batchedActivityAfterStep,
      batchedReadbacksBeforeStep,
      batchedReadbacksBeforeDiagnostics,
      queuedTargetNormRelativeError:
        Math.abs(queuedTargetNorm - expectedTargetNorm) /
        Math.max(1, expectedTargetNorm),
      releasedActivity,
      releasedRevision,
      releaseAddedReadbacks:
        readbacksAfterRelease - readbacksBeforeRelease,
      releasePreservedPositions: exactArrayEqual(
        positionsBeforeRelease,
        positionsAfterRelease,
      ),
      releasePreservedVelocities: exactArrayEqual(
        velocitiesBeforeRelease,
        velocitiesAfterRelease,
      ),
      staleRecordFinite: staleDiagnostics.history[0]!.finite,
      releaseAllPreservedPositions: exactArrayEqual(
        checkpointPositions,
        positionsAfterReleaseAll,
      ),
      releaseAllPreservedVelocities: exactArrayEqual(
        checkpointVelocities,
        velocitiesAfterReleaseAll,
      ),
      finalActivity,
      finalRevision,
      releaseAllAddedReadbacks:
        readbacksAfterReleaseAll - readbacksBeforeReleaseAll,
      releasedTargetNorm:
        releasedDiagnostics.history[0]!.componentGradientNorms.target,
      trajectoriesMatch:
        exactArrayEqual(releasedPositions, controlPositions) &&
        exactArrayEqual(releasedVelocities, controlVelocities),
      releasedHistoryFinite: releasedDiagnostics.history[0]!.finite,
      targetBacktrack: {
        status: backtrackLocal.status,
        alpha: backtrackLocal.alpha,
        backtrackCount: backtrackLocal.backtrackCount,
        energyEvaluationCount: backtrackLocal.energyEvaluationCount,
        gradientDotDirection: backtrackLocal.gradientDotDirection,
        acceptedEnergyDelta: backtrackLocal.acceptedEnergyDelta,
        armijoDeltaBound: backtrackLocal.armijoDeltaBound,
        minimumTrialDeterminant:
          backtrackLocal.minimumTrialDeformationDeterminant,
        assembledAccepted: backtrackHistory.assembledAccepted,
        assembledReverted: backtrackHistory.assembledReverted,
        finite: backtrackHistory.finite,
      },
    };
    backtrackSolver.destroy();
    control.destroy();
    solver.destroy();
    return result;
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.initialRevision).toBe(0);
  expect(result.initialActivityFrozen).toBe(true);
  expect(result.initialActivity.flags).toBe(0);
  expect(result.errors).toHaveLength(5);
  expect(result.invalidRevision).toBe(result.initialRevision);
  expect(result.invalidActivity).toEqual(result.initialActivity);
  expect(result.batchedRevisionBeforeStep).toBe(2);
  expect(result.batchedRevisionAfterStep).toBe(
    result.batchedRevisionBeforeStep,
  );
  expect(result.batchedActivityAfterStep).toEqual(
    result.batchedActivityBeforeStep,
  );
  expect(result.batchedReadbacksBeforeStep).toBe(0);
  expect(result.batchedReadbacksBeforeDiagnostics).toBe(0);
  expect(result.queuedTargetNormRelativeError).toBeLessThanOrEqual(1e-3);
  expect(result.releasedRevision).toBe(3);
  expect(result.releasedActivity.quadraticTargets).toBe(false);
  expect(result.releasedActivity.quadraticTargetVertexCount).toBe(0);
  expect(result.releasePreservedPositions).toBe(true);
  expect(result.releasePreservedVelocities).toBe(true);
  expect(result.releaseAddedReadbacks).toBe(0);
  expect(result.staleRecordFinite).toBe(true);
  expect(result.releaseAllPreservedPositions).toBe(true);
  expect(result.releaseAllPreservedVelocities).toBe(true);
  expect(result.releaseAllAddedReadbacks).toBe(0);
  expect(result.finalRevision).toBe(6);
  expect(result.finalActivity.flags).toBe(0);
  expect(result.finalActivity.quadraticTargetVertexCount).toBe(0);
  expect(result.releasedTargetNorm).toBe(0);
  expect(result.trajectoriesMatch).toBe(true);
  expect(result.releasedHistoryFinite).toBe(true);
  expect(result.targetBacktrack.status).toBe("accepted");
  expect(result.targetBacktrack.alpha).toBeGreaterThan(0);
  expect(result.targetBacktrack.alpha).toBeLessThan(1);
  expect(result.targetBacktrack.backtrackCount).toBeGreaterThanOrEqual(1);
  expect(result.targetBacktrack.energyEvaluationCount).toBeLessThan(
    result.targetBacktrack.backtrackCount + 1,
  );
  expect(result.targetBacktrack.gradientDotDirection).toBeLessThan(0);
  expect(result.targetBacktrack.acceptedEnergyDelta).toBeLessThanOrEqual(
    result.targetBacktrack.armijoDeltaBound,
  );
  expect(result.targetBacktrack.minimumTrialDeterminant).toBeGreaterThan(1e-4);
  expect(result.targetBacktrack.assembledAccepted).toBe(true);
  expect(result.targetBacktrack.assembledReverted).toBe(false);
  expect(result.targetBacktrack.finite).toBe(true);
});
