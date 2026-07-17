import { expect, test } from "@playwright/test";
import type { BrowserContext, Page } from "@playwright/test";

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
  sharedPage.on("pageerror", (error) =>
    sharedBrowserErrors.push(error.message),
  );
  sharedPage.on("console", (message) => {
    if (message.type() === "error") sharedBrowserErrors.push(message.text());
  });
  await sharedPage.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });
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
      throw new Error("GPU globalization tests require hardware WebGPU.");
    }
    const device = await adapter.requestDevice();
    const uncapturedErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      uncapturedErrors.push(event.error.message);
    });
    const definition = phase1.buildPhase1StableNeoHookeanOracleDefinition();
    const scene = cpuApi.buildPrecomputedScene(definition, {
      retainExactBases: true,
    });
    const input = scenes.toJGS2GpuInput(scene);
    Object.assign(globalThis, {
      __jgs2GlobalizationE2E: {
        cpuApi,
        scenes,
        phase1,
        gpuApi,
        adapter,
        device,
        uncapturedErrors,
        definition,
        scene,
        input,
      },
    });
  });
});

test.beforeEach(async () => {
  sharedBrowserErrors.length = 0;
  await sharedPage.evaluate(() => {
    const shared = (
      globalThis as typeof globalThis & {
        __jgs2GlobalizationE2E: { uncapturedErrors: string[] };
      }
    ).__jgs2GlobalizationE2E;
    shared.uncapturedErrors.length = 0;
  });
});

test.afterAll(async () => {
  if (sharedPage) {
    await sharedPage.evaluate(async () => {
      const shared = (
        globalThis as typeof globalThis & {
          __jgs2GlobalizationE2E: { device: GPUDevice };
        }
      ).__jgs2GlobalizationE2E;
      await shared.device.queue.onSubmittedWorkDone();
      shared.device.destroy();
      Reflect.deleteProperty(globalThis, "__jgs2GlobalizationE2E");
    });
  }
  await sharedContext?.close();
});

interface AssembledCaseResult {
  readonly id: string;
  readonly positions: readonly number[];
  readonly historyCount: number;
  readonly sourceMinimum: number;
  readonly candidateMinimum: number;
  readonly acceptedMinimum: number;
  readonly sourceGeometryValid: boolean;
  readonly candidateGeometryValid: boolean;
  readonly acceptedMinimumValid: boolean;
  readonly localNumericsValid: boolean;
  readonly geometryValid: boolean;
  readonly accepted: boolean;
  readonly reverted: boolean;
  readonly revertCount: number;
  readonly localFailureCount: number;
  readonly maximumUpdate: number;
  readonly converged: boolean;
  readonly finite: boolean;
  readonly sourceEnergy: number;
  readonly candidateEnergy: number;
  readonly acceptedEnergy: number;
  readonly energyValid: boolean;
}

interface AssembledGpuResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly source: readonly number[];
  readonly candidateB: readonly number[];
  readonly candidateC: readonly number[];
  readonly combined: readonly number[];
  readonly nonfinite: readonly number[];
  readonly cases: readonly AssembledCaseResult[];
  readonly explicitReadbacks: number;
  readonly pinnedTarget: {
    readonly source: readonly number[];
    readonly rejectedTarget: readonly number[];
    readonly positions: readonly number[];
    readonly sourceMinimum: number;
    readonly candidateMinimum: number;
    readonly acceptedMinimum: number;
    readonly finalMinimum: number;
    readonly accepted: boolean;
    readonly reverted: boolean;
    readonly explicitReadbacks: number;
  };
}

test("P1-GPU-GLOBALIZATION: assembled Jacobi feasibility reverts the complete pose", async () => {
  const result = await sharedPage.evaluate<AssembledGpuResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2GlobalizationE2E: any }
    ).__jgs2GlobalizationE2E;
    const {
      adapter,
      device,
      uncapturedErrors,
      definition,
      scene,
      input,
    } = shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    device.pushErrorScope("validation");
    const solver = await shared.gpuApi.JGS2GpuSolver.create(device, input, {
      timestep: definition.settings.timestep,
      gravity: [0, 0, 0],
      iterations: 1,
      floorStiffness: 0,
      velocityDamping: 1,
      contactTangentialDamping: 0,
      contactMargin: 0,
      horizontalBodyCorrection: false,
      parityMode: true,
    });

    const source = input.positions.slice();
    const candidateB = source.slice();
    const candidateC = source.slice();
    const combined = source.slice();
    const nonfinite = source.slice();
    // Rest vertices are A=(0,0,0), B=(1,0,0), C=(0,1,0), D=(0,0,1).
    // Each shear alone retains J=1. Jacobi assembly produces 1-1.1^2=-0.21.
    candidateB[1 * 4 + 1] += 1.1;
    candidateC[2 * 4] += 1.1;
    combined[1 * 4 + 1] += 1.1;
    combined[2 * 4] += 1.1;
    // Every uploaded scalar remains finite, but the f32 edge subtraction
    // overflows. This reaches the production nonfinite determinant path
    // without bypassing the test helper's finite-input contract.
    nonfinite[0] = -3e38;
    nonfinite[1 * 4] = 3e38;

    const cases: AssembledCaseResult[] = [];
    for (const [id, candidate] of [
      ["candidate-b", candidateB],
      ["candidate-c", candidateC],
      ["combined", combined],
      ["nonfinite", nonfinite],
    ] as const) {
      const evaluated = await solver.evaluateAssembledCandidateForTest(
        source,
        candidate,
        {
          gravity: [0, 0, 0],
          floorStiffness: 0,
          horizontalBodyCorrection: false,
          parityMode: true,
        },
      );
      const history = evaluated.globalization.history[0]!;
      cases.push({
        id,
        positions: [...evaluated.positions],
        historyCount: evaluated.globalization.historyCount,
        sourceMinimum: history.sourceMinimumDeformationDeterminant,
        candidateMinimum: history.candidateMinimumDeformationDeterminant,
        acceptedMinimum: history.acceptedMinimumDeformationDeterminant,
        sourceGeometryValid: history.sourceGeometryValid,
        candidateGeometryValid: history.candidateGeometryValid,
        acceptedMinimumValid:
          history.acceptedMinimumDeformationDeterminantValid,
        localNumericsValid: history.localNumericsValid,
        geometryValid: history.geometryValid,
        accepted: history.assembledAccepted,
        reverted: history.assembledReverted,
        revertCount: history.revertCount,
        localFailureCount: history.localFailureCount,
        maximumUpdate: history.maximumUpdate,
        converged: history.converged,
        finite: history.finite,
        sourceEnergy: history.sourceEnergy,
        candidateEnergy: history.candidateEnergy,
        acceptedEnergy: history.acceptedEnergy,
        energyValid: history.energyValid,
      });
    }

    // Regression for the post-gate pinned-target hazard. This is a coherent
    // precomputation: the rest positions and inverse Dm remain unchanged. The
    // uploaded source is a feasible 180-degree rigid rotation with J=+1. A is
    // already fixed at its unchanged target; newly pinning only B proposes its
    // rest target (+1,0,0), while dynamic C and D have zero material/inertial
    // gradient at the rigid source. That complete candidate has J=-1. The
    // assembled gate must reject it, and finalize must preserve the reverted
    // source rather than snapping B again after the last feasibility check.
    const pinnedSource = input.positions.slice();
    pinnedSource.set([0, 0, 0, 1], 0);
    pinnedSource.set([-1, 0, 0, 1], 4);
    pinnedSource.set([0, -1, 0, 1], 8);
    pinnedSource.set([0, 0, 1, 1], 12);
    const pinnedInfo = input.vertexInfo.slice();
    pinnedInfo[1 * 4 + 2] = 1;
    const rejectedPinnedTarget = pinnedSource.slice();
    rejectedPinnedTarget.set(input.vertexRest.subarray(4, 7), 4);
    const pinnedSolver = await shared.gpuApi.JGS2GpuSolver.create(
      device,
      {
        ...input,
        positions: pinnedSource,
        velocities: new Float32Array(input.vertexCount * 4),
        vertexInfo: pinnedInfo,
      },
      {
        timestep: definition.settings.timestep,
        gravity: [0, 0, 0],
        iterations: 1,
        floorStiffness: 0,
        velocityDamping: 1,
        contactTangentialDamping: 0,
        contactMargin: 0,
        horizontalBodyCorrection: false,
        parityMode: true,
      },
    );
    pinnedSolver.stepExactIterations(1);
    await pinnedSolver.awaitIdle();
    const pinnedPositions = await pinnedSolver.readPositions();
    const pinnedDiagnostics =
      await pinnedSolver.readGlobalizationDiagnostics();
    const pinnedHistory = pinnedDiagnostics.history[0]!;
    const pinnedTarget = {
      source: [...pinnedSource],
      rejectedTarget: [...rejectedPinnedTarget],
      positions: [...pinnedPositions],
      sourceMinimum: pinnedHistory.sourceMinimumDeformationDeterminant,
      candidateMinimum: pinnedHistory.candidateMinimumDeformationDeterminant,
      acceptedMinimum: pinnedHistory.acceptedMinimumDeformationDeterminant,
      finalMinimum: shared.gpuApi.minimumJGS2InputDeformationDeterminant({
        ...input,
        positions: pinnedPositions,
      }),
      accepted: pinnedHistory.assembledAccepted,
      reverted: pinnedHistory.assembledReverted,
      explicitReadbacks: pinnedSolver.explicitDiagnosticReadbackCount,
    };
    pinnedSolver.destroy();

    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    const explicitReadbacks = solver.explicitDiagnosticReadbackCount;
    solver.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      source: [...source],
      candidateB: [...candidateB],
      candidateC: [...candidateC],
      combined: [...combined],
      nonfinite: [...nonfinite],
      cases,
      explicitReadbacks,
      pinnedTarget,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.cases).toHaveLength(4);
  for (const entry of result.cases) {
    expect(entry.historyCount, entry.id).toBe(1);
    expect(entry.sourceMinimum, entry.id).toBeCloseTo(1, 5);
    expect(entry.sourceGeometryValid, entry.id).toBe(true);
    expect(entry.acceptedMinimumValid, entry.id).toBe(true);
    expect(entry.localNumericsValid, entry.id).toBe(true);
    expect(entry.localFailureCount, entry.id).toBe(0);
  }

  const candidateB = result.cases[0]!;
  const candidateC = result.cases[1]!;
  for (const [entry, expected] of [
    [candidateB, result.candidateB],
    [candidateC, result.candidateC],
  ] as const) {
    expect(entry.candidateMinimum, entry.id).toBeCloseTo(1, 5);
    expect(entry.acceptedMinimum, entry.id).toBeCloseTo(1, 5);
    expect(entry.accepted, entry.id).toBe(true);
    expect(entry.reverted, entry.id).toBe(false);
    expect(entry.revertCount, entry.id).toBe(0);
    expect(entry.positions, entry.id).toEqual(expected);
    expect(entry.energyValid, entry.id).toBe(true);
    expect(entry.candidateGeometryValid, entry.id).toBe(true);
    expect(entry.geometryValid, entry.id).toBe(true);
    expect(entry.finite, entry.id).toBe(true);
    expect(entry.candidateEnergy, entry.id).toBeGreaterThan(
      entry.sourceEnergy,
    );
    expect(entry.acceptedEnergy, entry.id).toBe(entry.candidateEnergy);
  }

  const combined = result.cases[2]!;
  expect(combined.candidateMinimum).toBeCloseTo(-0.21, 5);
  expect(combined.candidateGeometryValid).toBe(true);
  expect(combined.geometryValid).toBe(true);
  expect(combined.acceptedMinimum).toBeCloseTo(1, 5);
  expect(combined.accepted).toBe(false);
  expect(combined.reverted).toBe(true);
  expect(combined.revertCount).toBe(1);
  expect(combined.positions).toEqual(result.source);
  expect(combined.positions).not.toEqual(result.combined);
  expect(combined.maximumUpdate).toBe(0);
  expect(combined.converged).toBe(false);
  expect(combined.energyValid).toBe(false);
  expect(combined.finite).toBe(true);

  const nonfinite = result.cases[3]!;
  expect(nonfinite.candidateMinimum).toBe(0);
  expect(nonfinite.candidateGeometryValid).toBe(false);
  expect(nonfinite.geometryValid).toBe(false);
  expect(nonfinite.acceptedMinimum).toBeCloseTo(1, 5);
  expect(nonfinite.accepted).toBe(false);
  expect(nonfinite.reverted).toBe(true);
  expect(nonfinite.revertCount).toBe(1);
  expect(nonfinite.positions).toEqual(result.source);
  expect(nonfinite.positions).not.toEqual(result.nonfinite);
  expect(nonfinite.maximumUpdate).toBe(0);
  expect(nonfinite.converged).toBe(false);
  expect(nonfinite.energyValid).toBe(false);
  expect(nonfinite.finite).toBe(false);
  expect(result.explicitReadbacks).toBe(8);

  expect(result.pinnedTarget.sourceMinimum).toBeCloseTo(1, 5);
  expect(result.pinnedTarget.candidateMinimum).toBeCloseTo(-1, 5);
  expect(result.pinnedTarget.acceptedMinimum).toBeCloseTo(1, 5);
  expect(result.pinnedTarget.finalMinimum).toBeCloseTo(
    result.pinnedTarget.acceptedMinimum,
    5,
  );
  expect(result.pinnedTarget.finalMinimum).toBeGreaterThan(1e-4);
  expect(result.pinnedTarget.accepted).toBe(false);
  expect(result.pinnedTarget.reverted).toBe(true);
  expect(result.pinnedTarget.positions).toEqual(result.pinnedTarget.source);
  expect(result.pinnedTarget.positions).not.toEqual(
    result.pinnedTarget.rejectedTarget,
  );
  expect(result.pinnedTarget.explicitReadbacks).toBe(2);
});

interface InitialSourcePreflightResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly hostMinimumDeterminant: number;
  readonly rejection: string;
}

test("P1-GPU-GLOBALIZATION: initial source uses production GPU-f32 feasibility preflight", async () => {
  const result = await sharedPage.evaluate<InitialSourcePreflightResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2GlobalizationE2E: any }
    ).__jgs2GlobalizationE2E;
    const { adapter, device, uncapturedErrors, definition, input, gpuApi } =
      shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    device.pushErrorScope("validation");
    const positions = input.positions.slice();
    positions.set([0, 0, 0, 1], 0);
    positions.set([1, 1, 1, 1], 4);
    positions.set([1, 1.0100129842758179, 1, 1], 8);
    positions.set([1, 0.00019997358322143555, 1.009993553161621, 1], 12);
    const adversarialInput = { ...input, positions };
    const hostMinimumDeterminant =
      gpuApi.minimumJGS2InputDeformationDeterminant(adversarialInput);
    let rejection = "";
    try {
      const solver = await gpuApi.JGS2GpuSolver.create(
        device,
        adversarialInput,
        {
          timestep: definition.settings.timestep,
          gravity: [0, 0, 0],
          floorStiffness: 0,
          horizontalBodyCorrection: false,
          parityMode: true,
        },
      );
      solver.destroy();
    } catch (error) {
      rejection = error instanceof Error ? error.message : String(error);
    }
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      hostMinimumDeterminant,
      rejection,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.hostMinimumDeterminant).toBeGreaterThan(1e-4);
  expect(result.rejection).toMatch(
    /not feasible under the production GPU f32 determinant calculation/i,
  );
});

interface FeasibilityFirstResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly status: string;
  readonly alpha: number;
  readonly backtrackCount: number;
  readonly energyEvaluationCount: number;
  readonly acceptedEnergyDelta: number;
  readonly armijoDeltaBound: number;
  readonly gradientDotDirection: number;
  readonly minimumTrialDeterminant: number;
  readonly solvedMinimumDeterminant: number;
  readonly assembledAccepted: boolean;
  readonly assembledReverted: boolean;
  readonly localFailureCount: number;
  readonly explicitReadbacks: number;
}

test("P1-GPU-GLOBALIZATION: local Armijo checks determinant feasibility before energy", async () => {
  const result = await sharedPage.evaluate<FeasibilityFirstResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2GlobalizationE2E: any }
    ).__jgs2GlobalizationE2E;
    const {
      cpuApi,
      adapter,
      device,
      uncapturedErrors,
      definition,
      scene,
      input,
    } = shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    device.pushErrorScope("validation");
    const sourceVertex = 1;
    const precomputation = scene.vertexPrecomputations.find(
      (entry) => entry.vertex === sourceVertex,
    );
    if (!precomputation?.exactBasis) {
      throw new Error("Missing retained source basis for Armijo fixture.");
    }
    const context = {
      mesh: scene.mesh,
      restData: scene.restTetraData,
      materials: scene.materials,
      lumpedMasses: scene.lumpedMasses,
      restSystem: scene.restSystem,
      timestep: definition.settings.timestep,
      predictedPositions: scene.mesh.positions,
      sourceVertex,
      exactBasis: precomputation.exactBasis,
    };
    const local = cpuApi.assembleStableNeoHookeanJGS2LocalSystem(
      context,
      scene.mesh.positions,
      [],
    );
    const desiredDirection = new Float64Array([-(1 - 5e-5), 0, 0]);
    const inertiaScale =
      scene.lumpedMasses[sourceVertex]! /
      (definition.settings.timestep * definition.settings.timestep);
    const desiredPredicted = new Float64Array(3);
    for (let row = 0; row < 3; row += 1) {
      let product = 0;
      for (let column = 0; column < 3; column += 1) {
        product +=
          local.hessian[row * 3 + column]! * desiredDirection[column]!;
      }
      desiredPredicted[row] =
        scene.mesh.positions[sourceVertex * 3 + row]! +
        product / inertiaScale;
    }

    // Isolate the source-exact production objective without retraining a
    // different private fixture: pin the other vertices in the uploaded ABI
    // and turn every complementary Cubature slot into zero-weight padding.
    const vertexInfo = input.vertexInfo.slice();
    for (const vertex of [0, 2, 3]) {
      vertexInfo[vertex * 4 + 2] = 1;
    }
    const cubatureTetIds = new Uint32Array(input.cubatureTetIds.length);
    cubatureTetIds.fill(0xffffffff);
    const cubatureWeights = new Float32Array(input.cubatureWeights.length);
    const cubatureBasis = new Float32Array(input.cubatureBasis.length);
    const velocities = new Float32Array(input.vertexCount * 4);
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const position = input.positions[sourceVertex * 4 + coordinate]!;
      velocities[sourceVertex * 4 + coordinate] = Math.fround(
        (desiredPredicted[coordinate]! - position) /
          definition.settings.timestep,
      );
    }
    const solver = await shared.gpuApi.JGS2GpuSolver.create(
      device,
      {
        ...input,
        velocities,
        vertexInfo,
        cubatureTetIds,
        cubatureWeights,
        cubatureBasis,
      },
      {
        timestep: definition.settings.timestep,
        gravity: [0, 0, 0],
        iterations: 1,
        floorStiffness: 0,
        velocityDamping: 1,
        contactTangentialDamping: 0,
        contactMargin: 0,
        horizontalBodyCorrection: false,
        parityMode: true,
      },
    );
    solver.stepExactIterations(1);
    await solver.awaitIdle();
    const positions4 = await solver.readPositions();
    const diagnostics = await solver.readGlobalizationDiagnostics();
    const localDiagnostic = diagnostics.local[sourceVertex]!;
    const history = diagnostics.history[0]!;
    const solvedPositions = new Float64Array(input.vertexCount * 3);
    for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        solvedPositions[vertex * 3 + coordinate] =
          positions4[vertex * 4 + coordinate]!;
      }
    }
    const material = cpuApi.evaluateStableNeoHookeanMesh(
      scene.mesh,
      scene.restTetraData,
      scene.materials,
      solvedPositions,
    );

    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    const explicitReadbacks = solver.explicitDiagnosticReadbackCount;
    solver.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      status: localDiagnostic.status,
      alpha: localDiagnostic.alpha,
      backtrackCount: localDiagnostic.backtrackCount,
      energyEvaluationCount: localDiagnostic.energyEvaluationCount,
      acceptedEnergyDelta: localDiagnostic.acceptedEnergyDelta,
      armijoDeltaBound: localDiagnostic.armijoDeltaBound,
      gradientDotDirection: localDiagnostic.gradientDotDirection,
      minimumTrialDeterminant:
        localDiagnostic.minimumTrialDeformationDeterminant,
      solvedMinimumDeterminant: Math.min(
        ...material.deformationDeterminants,
      ),
      assembledAccepted: history.assembledAccepted,
      assembledReverted: history.assembledReverted,
      localFailureCount: history.localFailureCount,
      explicitReadbacks,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.status).toBe("accepted");
  expect(result.alpha).toBeGreaterThan(0);
  expect(result.alpha).toBeLessThan(1);
  expect(result.backtrackCount).toBeGreaterThanOrEqual(1);
  // At least one geometric trial was skipped without evaluating its energy.
  expect(result.energyEvaluationCount).toBeLessThan(
    result.backtrackCount + 1,
  );
  expect(result.gradientDotDirection).toBeLessThan(0);
  expect(result.acceptedEnergyDelta).toBeLessThanOrEqual(
    result.armijoDeltaBound,
  );
  expect(result.minimumTrialDeterminant).toBeGreaterThan(1e-4);
  expect(result.solvedMinimumDeterminant).toBeGreaterThan(1e-4);
  expect(result.assembledAccepted).toBe(true);
  expect(result.assembledReverted).toBe(false);
  expect(result.localFailureCount).toBe(0);
  expect(result.explicitReadbacks).toBe(2);
});

interface FloorActiveResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly explicitReadbacksBeforeDiagnostics: number;
  readonly explicitReadbacks: number;
  readonly activeContactCount: number;
  readonly floorContactEnergy: number;
  readonly contactGradientNorm: number;
  readonly assembledAccepted: boolean;
  readonly assembledReverted: boolean;
  readonly historyFinite: boolean;
  readonly energyValid: boolean;
  readonly acceptedMinimumDeterminant: number;
  readonly acceptedEnergyRelativeError: number;
  readonly residualRelativeError: number;
  readonly acceptedLocals: readonly {
    readonly energyDelta: number;
    readonly armijoBound: number;
    readonly minimumDeterminant: number;
  }[];
}

test("P1-GPU-GLOBALIZATION: stable floor penalty participates in Armijo and convergence", async () => {
  const result = await sharedPage.evaluate<FloorActiveResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2GlobalizationE2E: any }
    ).__jgs2GlobalizationE2E;
    const {
      adapter,
      device,
      uncapturedErrors,
      definition,
      scene,
      input,
      gpuApi,
    } = shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    device.pushErrorScope("validation");
    const solver = await gpuApi.JGS2GpuSolver.create(device, input, {
      timestep: definition.settings.timestep,
      gravity: [0, 0, 0],
      iterations: 1,
      floorHeight: 0.25,
      floorStiffness: 2_500,
      velocityDamping: 1,
      contactTangentialDamping: 0,
      contactMargin: 0,
      horizontalBodyCorrection: false,
      parityMode: true,
    });
    solver.stepExactIterations(1);
    await solver.awaitIdle();
    const explicitReadbacksBeforeDiagnostics =
      solver.explicitDiagnosticReadbackCount;
    const diagnostics = await solver.readGlobalizationDiagnostics();
    const oracle = await solver.readOracleDiagnostics();
    const history = diagnostics.history[0]!;
    const acceptedLocals = Array.from(
      scene.restSystem.activeVertices,
      (vertex: number) => diagnostics.local[vertex]!,
    )
      .filter((entry: { status: string }) => entry.status === "accepted")
      .map(
        (entry: {
          acceptedEnergyDelta: number;
          armijoDeltaBound: number;
          minimumTrialDeformationDeterminant: number;
        }) => ({
          energyDelta: entry.acceptedEnergyDelta,
          armijoBound: entry.armijoDeltaBound,
          minimumDeterminant: entry.minimumTrialDeformationDeterminant,
        }),
      );
    const explicitReadbacks = solver.explicitDiagnosticReadbackCount;
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
      activeContactCount: oracle.activeContactCount,
      floorContactEnergy: oracle.components.floorContact,
      contactGradientNorm: history.componentGradientNorms.contact,
      assembledAccepted: history.assembledAccepted,
      assembledReverted: history.assembledReverted,
      historyFinite: history.finite,
      energyValid: history.energyValid,
      acceptedMinimumDeterminant:
        history.acceptedMinimumDeformationDeterminant,
      acceptedEnergyRelativeError: gpuApi.jgs2DiagnosticRelativeError(
        history.acceptedEnergy,
        oracle.energy,
      ),
      residualRelativeError: gpuApi.jgs2DiagnosticRelativeError(
        history.relativeResidual,
        oracle.relativeResidual,
      ),
      acceptedLocals,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.explicitReadbacksBeforeDiagnostics).toBe(0);
  expect(result.explicitReadbacks).toBe(2);
  expect(result.activeContactCount).toBeGreaterThan(0);
  expect(result.floorContactEnergy).toBeGreaterThan(0);
  expect(result.contactGradientNorm).toBeGreaterThan(0);
  expect(result.assembledAccepted).toBe(true);
  expect(result.assembledReverted).toBe(false);
  expect(result.historyFinite).toBe(true);
  expect(result.energyValid).toBe(true);
  expect(result.acceptedMinimumDeterminant).toBeGreaterThan(1e-4);
  expect(result.acceptedEnergyRelativeError).toBeLessThan(2e-5);
  expect(result.residualRelativeError).toBeLessThan(2e-5);
  expect(result.acceptedLocals.length).toBeGreaterThan(0);
  for (const local of result.acceptedLocals) {
    expect(local.energyDelta).toBeLessThanOrEqual(local.armijoBound);
    expect(local.minimumDeterminant).toBeGreaterThan(1e-4);
  }
});

interface ShiftParityCase {
  readonly id: string;
  readonly gpuStatus: number;
  readonly cpuStatus: string;
  readonly directionRelativeError: number;
  readonly normalizedEigenvalueError: number;
  readonly normalizedShiftError: number;
  readonly scaleRelativeError: number;
  readonly gradientDotDirectionRelativeError: number;
  readonly shiftedResidual: number;
}

interface ShiftParityResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly cases: readonly ShiftParityCase[];
}

test("P1-GPU-GLOBALIZATION: shared production shift solver matches the CPU policy", async () => {
  const result = await sharedPage.evaluate<ShiftParityResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2GlobalizationE2E: any }
    ).__jgs2GlobalizationE2E;
    const { cpuApi, gpuApi, adapter, device, uncapturedErrors } = shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    device.pushErrorScope("validation");

    const eta = gpuApi.JGS2_GLOBALIZATION_RELATIVE_EIGENVALUE_FLOOR;
    const definitions = [
      {
        id: "indefinite-accepted",
        hessian: new Float32Array([-5e-4, 0, 0, 0, 1, 0, 0, 0, 0.75]),
        gradient: new Float32Array([1, -2, 0.5]),
        inertia: Math.fround(1),
      },
      {
        id: "inside-shift-cap",
        hessian: new Float32Array([
          Math.fround(eta - 0.000999), 0, 0,
          0, 1, 0,
          0, 0, 1,
        ]),
        gradient: new Float32Array([1, -2, 0.5]),
        inertia: Math.fround(1),
      },
      {
        id: "outside-shift-cap",
        hessian: new Float32Array([
          Math.fround(eta - 0.001001), 0, 0,
          0, 1, 0,
          0, 0, 1,
        ]),
        gradient: new Float32Array([1, -2, 0.5]),
        inertia: Math.fround(1),
      },
      ...[1e-12, 1, 1e12].map((scale) => ({
        id: `scale-${scale}`,
        hessian: new Float32Array([
          -5e-4 * scale, 0, 0,
          0, scale, 0,
          0, 0, 0.75 * scale,
        ]),
        gradient: new Float32Array([scale, -2 * scale, 0.5 * scale]),
        inertia: Math.fround(scale),
      })),
    ];
    const input = new Float32Array(definitions.length * 16);
    for (const [index, definition] of definitions.entries()) {
      const base = index * 16;
      // Three padded WGSL matrix columns followed by gradient.xyz/inertia.
      input.set(definition.hessian.subarray(0, 3), base);
      input.set(definition.hessian.subarray(3, 6), base + 4);
      input.set(definition.hessian.subarray(6, 9), base + 8);
      input.set(definition.gradient, base + 12);
      input[base + 15] = definition.inertia;
    }
    const inputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputFloatCount = definitions.length * 12;
    const outputBuffer = device.createBuffer({
      size: outputFloatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: outputFloatCount * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(inputBuffer, 0, input);
    const wrapper = /* wgsl */ `
${gpuApi.jgs2GlobalizationWgsl}
@group(0) @binding(0) var<storage, read> rawInput: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> rawOutput: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${definitions.length}u) { return; }
  let inputBase = id.x * 4u;
  let outputBase = id.x * 3u;
  let hessian = mat3x3f(
    rawInput[inputBase].xyz,
    rawInput[inputBase + 1u].xyz,
    rawInput[inputBase + 2u].xyz,
  );
  let gradientInertia = rawInput[inputBase + 3u];
  let solved = jgs2_globalization_solve(
    hessian,
    gradientInertia.xyz,
    gradientInertia.w,
  );
  rawOutput[outputBase] = vec4f(
    solved.direction,
    solved.gradient_dot_direction,
  );
  rawOutput[outputBase + 1u] = vec4f(
    solved.scale,
    solved.minimum_eigenvalue_normalized,
    solved.normalized_shift,
    solved.shifted_relative_residual,
  );
  rawOutput[outputBase + 2u] = vec4f(
    f32(solved.status),
    solved.maximum_relative_asymmetry,
    0.0,
    0.0,
  );
}`;
    const module = device.createShaderModule({
      label: "jgs2-shared-globalization-shift-test",
      code: wrapper,
    });
    const compilation = await module.getCompilationInfo();
    const compilationErrors = compilation.messages
      .filter((message) => message.type === "error")
      .map((message) => message.message);
    if (compilationErrors.length > 0) {
      throw new Error(compilationErrors.join("\n"));
    }
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(
      outputBuffer,
      0,
      readback,
      0,
      outputFloatCount * 4,
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();

    const relativeScalarError = (actual: number, expected: number) =>
      Math.abs(actual - expected) / Math.max(1e-30, Math.abs(expected));
    const cases: ShiftParityCase[] = definitions.map((definition, index) => {
      const base = index * 12;
      const cpu = cpuApi.computeJGS2LocalDescentDirection(
        definition.hessian,
        definition.gradient,
        { inertiaScale: definition.inertia },
      );
      return {
        id: definition.id,
        gpuStatus: output[base + 8]!,
        cpuStatus: cpu.status,
        directionRelativeError: cpuApi.relativeError(
          output.subarray(base, base + 3),
          cpu.direction,
        ),
        normalizedEigenvalueError: Math.abs(
          output[base + 5]! - cpu.minimumEigenvalue / cpu.scale,
        ),
        normalizedShiftError: Math.abs(
          output[base + 6]! - cpu.normalizedShift,
        ),
        scaleRelativeError: relativeScalarError(output[base + 4]!, cpu.scale),
        gradientDotDirectionRelativeError:
          cpu.accepted && cpu.gradientDotDirection !== 0
            ? relativeScalarError(
                output[base + 3]!,
                cpu.gradientDotDirection,
              )
            : Math.abs(output[base + 3]!),
        shiftedResidual: output[base + 7]!,
      };
    });
    const validationError = await device.popErrorScope();
    inputBuffer.destroy();
    outputBuffer.destroy();
    readback.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      cases,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(result.cases).toHaveLength(6);
  for (const entry of result.cases) {
    const accepted = entry.cpuStatus === "accepted";
    expect(entry.gpuStatus, entry.id).toBe(
      accepted ? 0 : 3,
    );
    expect(entry.directionRelativeError, entry.id).toBeLessThan(2e-4);
    expect(entry.normalizedEigenvalueError, entry.id).toBeLessThan(2e-6);
    expect(entry.normalizedShiftError, entry.id).toBeLessThan(2e-6);
    expect(entry.scaleRelativeError, entry.id).toBeLessThan(2e-6);
    expect(entry.gradientDotDirectionRelativeError, entry.id).toBeLessThan(
      2e-4,
    );
    if (accepted) {
      expect(entry.shiftedResidual, entry.id).toBeLessThan(2e-5);
    } else {
      expect(entry.shiftedResidual, entry.id).toBe(0);
    }
  }
});

interface ConvergenceParityCase {
  readonly id: string;
  readonly vertexCount: number;
  readonly boundary: boolean;
  readonly relativeResidualError: number;
  readonly maximumUpdateError: number;
  readonly normalizedUpdateError: number;
  readonly denominatorError: number;
  readonly componentNormError: number;
  readonly gpuConverged: boolean;
  readonly cpuConverged: boolean;
  readonly gpuResidualSatisfied: boolean;
  readonly cpuResidualSatisfied: boolean;
  readonly gpuUpdateSatisfied: boolean;
  readonly cpuUpdateSatisfied: boolean;
  readonly gpuFinite: boolean;
  readonly cpuFinite: boolean;
  readonly deterministicRepeat: boolean;
}

interface ConvergenceParityResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly cases: readonly ConvergenceParityCase[];
  readonly dynamicSceneScale: number;
  readonly allVertexSceneScale: number;
  readonly explicitReadbacks: number;
}

test("P1-GPU-GLOBALIZATION: convergence reduction requires residual, update, feasibility, and no failures", async () => {
  const result = await sharedPage.evaluate<ConvergenceParityResult>(async () => {
    const shared = (
      globalThis as typeof globalThis & { __jgs2GlobalizationE2E: any }
    ).__jgs2GlobalizationE2E;
    const {
      cpuApi,
      adapter,
      device,
      uncapturedErrors,
      definition,
      scene,
      input,
    } = shared;
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    device.pushErrorScope("validation");
    const maximumVertexCount = 257;
    const positions = new Float32Array(maximumVertexCount * 4);
    const velocities = new Float32Array(maximumVertexCount * 4);
    const vertexRest = new Float32Array(maximumVertexCount * 4);
    const vertexColors = new Float32Array(maximumVertexCount * 4);
    const vertexInfo = new Uint32Array(maximumVertexCount * 4);
    positions.set(input.positions);
    if (input.velocities) velocities.set(input.velocities);
    vertexRest.set(input.vertexRest);
    vertexColors.set(input.vertexColors);
    vertexInfo.set(input.vertexInfo);
    for (let vertex = input.vertexCount; vertex < maximumVertexCount; vertex += 1) {
      const base = vertex * 4;
      positions.set([0, 0, 0, 1], base);
      vertexRest.set([0, 0, 0, 1], base);
      vertexColors.set([0.2, 0.2, 0.2, 1], base);
      vertexInfo.set([input.adjacency.length, 0, 1, 0], base);
    }
    // This disconnected pinned vertex is deliberately far outside the
    // dynamic body. Including it in the convergence scale would incorrectly
    // relax the maximum-update gate by six orders of magnitude.
    positions.set([1e6, -2e6, 3e6, 1], (maximumVertexCount - 1) * 4);
    vertexRest.set([1e6, -2e6, 3e6, 1], (maximumVertexCount - 1) * 4);
    const cubatureRecordCount = maximumVertexCount * input.cubatureK;
    const cubatureTetIds = new Uint32Array(cubatureRecordCount);
    cubatureTetIds.fill(0xffffffff);
    cubatureTetIds.set(input.cubatureTetIds);
    const cubatureWeights = new Float32Array(cubatureRecordCount);
    cubatureWeights.set(input.cubatureWeights);
    const cubatureBasis = new Float32Array(
      cubatureRecordCount * shared.gpuApi.JGS2_CUBATURE_BASIS_FLOATS,
    );
    cubatureBasis.set(input.cubatureBasis);
    const convergenceInput = {
      ...input,
      vertexCount: maximumVertexCount,
      positions,
      velocities,
      vertexRest,
      vertexColors,
      vertexInfo,
      cubatureTetIds,
      cubatureWeights,
      cubatureBasis,
    };
    const solver = await shared.gpuApi.JGS2GpuSolver.create(
      device,
      convergenceInput,
      {
      timestep: definition.settings.timestep,
      gravity: [0, 0, 0],
      floorStiffness: 0,
      residualTolerance: 1e-3,
      normalizedUpdateTolerance: 1e-3,
      horizontalBodyCorrection: false,
      parityMode: true,
      },
    );
    const dimension = convergenceInput.vertexCount * 3;
    const inertia = new Float32Array(dimension);
    const material = new Float32Array(dimension);
    const zero = new Float32Array(dimension);
    inertia[1 * 3] = 1;
    inertia[2 * 3 + 1] = 2;
    material[1 * 3] = -1;
    material[2 * 3 + 1] = -2;
    const smallUpdate = new Float32Array(dimension);
    smallUpdate[1 * 3] = 1e-5;
    const largeUpdate = smallUpdate.slice();
    largeUpdate[1 * 3] = 0.01;
    const highResidualMaterial = new Float32Array(dimension);
    const definitions = [
      {
        id: "converged-cancellation",
        material,
        updates: smallUpdate,
        assembledAccepted: true,
        assembledReverted: false,
        localFailureCount: 0,
      },
      {
        id: "high-residual",
        material: highResidualMaterial,
        updates: smallUpdate,
        assembledAccepted: true,
        assembledReverted: false,
        localFailureCount: 0,
      },
      {
        id: "high-update",
        material,
        updates: largeUpdate,
        assembledAccepted: true,
        assembledReverted: false,
        localFailureCount: 0,
      },
      {
        id: "assembled-revert",
        material,
        updates: smallUpdate,
        assembledAccepted: false,
        assembledReverted: true,
        localFailureCount: 0,
      },
      {
        id: "local-failure",
        material,
        updates: smallUpdate,
        assembledAccepted: true,
        assembledReverted: false,
        localFailureCount: 1,
      },
    ];
    const independentlyComputeScale = (dynamicOnly: boolean) => {
      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let minZ = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      let maxZ = Number.NEGATIVE_INFINITY;
      for (let vertex = 0; vertex < convergenceInput.vertexCount; vertex += 1) {
        const base = vertex * 4;
        if (dynamicOnly && convergenceInput.vertexInfo[base + 2] !== 0) {
          continue;
        }
        minX = Math.min(minX, convergenceInput.positions[base]!);
        minY = Math.min(minY, convergenceInput.positions[base + 1]!);
        minZ = Math.min(minZ, convergenceInput.positions[base + 2]!);
        maxX = Math.max(maxX, convergenceInput.positions[base]!);
        maxY = Math.max(maxY, convergenceInput.positions[base + 1]!);
        maxZ = Math.max(maxZ, convergenceInput.positions[base + 2]!);
      }
      return Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
    };
    const sceneScale = independentlyComputeScale(true);
    const allVertexSceneScale = independentlyComputeScale(false);
    const relativeError = (left: number, right: number) =>
      Math.abs(left - right) / Math.max(1, Math.abs(left), Math.abs(right));
    const cases: ConvergenceParityCase[] = [];
    const compareCase = (
      id: string,
      vertexCount: number,
      boundary: boolean,
      cpu: any,
      gpu: any,
      deterministicRepeat: boolean,
    ) => {
      const componentNormError = Math.max(
        ...[
          "inertia",
          "material",
          "externalForce",
          "target",
          "contact",
        ].map((key) =>
          relativeError(
            gpu.componentGradientNorms[key],
            cpu.componentGradientNorms[key],
          ),
        ),
      );
      cases.push({
        id,
        vertexCount,
        boundary,
        relativeResidualError: relativeError(
          gpu.relativeResidual,
          cpu.relativeResidual,
        ),
        maximumUpdateError: relativeError(
          gpu.maximumUpdate,
          cpu.maximumUpdate,
        ),
        normalizedUpdateError: relativeError(
          gpu.normalizedMaximumUpdate,
          cpu.normalizedMaximumUpdate,
        ),
        denominatorError: relativeError(
          gpu.residualDenominator,
          cpu.residualDenominator,
        ),
        componentNormError,
        gpuConverged: gpu.converged,
        cpuConverged: cpu.converged,
        gpuResidualSatisfied: gpu.residualSatisfied,
        cpuResidualSatisfied: cpu.residualSatisfied,
        gpuUpdateSatisfied: gpu.updateSatisfied,
        cpuUpdateSatisfied: cpu.updateSatisfied,
        gpuFinite: gpu.finite,
        cpuFinite: cpu.finite,
        deterministicRepeat,
      });
    };
    for (const definitionCase of definitions) {
      const gradients = {
        inertia,
        material: definitionCase.material,
        externalForce: zero,
        target: zero,
        contact: zero,
      };
      const cpu = cpuApi.evaluateJGS2Convergence({
        gradients,
        acceptedUpdates: definitionCase.updates,
        sceneScale,
        residualTolerance: 1e-3,
        normalizedUpdateTolerance: 1e-3,
        feasible: true,
        reverted: definitionCase.assembledReverted,
        localFailureCount: definitionCase.localFailureCount,
      });
      const gpu = await solver.evaluateConvergenceReductionForTest(
        {
          gradientComponents: [
            inertia,
            definitionCase.material,
            zero,
            zero,
            zero,
          ],
          acceptedUpdates: definitionCase.updates,
          assembledAccepted: definitionCase.assembledAccepted,
          assembledReverted: definitionCase.assembledReverted,
          localFailureCount: definitionCase.localFailureCount,
        },
        {
          residualTolerance: 1e-3,
          normalizedUpdateTolerance: 1e-3,
        },
      );
      compareCase(
        definitionCase.id,
        convergenceInput.vertexCount,
        false,
        cpu,
        gpu,
        true,
      );
    }
    for (const vertexCount of [1, 127, 128, 129, 257]) {
      const boundaryInertia = new Float32Array(dimension);
      const boundaryMaterial = new Float32Array(dimension);
      const boundaryUpdates = new Float32Array(dimension);
      const last = vertexCount - 1;
      boundaryInertia[last * 3] = 1.25;
      boundaryMaterial[last * 3] = -0.5;
      boundaryUpdates[last * 3 + 1] = 2e-5;
      const gradients = {
        inertia: boundaryInertia,
        material: boundaryMaterial,
        externalForce: zero,
        target: zero,
        contact: zero,
      };
      const cpu = cpuApi.evaluateJGS2Convergence({
        gradients,
        acceptedUpdates: boundaryUpdates,
        sceneScale,
        residualTolerance: 1e-3,
        normalizedUpdateTolerance: 1e-3,
        feasible: true,
        reverted: false,
        localFailureCount: 0,
      });
      const reductionInput = {
        gradientComponents: [
          boundaryInertia,
          boundaryMaterial,
          zero,
          zero,
          zero,
        ],
        acceptedUpdates: boundaryUpdates,
        assembledAccepted: true,
        assembledReverted: false,
        localFailureCount: 0,
        reductionVertexCount: vertexCount,
      };
      const first = await solver.evaluateConvergenceReductionForTest(
        reductionInput,
        {
          residualTolerance: 1e-3,
          normalizedUpdateTolerance: 1e-3,
        },
      );
      const second = await solver.evaluateConvergenceReductionForTest(
        reductionInput,
        {
          residualTolerance: 1e-3,
          normalizedUpdateTolerance: 1e-3,
        },
      );
      compareCase(
        `reduction-boundary-${vertexCount}`,
        vertexCount,
        true,
        cpu,
        first,
        JSON.stringify(first) === JSON.stringify(second),
      );
    }
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    const explicitReadbacks = solver.explicitDiagnosticReadbackCount;
    solver.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
      cases,
      dynamicSceneScale: sceneScale,
      allVertexSceneScale,
      explicitReadbacks,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(sharedBrowserErrors).toEqual([]);
  expect(result.dynamicSceneScale).toBeGreaterThan(1);
  expect(result.dynamicSceneScale).toBeLessThan(2);
  expect(result.allVertexSceneScale).toBeGreaterThan(1e6);
  expect(result.allVertexSceneScale / result.dynamicSceneScale).toBeGreaterThan(
    1e6,
  );
  expect(result.cases).toHaveLength(10);
  for (const entry of result.cases) {
    expect(entry.relativeResidualError, entry.id).toBeLessThan(2e-6);
    expect(entry.maximumUpdateError, entry.id).toBeLessThan(2e-6);
    expect(entry.normalizedUpdateError, entry.id).toBeLessThan(2e-6);
    expect(entry.denominatorError, entry.id).toBeLessThan(2e-6);
    expect(entry.componentNormError, entry.id).toBeLessThan(2e-6);
    expect(entry.gpuConverged, entry.id).toBe(entry.cpuConverged);
    expect(entry.gpuResidualSatisfied, entry.id).toBe(
      entry.cpuResidualSatisfied,
    );
    expect(entry.gpuUpdateSatisfied, entry.id).toBe(entry.cpuUpdateSatisfied);
    expect(entry.gpuFinite, entry.id).toBe(entry.cpuFinite);
    expect(entry.deterministicRepeat, entry.id).toBe(true);
  }
  expect(
    result.cases.filter((entry) => entry.boundary).map((entry) => entry.vertexCount),
  ).toEqual([1, 127, 128, 129, 257]);
  expect(result.cases[0]!.gpuConverged).toBe(true);
  expect(result.cases.slice(1).every((entry) => !entry.gpuConverged)).toBe(true);
  expect(result.explicitReadbacks).toBe(15);
});
