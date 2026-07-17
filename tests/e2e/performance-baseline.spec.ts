import { writeFile } from "node:fs/promises";

import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { buildSceneDefinition } from "../../src/scenes";
import {
  DEFAULT_JGS2_E2E_PERFORMANCE_OPTIONS,
  LIGHTWEIGHT_JGS2_E2E_TELEMETRY_OPTIONS,
  assessJGS2ComputeBudget,
  buildJGS2PerformanceBenchmark,
  type JGS2CpuFrameProfile,
  type JGS2GpuFrameProfile,
  type JGS2PerformanceProfileOptions,
} from "../../src/performance";
import { extractBoundarySurface } from "../../src/simulation/cpu";

const SCENE_ID = "stress";
const FULL_E2E_QUALIFICATION =
  process.env.JGS2_FULL_E2E === "1" ||
  process.env.npm_lifecycle_event === "test:e2e:full";
const PROFILE_OPTIONS = FULL_E2E_QUALIFICATION
  ? DEFAULT_JGS2_E2E_PERFORMANCE_OPTIONS
  : LIGHTWEIGHT_JGS2_E2E_TELEMETRY_OPTIONS;
const WARMUP_FRAME_COUNT = PROFILE_OPTIONS.warmupFrameCount;
const MEASURED_FRAME_COUNT = PROFILE_OPTIONS.measuredFrameCount;

interface PerformanceHarness {
  readonly ready: Promise<void>;
  profileCpuFrames(
    options: JGS2PerformanceProfileOptions,
  ): Promise<JGS2CpuFrameProfile>;
  profileGpuFrames(
    options: JGS2PerformanceProfileOptions,
  ): Promise<JGS2GpuFrameProfile>;
  profileCombinedFrames(options: JGS2PerformanceProfileOptions): Promise<{
    readonly cpuProfile: JGS2CpuFrameProfile;
    readonly gpuProfile: JGS2GpuFrameProfile;
  }>;
  diagnostics(): Promise<{
    readonly frame: number;
    readonly finite: boolean;
  }>;
  diagnosticReadbackCount(): number;
}

interface HarnessWindow extends Window {
  readonly __jgs2Test?: PerformanceHarness;
}

interface HardwareAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly isFallbackAdapter: boolean;
  readonly features: readonly string[];
  readonly limits: {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxStorageBuffersPerShaderStage: number;
    readonly maxComputeWorkgroupsPerDimension: number;
  };
}

interface BrowserEnvironment {
  readonly adapter: HardwareAdapterInfo;
  readonly userAgent: string;
  readonly platform: string;
  readonly logicalProcessorCount: number;
  readonly devicePixelRatio: number;
}

test.use({ trace: "off" });

test("Phase 0 stress-scene baseline records complete hardware frames", async ({
  page,
  browserName,
}, testInfo) => {
  test.setTimeout(120_000);

  const browserErrors: string[] = [];
  page.on("pageerror", (error) => {
    browserErrors.push(`Page error: ${error.message}`);
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`Console error: ${message.text()}`);
    }
  });

  await page.goto(`/?scene=${SCENE_ID}&test=1`, {
    waitUntil: "domcontentloaded",
  });
  const environment = await requireHardwareWebGPU(page, testInfo);
  await waitForTestHarness(page);

  // These diagnostic readbacks bracket the benchmark. The timed interval below
  // contains only complete one-frame harness steps and performance.now() calls.
  const readbacksBeforeInitialDiagnostics = await page.evaluate(() =>
    (window as HarnessWindow).__jgs2Test!.diagnosticReadbackCount(),
  );
  expect(readbacksBeforeInitialDiagnostics).toBe(0);
  const initial = await page.evaluate(async () => {
    return (window as HarnessWindow).__jgs2Test!.diagnostics();
  });
  const readbacksAfterInitialDiagnostics = await page.evaluate(() =>
    (window as HarnessWindow).__jgs2Test!.diagnosticReadbackCount(),
  );
  expect(readbacksAfterInitialDiagnostics).toBe(
    readbacksBeforeInitialDiagnostics + 2,
  );
  expect(initial.frame, "performance baseline must begin at frame zero").toBe(0);
  expect(initial.finite, "initial GPU state must be finite").toBe(true);

  const profileOptions = {
    warmupFrameCount: WARMUP_FRAME_COUNT,
    measuredFrameCount: MEASURED_FRAME_COUNT,
  } as const satisfies JGS2PerformanceProfileOptions;
  let cpuProfile: JGS2CpuFrameProfile;
  let gpuProfile: JGS2GpuFrameProfile;
  if (FULL_E2E_QUALIFICATION) {
    cpuProfile = await page.evaluate(
      async (options) =>
        (window as HarnessWindow).__jgs2Test!.profileCpuFrames(options),
      profileOptions,
    );

    // Replay from an identical fresh state so timestamp passes cannot perturb
    // or contaminate the formal wall/CPU sample interval.
    await page.reload({ waitUntil: "domcontentloaded" });
    await waitForTestHarness(page);
    gpuProfile = await page.evaluate(
      async (options) =>
        (window as HarnessWindow).__jgs2Test!.profileGpuFrames(options),
      profileOptions,
    );
  } else {
    const profiles = await page.evaluate(
      async (options) =>
        (window as HarnessWindow).__jgs2Test!.profileCombinedFrames(options),
      profileOptions,
    );
    cpuProfile = profiles.cpuProfile;
    gpuProfile = profiles.gpuProfile;
  }
  expect(cpuProfile.diagnosticReadbacksBefore).toBe(
    readbacksAfterInitialDiagnostics,
  );
  expect(cpuProfile.diagnosticReadbacksAfter).toBe(
    readbacksAfterInitialDiagnostics + 2,
  );

  const benchmark = buildJGS2PerformanceBenchmark(cpuProfile, gpuProfile);

  expect(
    benchmark.stateEquivalent,
    FULL_E2E_QUALIFICATION
      ? "timestamp replay must be byte-identical"
      : "combined smoke views must share the same final state",
  ).toBe(true);
  expect(browserErrors, "the baseline run must not report browser errors").toEqual([]);
  const adapterAdvertisedTimestampQuery =
    environment.adapter.features.includes("timestamp-query");
  expect(benchmark.gpuTimestamp.supported).toBe(
    adapterAdvertisedTimestampQuery,
  );
  expect(benchmark.gpuTimestamp.timestampMapCount).toBe(
    adapterAdvertisedTimestampQuery ? 1 : 0,
  );
  expect(
    [...cpuProfile.finalState.positions, ...cpuProfile.finalState.velocities].every(
      Number.isFinite,
    ),
    "final benchmark state must remain finite",
  ).toBe(true);

  const computeBudget = assessJGS2ComputeBudget(benchmark);

  const scene = buildSceneDefinition(SCENE_ID);
  const surface = extractBoundarySurface(scene.mesh);
  const bodyIds = new Set<number>(scene.mesh.bodyIds);
  const report = {
    schema: "org.jgs2.phase0-performance-baseline",
    schemaVersion: 3,
    criterion: "P0-EC-13",
    measurementTier: FULL_E2E_QUALIFICATION
      ? "qualification"
      : "quick-smoke",
    formalQualification: FULL_E2E_QUALIFICATION,
    samplingMode: FULL_E2E_QUALIFICATION
      ? "separate uninstrumented CPU/wall and GPU timestamp replays"
      : "single combined CPU/wall and GPU timestamp pass",
    cpuTimingIncludesTimestampInstrumentation:
      !FULL_E2E_QUALIFICATION && benchmark.gpuTimestamp.supported,
    stateEquivalenceMeaning: FULL_E2E_QUALIFICATION
      ? "Independent CPU/wall and GPU timestamp replays ended byte-identically."
      : "CPU and GPU views share one combined pass and one final state readback; this is not independent replay equivalence.",
    recordedAtUtc: new Date().toISOString(),
    project: {
      playwrightProject: testInfo.project.name,
      browserName,
      browserVersion: page.context().browser()?.version() ?? "unknown",
    },
    environment,
    workload: {
      sceneId: SCENE_ID,
      mode: "deterministic-test-harness",
      elementCounts: {
        vertices: scene.mesh.positions.length / 3,
        tetrahedra: scene.mesh.tetrahedra.length / 4,
        surfaceTriangles: surface.triangles.length / 3,
        surfaceEdges: surface.edges.length / 2,
        bodies: bodyIds.size,
        materials: scene.materials.length,
        fixedVertices: scene.mesh.fixed.reduce(
          (count, fixed) => count + Number(fixed !== 0),
          0,
        ),
      },
      timestepSeconds: scene.settings.timestep,
      solverIterationsPerFrame: scene.settings.solverIterations,
      cubatureSamplesPerVertex: scene.settings.cubatureSamples,
    },
    measurement: benchmark,
    validity: {
      initialFrame: initial.frame,
      cpuFinalFrame: cpuProfile.finalSimulationFrame,
      gpuFinalFrame: gpuProfile.finalSimulationFrame,
      initialFinite: initial.finite,
      finalFinite: true,
      browserErrors,
      diagnosticReadbacks: {
        beforeInitialDiagnostics: readbacksBeforeInitialDiagnostics,
        afterInitialDiagnostics: readbacksAfterInitialDiagnostics,
        cpuProfileBefore: cpuProfile.diagnosticReadbacksBefore,
        cpuProfileAfter: cpuProfile.diagnosticReadbacksAfter,
        gpuProfileBefore: gpuProfile.diagnosticReadbacksBefore,
        gpuProfileAfter: gpuProfile.diagnosticReadbacksAfter,
      },
      timestampMapCount: benchmark.gpuTimestamp.timestampMapCount,
      stateEquivalent: benchmark.stateEquivalent,
      performanceTargetApplied: true,
      necessaryComputeBudget: computeBudget,
    },
  };

  const reportPath = testInfo.outputPath("phase0-performance-baseline.json");
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await testInfo.attach("phase0-performance-baseline.json", {
    path: reportPath,
    contentType: "application/json",
  });

  console.log(
    `Phase 0 ${SCENE_ID} ${FULL_E2E_QUALIFICATION ? "qualification" : "quick-smoke"} baseline ` +
      `(${MEASURED_FRAME_COUNT} frames): ` +
      `serialized ${benchmark.endToEndFrame.averageFramesPerSecond!.toFixed(1)} FPS average, ` +
      `${benchmark.endToEndFrame.onePercentLowFramesPerSecond!.toFixed(1)} FPS 1% low; ` +
      `wall ${benchmark.endToEndFrame.meanMilliseconds.toFixed(3)} ms/frame ` +
      `(p95 ${benchmark.endToEndFrame.p95Milliseconds.toFixed(3)}); ` +
      `CPU ${benchmark.cpuFrameSubmission.meanMilliseconds.toFixed(3)} ms/frame, ` +
      `${benchmark.cpuSimulationSubmission.meanMilliseconds.toFixed(3)} ms/step; ` +
      (benchmark.gpuFrame
        ? `GPU ${benchmark.gpuFrame.meanMilliseconds.toFixed(3)} ms/frame, ` +
          `${benchmark.gpuSimulationStep!.meanMilliseconds.toFixed(3)} ms/step, ` +
          `${benchmark.gpuRender!.meanMilliseconds.toFixed(3)} ms/render; `
        : "GPU timestamp-query unsupported; ") +
      `step-compute budget ${computeBudget.serializedStepBudgetMilliseconds.toFixed(3)} ms ` +
      `(${computeBudget.passesNecessaryComputeBudget ? "PASS" : "FAIL"}; ` +
      `wall mean ${computeBudget.meetsSerializedWallMeanBudget ? "PASS" : "FAIL"}, ` +
      `GPU p95 ${computeBudget.meetsGpuFrameP95Budget === null ? "N/A" : computeBudget.meetsGpuFrameP95Budget ? "PASS" : "FAIL"})`,
  );
  expect(
    computeBudget.passesNecessaryComputeBudget,
    `stress compute budget: wall mean ${computeBudget.serializedWallMeanMilliseconds.toFixed(3)} ms, ` +
      `wall p95 ${computeBudget.serializedWallP95Milliseconds.toFixed(3)} ms (diagnostic), ` +
      `GPU p95 ${computeBudget.gpuFrameP95Milliseconds?.toFixed(3) ?? "unavailable"} ms, ` +
      `budget ${computeBudget.serializedStepBudgetMilliseconds.toFixed(3)} ms`,
  ).toBe(true);
});

async function waitForTestHarness(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const harness = (window as HarnessWindow).__jgs2Test;
    return (
      typeof harness?.profileCpuFrames === "function" &&
      typeof harness?.profileGpuFrames === "function" &&
      typeof harness?.profileCombinedFrames === "function" &&
      typeof harness?.diagnostics === "function" &&
      typeof harness?.diagnosticReadbackCount === "function"
    );
  });
  await page.evaluate(async () => {
    await (window as HarnessWindow).__jgs2Test!.ready;
  });
}

async function requireHardwareWebGPU(
  page: Page,
  testInfo: TestInfo,
): Promise<BrowserEnvironment> {
  const environment = await page.evaluate(async () => {
    const gpu = navigator.gpu;
    if (!gpu) {
      return null;
    }
    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });
    if (!adapter) {
      return null;
    }
    return {
      adapter: {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
        device: adapter.info.device,
        description: adapter.info.description,
        isFallbackAdapter: adapter.info.isFallbackAdapter,
        features: [...adapter.features].sort(),
        limits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize:
            adapter.limits.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage:
            adapter.limits.maxStorageBuffersPerShaderStage,
          maxComputeWorkgroupsPerDimension:
            adapter.limits.maxComputeWorkgroupsPerDimension,
        },
      },
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      logicalProcessorCount: navigator.hardwareConcurrency,
      devicePixelRatio: window.devicePixelRatio,
    } satisfies BrowserEnvironment;
  });

  if (!environment) {
    throw new Error(
      "Google Chrome could not create the hardware WebGPU adapter required for the Phase 0 baseline.",
    );
  }

  const adapterDescription = [
    environment.adapter.description,
    environment.adapter.vendor,
    environment.adapter.architecture,
    environment.adapter.device,
  ]
    .filter((detail, index, all) => detail && all.indexOf(detail) === index)
    .join(" / ");
  if (
    environment.adapter.isFallbackAdapter ||
    /swiftshader/i.test(adapterDescription)
  ) {
    throw new Error(
      `The Phase 0 performance baseline requires hardware WebGPU, but Chrome selected ${adapterDescription || "a software adapter"}.`,
    );
  }

  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapterDescription || "hardware adapter (details unavailable)",
  });
  return environment;
}
