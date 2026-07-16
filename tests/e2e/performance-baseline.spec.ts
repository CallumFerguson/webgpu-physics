import { writeFile } from "node:fs/promises";

import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { buildSceneDefinition } from "../../src/scenes";
import { extractBoundarySurface } from "../../src/simulation/cpu";

const SCENE_ID = "stress";
const WARMUP_FRAME_COUNT = 120;
const MEASURED_FRAME_COUNT = 600;

interface PerformanceHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  timedStepFrame(): Promise<GpuTimestampMeasurement>;
  diagnostics(): Promise<{
    readonly frame: number;
    readonly finite: boolean;
  }>;
  diagnosticReadbackCount(): number;
}

interface GpuTimestampMeasurement {
  readonly feature: "timestamp-query";
  readonly supported: boolean;
  readonly featureEnabled: boolean;
  readonly gpuNanoseconds: number | null;
  readonly gpuMilliseconds: number | null;
  readonly reason: string | null;
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

test("Phase 0 stress-scene baseline records 600 complete hardware frames", async ({
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

  const benchmark = await page.evaluate(
    async ({ measuredFrameCount, warmupFrameCount }) => {
      const harness = (window as HarnessWindow).__jgs2Test!;

      for (let frame = 0; frame < warmupFrameCount; frame += 1) {
        await harness.stepFrames(1);
      }

      // Keep the query resolve/map outside the wall-clock sample interval.
      const gpuTimestamp = await harness.timedStepFrame();

      const samples: number[] = [];
      for (let frame = 0; frame < measuredFrameCount; frame += 1) {
        const start = performance.now();
        await harness.stepFrames(1);
        samples.push(performance.now() - start);
      }
      return { gpuTimestamp, latenciesMilliseconds: samples };
    },
    {
      measuredFrameCount: MEASURED_FRAME_COUNT,
      warmupFrameCount: WARMUP_FRAME_COUNT,
    },
  );
  const { gpuTimestamp, latenciesMilliseconds } = benchmark;

  const readbacksBeforeFinalDiagnostics = await page.evaluate(() =>
    (window as HarnessWindow).__jgs2Test!.diagnosticReadbackCount(),
  );
  expect(
    readbacksBeforeFinalDiagnostics,
    "normal simulation/render submissions must not perform diagnostic readback",
  ).toBe(readbacksAfterInitialDiagnostics);

  const final = await page.evaluate(async () => {
    return (window as HarnessWindow).__jgs2Test!.diagnostics();
  });
  const readbacksAfterFinalDiagnostics = await page.evaluate(() =>
    (window as HarnessWindow).__jgs2Test!.diagnosticReadbackCount(),
  );
  expect(
    readbacksAfterFinalDiagnostics,
    "an explicit diagnostics call must be visible in the readback counter",
  ).toBe(readbacksBeforeFinalDiagnostics + 2);
  expect(final.frame, "warm-up and measured frames must all complete").toBe(
    WARMUP_FRAME_COUNT + 1 + MEASURED_FRAME_COUNT,
  );
  expect(final.finite, "final GPU state must remain finite").toBe(true);

  expect(latenciesMilliseconds, "one latency is required per measured frame").toHaveLength(
    MEASURED_FRAME_COUNT,
  );
  expect(
    latenciesMilliseconds.every(
      (latency) => Number.isFinite(latency) && latency >= 0,
    ),
    "every wall-clock frame latency must be finite and nonnegative",
  ).toBe(true);
  expect(
    latenciesMilliseconds.some((latency) => latency > 0),
    "the browser clock must advance during the measured interval",
  ).toBe(true);
  expect(browserErrors, "the baseline run must not report browser errors").toEqual([]);

  const adapterAdvertisedTimestampQuery =
    environment.adapter.features.includes("timestamp-query");
  expect(gpuTimestamp.feature).toBe("timestamp-query");
  expect(gpuTimestamp.featureEnabled).toBe(adapterAdvertisedTimestampQuery);
  expect(gpuTimestamp.supported).toBe(adapterAdvertisedTimestampQuery);
  if (adapterAdvertisedTimestampQuery) {
    expect(gpuTimestamp.reason).toBeNull();
    expect(gpuTimestamp.gpuNanoseconds).not.toBeNull();
    expect(gpuTimestamp.gpuMilliseconds).not.toBeNull();
    expect(
      Number.isFinite(gpuTimestamp.gpuMilliseconds) &&
        gpuTimestamp.gpuMilliseconds! >= 0,
      "supported GPU timestamp must be finite and nonnegative",
    ).toBe(true);
  } else {
    expect(gpuTimestamp.gpuNanoseconds).toBeNull();
    expect(gpuTimestamp.gpuMilliseconds).toBeNull();
    expect(gpuTimestamp.reason).toContain("does not have");
  }

  const summary = summarizeLatencies(latenciesMilliseconds);
  expect(summary.minimum).toBeLessThanOrEqual(summary.p50);
  expect(summary.p50).toBeLessThanOrEqual(summary.p95);
  expect(summary.p95).toBeLessThanOrEqual(summary.p99);
  expect(summary.p99).toBeLessThanOrEqual(summary.maximum);

  const scene = buildSceneDefinition(SCENE_ID);
  const surface = extractBoundarySurface(scene.mesh);
  const bodyIds = new Set<number>(scene.mesh.bodyIds);
  const report = {
    schema: "org.jgs2.phase0-performance-baseline",
    schemaVersion: 2,
    criterion: "P0-EC-13",
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
    measurement: {
      timer: "window.performance.now",
      definition:
        "Wall-clock latency of await window.__jgs2Test.stepFrames(1), including one simulation step, rendering, and GPU queue completion.",
      warmupFrameCount: WARMUP_FRAME_COUNT,
      measuredFrameCount: MEASURED_FRAME_COUNT,
      firstMeasuredSimulationFrame: WARMUP_FRAME_COUNT + 2,
      lastMeasuredSimulationFrame:
        WARMUP_FRAME_COUNT + 1 + MEASURED_FRAME_COUNT,
      excludedFromTimedInterval: [
        "scene preprocessing",
        "diagnostics readback",
        "GPU timestamp probe frame, query resolve, and map",
        "screenshots",
        "Playwright tracing",
      ],
      latencyMilliseconds: {
        ...summary,
        samples: latenciesMilliseconds.map((latency, sampleIndex) => ({
          sampleIndex,
          simulationFrame: WARMUP_FRAME_COUNT + sampleIndex + 2,
          latency,
        })),
      },
      gpuSimulationTimestamp: {
        definition:
          "WebGPU timestamp-query interval between empty compute-pass boundaries immediately before and after one simulation frame command stream; rendering and readback are excluded.",
        probeSimulationFrame: WARMUP_FRAME_COUNT + 1,
        adapterAdvertised: adapterAdvertisedTimestampQuery,
        ...gpuTimestamp,
      },
    },
    validity: {
      initialFrame: initial.frame,
      finalFrame: final.frame,
      initialFinite: initial.finite,
      finalFinite: final.finite,
      browserErrors,
      diagnosticReadbacks: {
        beforeInitialDiagnostics: readbacksBeforeInitialDiagnostics,
        afterInitialDiagnostics: readbacksAfterInitialDiagnostics,
        beforeFinalDiagnostics: readbacksBeforeFinalDiagnostics,
        afterFinalDiagnostics: readbacksAfterFinalDiagnostics,
        unchangedDuringSimulationAndRendering:
          readbacksBeforeFinalDiagnostics === readbacksAfterInitialDiagnostics,
      },
      performanceTargetApplied: false,
    },
  };

  const reportPath = testInfo.outputPath("phase0-performance-baseline.json");
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await testInfo.attach("phase0-performance-baseline.json", {
    path: reportPath,
    contentType: "application/json",
  });

  console.log(
    `Phase 0 ${SCENE_ID} baseline (${MEASURED_FRAME_COUNT} frames): ` +
      `mean ${summary.mean.toFixed(3)} ms, p50 ${summary.p50.toFixed(3)} ms, ` +
      `p95 ${summary.p95.toFixed(3)} ms, p99 ${summary.p99.toFixed(3)} ms; ` +
      (gpuTimestamp.supported
        ? `GPU simulation ${gpuTimestamp.gpuMilliseconds!.toFixed(3)} ms`
        : "GPU timestamp-query unsupported"),
  );
});

async function waitForTestHarness(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const harness = (window as HarnessWindow).__jgs2Test;
    return (
      typeof harness?.stepFrames === "function" &&
      typeof harness?.timedStepFrame === "function" &&
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

function summarizeLatencies(latencies: readonly number[]): {
  readonly minimum: number;
  readonly maximum: number;
  readonly mean: number;
  readonly p50: number;
  readonly p95: number;
  readonly p99: number;
  readonly percentileMethod: "nearest-rank";
} {
  if (latencies.length === 0) {
    throw new Error("Cannot summarize an empty latency sample.");
  }
  const sorted = [...latencies].sort((left, right) => left - right);
  const percentile = (fraction: number): number => {
    const index = Math.max(0, Math.ceil(fraction * sorted.length) - 1);
    return sorted[index]!;
  };
  return {
    minimum: sorted[0]!,
    maximum: sorted.at(-1)!,
    mean:
      latencies.reduce((sum, latency) => sum + latency, 0) / latencies.length,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    percentileMethod: "nearest-rank",
  };
}
