import { inflateSync } from "node:zlib";
import { writeFile } from "node:fs/promises";

import {
  expect,
  test as base,
  type Page,
  type TestInfo,
} from "@playwright/test";

import {
  DEFAULT_JGS2_E2E_PERFORMANCE_OPTIONS,
  assessJGS2ComputeBudget,
  buildJGS2PerformanceBenchmark,
  type JGS2CpuFrameProfile,
  type JGS2GpuFrameProfile,
  type JGS2PerformanceProfileOptions,
} from "../../src/performance";

type Vec3 = readonly [number, number, number];

interface JGS2BodyDiagnostics {
  readonly bodyId: number;
  readonly mass: number;
  readonly centerOfMass: Vec3;
  readonly linearVelocity: Vec3;
  readonly linearMomentum: Vec3;
  readonly angularMomentum: Vec3;
  readonly minY: number;
}

interface JGS2Diagnostics {
  readonly frame: number;
  readonly finite: boolean;
  readonly source: "cpu-readback";
  readonly lastStepIterations: number;
  readonly runtime: {
    readonly parityMode: boolean;
    readonly velocityDamping: number;
    readonly contactTangentialDamping: number;
    readonly horizontalBodyCorrection: boolean;
  };
  readonly pinnedMaxError: number;
  readonly pinnedMaxErrorValid: boolean;
  readonly minTetDeterminant: number;
  readonly minTetDeterminantValid: boolean;
  readonly minimumContactDistance: number;
  readonly minimumContactDistanceValid: boolean;
  readonly activeContactCount: number;
  readonly activeContactCountValid: boolean;
  readonly candidateBufferOverflow: boolean;
  readonly candidateBufferOverflowValid: boolean;
  readonly relativeResidual: number;
  readonly relativeResidualValid: boolean;
  readonly maximumUpdate: number;
  readonly maximumUpdateValid: boolean;
  readonly totalLinearMomentum: Vec3;
  readonly totalLinearMomentumValid: boolean;
  readonly totalAngularMomentum: Vec3;
  readonly totalAngularMomentumValid: boolean;
  readonly floorHeight: number;
  readonly timestep: number;
  readonly bounds: {
    readonly min: Vec3;
    readonly max: Vec3;
  };
  readonly landmark: Vec3;
  readonly comparisonLandmark?: Vec3;
  readonly bodies: readonly JGS2BodyDiagnostics[];
}

interface JGS2TestHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  stepIterations(iterationCount: number): Promise<void>;
  profileCpuFrames(
    options: JGS2PerformanceProfileOptions,
  ): Promise<JGS2CpuFrameProfile>;
  profileGpuFrames(
    options: JGS2PerformanceProfileOptions,
  ): Promise<JGS2GpuFrameProfile>;
  waitForGpu(): Promise<void>;
  diagnostics(): Promise<JGS2Diagnostics>;
  configuration(): {
    readonly fixtureId: "phase0.force-free-cuboid" | null;
    readonly gravity: Vec3;
    readonly floorStiffness: number;
    readonly parityMode: boolean;
    readonly velocityDamping: number;
    readonly contactTangentialDamping: number;
    readonly horizontalBodyCorrection: boolean;
  };
  submissionPolicy(): {
    readonly maximumOutstanding: number;
    readonly currentOutstanding: number;
    readonly solverSubmissions: number;
    readonly renderSubmissions: number;
    readonly readbackSubmissions: number;
    readonly testBatchFrameLimit: number;
    readonly solverBatchFrameLimit: number;
    readonly productionStepsPerSubmission: 1;
  };
  focusOnPrimaryBody(): Promise<Vec3>;
  runForceFreeCorpus(): Promise<
    readonly {
      readonly id: string;
      readonly finite: boolean;
      readonly minimumDeterminant: number;
      readonly linearMomentumError: number;
      readonly angularMomentumError: number;
    }[]
  >;
}

declare global {
  interface Window {
    __jgs2Test?: JGS2TestHarness;
  }
}

interface DecodedPng {
  readonly width: number;
  readonly height: number;
  readonly channels: 3 | 4;
  readonly pixels: Uint8Array;
}

interface SceneCase {
  readonly id: "minimal" | "stiffness" | "drop" | "stress";
  readonly frames: number;
  readonly minChangedPixelRatio: number;
  readonly minLandmarkMotionFraction: number;
  readonly minBoundsMotionFraction: number;
}

const scenes = [
  {
    id: "minimal",
    frames: 45,
    minChangedPixelRatio: 0.001,
    minLandmarkMotionFraction: 0.003,
    minBoundsMotionFraction: 0.001,
  },
  {
    id: "stiffness",
    frames: 60,
    minChangedPixelRatio: 0.002,
    minLandmarkMotionFraction: 0.006,
    minBoundsMotionFraction: 0.003,
  },
  {
    id: "drop",
    frames: 90,
    minChangedPixelRatio: 0.005,
    minLandmarkMotionFraction: 0.05,
    minBoundsMotionFraction: 0.025,
  },
  {
    id: "stress",
    frames: 60,
    minChangedPixelRatio: 0.003,
    minLandmarkMotionFraction: 0.003,
    minBoundsMotionFraction: 0.001,
  },
] as const satisfies readonly SceneCase[];

const longRunScenes = [
  { id: "drop", expectedBodyCount: 1 },
  { id: "stress", expectedBodyCount: 6 },
] as const;

const test = base.extend<{ pageErrorGuard: void }>({
  pageErrorGuard: [
    async ({ page }, use) => {
      const errors: string[] = [];

      page.on("pageerror", (error) => {
        errors.push(`Page error: ${error.message}`);
      });
      page.on("console", (message) => {
        if (message.type() === "error") {
          errors.push(`Console error: ${message.text()}`);
        }
      });

      await use();

      expect(errors, "The page reported browser errors").toEqual([]);
    },
    { auto: true },
  ],
});

// These tests include benchmark intervals. Playwright tracing records page
// activity even when a successful trace is later discarded, so keep it out of
// the timing domain. Other E2E files retain the project-level failure traces.
test.use({ trace: "off" });

test("renders a useful DOM error when WebGPU is unavailable", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window.navigator, "gpu", {
      configurable: true,
      value: undefined,
    });
  });

  await page.goto("/?scene=minimal", { waitUntil: "domcontentloaded" });

  expect(await page.evaluate(() => navigator.gpu === undefined)).toBe(true);
  const root = page.locator("#root");
  await expect(root).toBeVisible();
  await expect(root).toContainText(/webgpu/i);
  await expect(root).toContainText(
    /unavailable|not supported|requires|required|error|failed/i,
  );
  const performanceHud = page.getByTestId("live-performance");
  await expect(performanceHud).toBeVisible();
  await expect(performanceHud).toHaveAttribute("data-status", "unavailable");
  await expect(performanceHud).toHaveAttribute("data-sample-count", "0");
  await expect(performanceHud).toHaveAttribute(
    "data-gpu-timing",
    "unavailable",
  );
  await expect(page.getByTestId("live-fps")).toHaveText("\u2014");
  await expect(page.getByTestId("live-cpu-step-ms")).toHaveText("\u2014");
  await expect(page.getByTestId("live-gpu-step-ms")).toHaveText("N/A");
});

test("shows live FPS and CPU/GPU timings in a production scene", async ({
  page,
}, testInfo) => {
  test.setTimeout(90_000);
  await page.goto("/?scene=minimal", { waitUntil: "domcontentloaded" });
  const adapterFeatures = await requireHardwareWebGPU(page, testInfo);

  const performanceHud = page.getByTestId("live-performance");
  await expect(performanceHud).toBeVisible();
  await expect
    .poll(
      async () =>
        Number((await performanceHud.getAttribute("data-sample-count")) ?? 0),
      { timeout: 30_000 },
    )
    .toBeGreaterThanOrEqual(100);
  await expect(performanceHud).toHaveAttribute("data-status", "ready");

  const readMetric = async (testId: string): Promise<number> => {
    const text = await page.getByTestId(testId).textContent();
    const value = Number(text);
    expect(Number.isFinite(value), `${testId} must be finite`).toBe(true);
    return value;
  };
  const averageFps = await readMetric("live-fps");
  const onePercentLowFps = await readMetric("live-one-percent-low");
  const frameMilliseconds = await readMetric("live-frame-ms");
  const simulationRate = await readMetric("live-simulation-rate");
  const cpuFrameMilliseconds = await readMetric("live-cpu-frame-ms");
  const cpuStepMilliseconds = await readMetric("live-cpu-step-ms");
  const cpuRenderMilliseconds = await readMetric("live-cpu-render-ms");
  expect(averageFps).toBeGreaterThan(0);
  expect(onePercentLowFps).toBeGreaterThan(0);
  expect(onePercentLowFps).toBeLessThanOrEqual(averageFps);
  expect(frameMilliseconds).toBeGreaterThan(0);
  expect(simulationRate).toBeGreaterThan(0);
  expect(cpuFrameMilliseconds).toBeGreaterThanOrEqual(0);
  expect(cpuStepMilliseconds).toBeGreaterThanOrEqual(0);
  expect(cpuRenderMilliseconds).toBeGreaterThanOrEqual(0);
  expect(Math.abs(averageFps - 1000 / frameMilliseconds)).toBeLessThan(0.2);
  expect(Math.abs(simulationRate - averageFps / 30)).toBeLessThan(0.02);
  expect(
    Math.abs(
      cpuFrameMilliseconds -
        (cpuStepMilliseconds + cpuRenderMilliseconds),
    ),
  ).toBeLessThan(0.004);

  if (adapterFeatures.includes("timestamp-query")) {
    await expect
      .poll(async () => performanceHud.getAttribute("data-gpu-timing"), {
        timeout: 15_000,
      })
      .toBe("available");
    expect(
      Number(
        (await performanceHud.getAttribute("data-gpu-sample-count")) ?? 0,
      ),
    ).toBeGreaterThanOrEqual(60);
    expect(await readMetric("live-gpu-frame-ms")).toBeGreaterThanOrEqual(0);
    expect(await readMetric("live-gpu-step-ms")).toBeGreaterThanOrEqual(0);
    expect(await readMetric("live-gpu-render-ms")).toBeGreaterThanOrEqual(0);
  } else {
    await expect(performanceHud).toHaveAttribute(
      "data-gpu-timing",
      "unavailable",
    );
    await expect(page.getByTestId("live-gpu-frame-ms")).toHaveText("N/A");
    await expect(page.getByTestId("live-gpu-step-ms")).toHaveText("N/A");
    await expect(page.getByTestId("live-gpu-render-ms")).toHaveText("N/A");
  }

  const firstUpdate = Number(
    (await performanceHud.getAttribute("data-update-sequence")) ?? 0,
  );
  await expect
    .poll(
      async () =>
        Number(
          (await performanceHud.getAttribute("data-update-sequence")) ?? 0,
        ),
    )
    .toBeGreaterThan(firstUpdate);

  const screenshotPath = testInfo.outputPath("live-performance-hud.png");
  const screenshot = await page.locator(".canvas-shell").screenshot({
    path: screenshotPath,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });
  expect(screenshot.byteLength).toBeGreaterThan(256);
  const decoded = decodePng(screenshot);
  expect(decoded.width).toBeGreaterThanOrEqual(128);
  expect(decoded.height).toBeGreaterThanOrEqual(128);
  await testInfo.attach("live-performance-hud.png", {
    body: screenshot,
    contentType: "image/png",
  });

  await page.setViewportSize({ width: 390, height: 844 });
  const canvasBounds = await page.getByTestId("gpu-canvas").boundingBox();
  const hudBounds = await performanceHud.boundingBox();
  expect(canvasBounds).not.toBeNull();
  expect(hudBounds).not.toBeNull();
  expect(hudBounds!.y).toBeGreaterThanOrEqual(
    canvasBounds!.y + canvasBounds!.height - 1,
  );
  const mobileScreenshot = await page.locator(".canvas-shell").screenshot({
    path: testInfo.outputPath("live-performance-hud-mobile.png"),
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });
  expect(mobileScreenshot.byteLength).toBeGreaterThan(256);
  await testInfo.attach("live-performance-hud-mobile.png", {
    body: mobileScreenshot,
    contentType: "image/png",
  });

  const updateBeforeVisibilityReset = Number(
    (await performanceHud.getAttribute("data-update-sequence")) ?? 0,
  );
  await page.evaluate(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect
    .poll(
      async () =>
        Number(
          (await performanceHud.getAttribute("data-update-sequence")) ?? 0,
        ),
    )
    .toBeGreaterThan(updateBeforeVisibilityReset);
  expect(
    Number((await performanceHud.getAttribute("data-sample-count")) ?? 0),
  ).toBeLessThan(20);
  await expect(performanceHud).toHaveAttribute("data-status", "collecting");
  await expect(performanceHud).toHaveAttribute("data-gpu-sample-count", "0");
});

test("parity mode disables project stabilizers and accepts an even exact iteration count", async ({
  page,
}, testInfo) => {
  await page.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });
  await requireHardwareWebGPU(page, testInfo);
  await waitForTestHarness(page);

  const deterministicHud = page.getByTestId("live-performance");
  await expect(deterministicHud).toBeHidden();
  await expect(deterministicHud).toHaveAttribute(
    "data-status",
    "test-controlled",
  );
  await expect(deterministicHud).toHaveAttribute("data-gpu-timing", "paused");
  await expect(deterministicHud).toHaveAttribute("data-sample-count", "0");
  await expect(deterministicHud).toHaveAttribute("data-update-sequence", "0");

  const initial = await readDiagnostics(page);
  assertDiagnostics(initial, 0, "minimal-parity");
  expect(initial.source).toBe("cpu-readback");
  expect(initial.runtime).toEqual({
    parityMode: true,
    velocityDamping: 1,
    contactTangentialDamping: 0,
    horizontalBodyCorrection: false,
  });
  expect(initial.minimumContactDistance).toBe(0);
  expect(initial.minimumContactDistanceValid).toBe(false);
  expect(initial.activeContactCount).toBe(0);
  expect(initial.activeContactCountValid).toBe(false);
  expect(initial.candidateBufferOverflow).toBe(false);
  expect(initial.candidateBufferOverflowValid).toBe(false);
  expect(initial.relativeResidual).toBe(0);
  expect(initial.relativeResidualValid).toBe(false);
  expect(initial.maximumUpdate).toBe(0);
  expect(initial.maximumUpdateValid).toBe(false);

  await page.evaluate(async () => {
    await window.__jgs2Test!.stepIterations(2);
    await window.__jgs2Test!.waitForGpu();
  });
  const after = await readDiagnostics(page);
  assertDiagnostics(after, 1, "minimal-parity");
  expect(after.lastStepIterations).toBe(2);
  expect(after.runtime.parityMode).toBe(true);
  expect(after.finite).toBe(true);
  await expect(deterministicHud).toHaveAttribute("data-sample-count", "0");
  await expect(deterministicHud).toHaveAttribute("data-update-sequence", "0");
});

test("P0-EC-10/11: a force-free body conserves momentum for 10 seconds", async ({
  page,
}, testInfo) => {
  const sceneId = "phase0.force-free-cuboid";
  await page.goto(`/?test=1&fixture=${sceneId}`, {
    waitUntil: "domcontentloaded",
  });
  await requireHardwareWebGPU(page, testInfo);
  await waitForTestHarness(page);
  await waitForCanvasPresentation(page);

  const configuration = await page.evaluate(() =>
    window.__jgs2Test!.configuration(),
  );
  expect(configuration).toEqual({
    fixtureId: sceneId,
    gravity: [0, 0, 0],
    floorStiffness: 0,
    parityMode: true,
    velocityDamping: 1,
    contactTangentialDamping: 0,
    horizontalBodyCorrection: false,
  });

  const initial = await readDiagnostics(page);
  assertDiagnostics(initial, 0, sceneId);
  assertBodyCount(initial, 1, sceneId);
  expect(initial.pinnedMaxErrorValid).toBe(false);
  expect(initial.pinnedMaxError).toBe(0);
  expect(initial.totalLinearMomentumValid).toBe(true);
  expect(initial.totalAngularMomentumValid).toBe(true);
  expect(vectorNorm(initial.totalLinearMomentum)).toBeGreaterThan(0);
  expect(vectorNorm(initial.bodies[0]!.angularMomentum)).toBeGreaterThan(0);
  const start = await captureCanvas(page, testInfo, "start.png");

  const finalFrame = frameAtSeconds(initial.timestep, 10, sceneId);
  await stepToFrame(page, 0, finalFrame);
  await page.evaluate(async () => window.__jgs2Test!.waitForGpu());
  await waitForCanvasPresentation(page);

  const final = await readDiagnostics(page);
  assertDiagnostics(final, finalFrame, sceneId);
  expect(final.lastStepIterations).toBe(12);
  expect(final.pinnedMaxErrorValid).toBe(false);
  expect(final.pinnedMaxError).toBe(0);
  expect(final.totalLinearMomentumValid).toBe(true);
  expect(final.totalAngularMomentumValid).toBe(true);
  const end = await captureCanvas(page, testInfo, "end.png");
  const focusedTarget = await page.evaluate(async () =>
    window.__jgs2Test!.focusOnPrimaryBody(),
  );
  await waitForCanvasPresentation(page);
  const focusedEnd = await captureCanvas(
    page,
    testInfo,
    "end-follow-camera.png",
  );
  expect(
    distance(focusedTarget, final.bodies[0]!.centerOfMass),
    "the test-only follow camera must target the final body center",
  ).toBeLessThan(1e-6);
  expect(
    comparePngs(end, focusedEnd).changedPixelRatio,
    "the follow-camera artifact must visibly reveal the translated final body",
  ).toBeGreaterThan(0.0005);

  const totalMass = initial.bodies.reduce((sum, body) => sum + body.mass, 0);
  const sceneScale = boundsDiagonal(initial);
  const linearError = vectorDifferenceNorm(
    final.totalLinearMomentum,
    initial.totalLinearMomentum,
  ) / Math.max(
    vectorNorm(initial.totalLinearMomentum),
    totalMass * sceneScale,
  );
  const angularError = vectorDifferenceNorm(
    final.totalAngularMomentum,
    initial.totalAngularMomentum,
  ) / Math.max(
    vectorNorm(initial.totalAngularMomentum),
    totalMass * sceneScale * sceneScale,
  );

  console.log(
    `${sceneId}: normalized linear momentum error=${linearError.toExponential(6)}, ` +
      `angular momentum error=${angularError.toExponential(6)}, ` +
      `minimum determinant=${final.minTetDeterminant.toExponential(6)}`,
  );
  expect(
    linearError,
    "force-free normalized total linear-momentum error after 10 seconds",
  ).toBeLessThan(0.005);
  expect(
    angularError,
    "force-free normalized total angular-momentum error after 10 seconds",
  ).toBeLessThan(0.005);
  expect(
    comparePngs(start, end).changedPixelRatio,
    "the translating and rotating conservation fixture must visibly move",
  ).toBeGreaterThan(0.0005);

  const submissionPolicy = await page.evaluate(() =>
    window.__jgs2Test!.submissionPolicy(),
  );
  expect(submissionPolicy.maximumOutstanding).toBeLessThanOrEqual(2);
  expect(submissionPolicy.currentOutstanding).toBe(0);
  expect(submissionPolicy.testBatchFrameLimit).toBe(120);
  expect(submissionPolicy.solverBatchFrameLimit).toBeGreaterThanOrEqual(
    submissionPolicy.testBatchFrameLimit,
  );
  expect(submissionPolicy.productionStepsPerSubmission).toBe(1);
  await recordScenePerformance(page, testInfo, sceneId);
});

test("P0-CORPUS-01: all 32 frozen force-free states conserve momentum on hardware", async ({
  page,
}, testInfo) => {
  test.setTimeout(600_000);
  const sceneId = "phase0.force-free-cuboid";
  await page.goto(`/?test=1&fixture=${sceneId}`, {
    waitUntil: "domcontentloaded",
  });
  await requireHardwareWebGPU(page, testInfo);
  await waitForTestHarness(page);

  const started = Date.now();
  const results = await page.evaluate(async () =>
    window.__jgs2Test!.runForceFreeCorpus(),
  );
  const elapsedMilliseconds = Date.now() - started;
  expect(results).toHaveLength(32);
  expect(new Set(results.map((result) => result.id)).size).toBe(32);

  let worstLinear = results[0]!;
  let worstAngular = results[0]!;
  let minimumDeterminant = Infinity;
  for (const result of results) {
    expect(result.finite, `${result.id} must remain finite`).toBe(true);
    expect(
      result.minimumDeterminant,
      `${result.id} must not invert at one-second checkpoints`,
    ).toBeGreaterThan(0);
    expect(
      result.linearMomentumError,
      `${result.id} normalized linear momentum error`,
    ).toBeLessThanOrEqual(0.005);
    expect(
      result.angularMomentumError,
      `${result.id} normalized angular momentum error`,
    ).toBeLessThanOrEqual(0.005);
    if (result.linearMomentumError > worstLinear.linearMomentumError) {
      worstLinear = result;
    }
    if (result.angularMomentumError > worstAngular.angularMomentumError) {
      worstAngular = result;
    }
    minimumDeterminant = Math.min(
      minimumDeterminant,
      result.minimumDeterminant,
    );
  }

  const submissionPolicy = await page.evaluate(() =>
    window.__jgs2Test!.submissionPolicy(),
  );
  expect(submissionPolicy.maximumOutstanding).toBeLessThanOrEqual(2);
  expect(submissionPolicy.currentOutstanding).toBe(0);
  console.log(
    `Phase 0 force-free corpus: 32 cases x 1200 frames in ` +
      `${elapsedMilliseconds} ms; worst linear=${worstLinear.linearMomentumError.toExponential(6)} ` +
      `(${worstLinear.id}), worst angular=${worstAngular.angularMomentumError.toExponential(6)} ` +
      `(${worstAngular.id}), minimum determinant=${minimumDeterminant.toExponential(6)}`,
  );
  await testInfo.attach("phase0-force-free-corpus.json", {
    body: Buffer.from(
      JSON.stringify(
        { elapsedMilliseconds, submissionPolicy, results },
        null,
        2,
      ),
    ),
    contentType: "application/json",
  });
});

test.describe("deterministic JGS2 scenes", () => {
  for (const scene of scenes) {
    test(`${scene.id}: frame 0 to frame ${scene.frames}`, async ({ page }, testInfo) => {
      await page.goto(`/?scene=${scene.id}&test=1`, {
        waitUntil: "domcontentloaded",
      });
      await requireHardwareWebGPU(page, testInfo);
      await waitForTestHarness(page);
      await waitForCanvasPresentation(page);

      const startDiagnostics = await readDiagnostics(page);
      assertDiagnostics(startDiagnostics, 0, scene.id);
      expect(startDiagnostics.pinnedMaxErrorValid).toBe(
        scene.id === "minimal" || scene.id === "stiffness",
      );
      const start = await captureCanvas(page, testInfo, "start.png");
      if (scene.id === "minimal") {
        const pageScreenshot = await page.screenshot({
          path: testInfo.outputPath("page-start.png"),
          fullPage: true,
          animations: "disabled",
          caret: "hide",
          scale: "css",
        });
        await testInfo.attach("page-start.png", {
          body: pageScreenshot,
          contentType: "image/png",
        });
      }

      await page.evaluate(async (frameCount) => {
        await window.__jgs2Test!.stepFrames(frameCount);
      }, scene.frames);
      await waitForCanvasPresentation(page);

      const endDiagnostics = await readDiagnostics(page);
      assertDiagnostics(endDiagnostics, scene.frames, scene.id);
      const end = await captureCanvas(page, testInfo, "end.png");

      const visualDifference = comparePngs(start, end);
      expect(
        visualDifference.changedPixelRatio,
        `${scene.id} must visibly change between frame 0 and frame ${scene.frames}`,
      ).toBeGreaterThan(scene.minChangedPixelRatio);

      const initialDiagonal = boundsDiagonal(startDiagnostics);
      const landmarkMotion = distance(
        startDiagnostics.landmark,
        endDiagnostics.landmark,
      );
      const downwardLandmarkMotion =
        startDiagnostics.landmark[1] - endDiagnostics.landmark[1];
      const boundsMotion = maximumBoundsMotion(
        startDiagnostics,
        endDiagnostics,
      );

      expect(
        downwardLandmarkMotion / initialDiagonal,
        `${scene.id} landmark must move downward by a visible fraction of the scene`,
      ).toBeGreaterThan(scene.minLandmarkMotionFraction);
      expect(
        landmarkMotion / initialDiagonal,
        `${scene.id} landmark motion must remain bounded`,
      ).toBeLessThan(2);
      expect(
        boundsMotion / initialDiagonal,
        `${scene.id} dynamic bounds must change as the body deforms or moves`,
      ).toBeGreaterThan(scene.minBoundsMotionFraction);
      expect(
        endDiagnostics.minTetDeterminant /
          startDiagnostics.minTetDeterminant,
        `${scene.id} tetrahedra must retain at least 1% of their initial minimum determinant`,
      ).toBeGreaterThan(0.01);
      expect(
        endDiagnostics.floorHeight,
        `${scene.id} floor height must remain constant`,
      ).toBe(startDiagnostics.floorHeight);

      if (scene.id === "drop" || scene.id === "stress") {
        expect(
          endDiagnostics.bounds.min[1],
          `${scene.id} must not pass materially through the floor`,
        ).toBeGreaterThanOrEqual(endDiagnostics.floorHeight - 0.12);
      }

      if (scene.id === "stiffness") {
        assertStiffnessComparison(
          startDiagnostics,
          endDiagnostics,
          initialDiagonal,
        );
      }

      console.log(
        `${scene.id}: ${(visualDifference.changedPixelRatio * 100).toFixed(2)}% ` +
          `changed pixels, mean RGB delta ${visualDifference.meanRgbDelta.toFixed(2)}, ` +
          `downward landmark motion ${downwardLandmarkMotion.toFixed(5)}, ` +
          `bounds motion ${boundsMotion.toFixed(5)}`,
      );
      await recordScenePerformance(page, testInfo, scene.id);
    });
  }
});

test.describe("long-run settled body stability", () => {
  for (const scene of longRunScenes) {
    test(`${scene.id}: remains in its lane for 20 simulated seconds`, async ({
      page,
    }, testInfo) => {
      await page.goto(`/?scene=${scene.id}&test=1`, {
        waitUntil: "domcontentloaded",
      });
      await requireHardwareWebGPU(page, testInfo);
      await waitForTestHarness(page);

      const initial = await readDiagnostics(page);
      assertDiagnostics(initial, 0, scene.id);
      assertBodyCount(initial, scene.expectedBodyCount, scene.id);

      const impactFrame = frameAtSeconds(initial.timestep, 1, scene.id);
      await stepToFrame(page, 0, impactFrame);
      const impact = await readDiagnostics(page);
      assertDiagnostics(impact, impactFrame, scene.id);
      assertBodyCount(impact, scene.expectedBodyCount, scene.id);

      const settledFrame = frameAtSeconds(initial.timestep, 4, scene.id);
      await stepToFrame(page, impactFrame, settledFrame);
      const settled = await readDiagnostics(page);
      assertDiagnostics(settled, settledFrame, scene.id);
      assertBodyCount(settled, scene.expectedBodyCount, scene.id);

      const finalFrame = frameAtSeconds(initial.timestep, 20, scene.id);
      await stepToFrame(page, settledFrame, finalFrame);
      const final = await readDiagnostics(page);
      assertDiagnostics(final, finalFrame, scene.id);
      assertBodyCount(final, scene.expectedBodyCount, scene.id);
      await waitForCanvasPresentation(page);
      await captureCanvas(page, testInfo, "settled-20s.png");

      expect(final.floorHeight, `${scene.id} floor height must remain stable`).toBe(
        initial.floorHeight,
      );
      expect(final.timestep, `${scene.id} timestep must remain stable`).toBe(
        initial.timestep,
      );
      expect(
        final.minTetDeterminant / initial.minTetDeterminant,
        `${scene.id} tetrahedra must retain at least 1% of their initial minimum determinant over 20 seconds`,
      ).toBeGreaterThan(0.01);
      expect(
        final.bounds.min[1],
        `${scene.id} must not pass materially through the floor over 20 seconds`,
      ).toBeGreaterThanOrEqual(final.floorHeight - 0.12);

      const initialBodies = indexBodies(initial);
      const impactBodies = indexBodies(impact);
      const settledBodies = indexBodies(settled);
      const finalBodies = indexBodies(final);

      for (const [bodyId, initialBody] of initialBodies) {
        const impactBody = requireBody(impactBodies, bodyId, scene.id, "1 second");
        const settledBody = requireBody(
          settledBodies,
          bodyId,
          scene.id,
          "4 seconds",
        );
        const finalBody = requireBody(finalBodies, bodyId, scene.id, "20 seconds");

        expect(
          impactBody.minY,
          `${scene.id} body ${bodyId} must have reached the floor by 1 second`,
        ).toBeLessThanOrEqual(impact.floorHeight + 0.05);
        expect(
          settledBody.minY,
          `${scene.id} body ${bodyId} must remain in floor contact at 4 seconds`,
        ).toBeLessThanOrEqual(settled.floorHeight + 0.05);
        expect(
          finalBody.minY,
          `${scene.id} body ${bodyId} must not penetrate the floor at 20 seconds`,
        ).toBeGreaterThanOrEqual(final.floorHeight - 0.12);

        const initialToFinalDrift = horizontalDistance(
          initialBody.centerOfMass,
          finalBody.centerOfMass,
        );
        const settledToFinalDrift = horizontalDistance(
          settledBody.centerOfMass,
          finalBody.centerOfMass,
        );
        expect(
          initialToFinalDrift,
          `${scene.id} body ${bodyId} must remain within its initial camera lane`,
        ).toBeLessThanOrEqual(0.5);
        expect(
          settledToFinalDrift,
          `${scene.id} body ${bodyId} horizontal COM drift from 4 to 20 seconds`,
        ).toBeLessThanOrEqual(0.05);

        const impactSpeed = horizontalSpeed(impactBody.linearVelocity);
        const settledSpeed = horizontalSpeed(settledBody.linearVelocity);
        const finalSpeed = horizontalSpeed(finalBody.linearVelocity);
        expect(
          settledSpeed,
          `${scene.id} body ${bodyId} tangential speed must not accelerate after impact`,
        ).toBeLessThanOrEqual(impactSpeed + 0.01);
        expect(
          finalSpeed,
          `${scene.id} body ${bodyId} tangential speed must not accelerate after settling`,
        ).toBeLessThanOrEqual(settledSpeed + 0.005);
        expect(
          finalSpeed,
          `${scene.id} body ${bodyId} tangential speed must decay from its impact value`,
        ).toBeLessThanOrEqual(impactSpeed * 0.5 + 0.005);
        expect(
          finalSpeed,
          `${scene.id} body ${bodyId} final horizontal speed`,
        ).toBeLessThanOrEqual(0.01);
        expect(
          Math.abs(settledBody.linearVelocity[1]),
          `${scene.id} body ${bodyId} must be vertically settled by 4 seconds`,
        ).toBeLessThan(0.1);
        expect(
          Math.abs(finalBody.linearVelocity[1]),
          `${scene.id} body ${bodyId} must remain vertically settled at 20 seconds`,
        ).toBeLessThan(0.05);

        console.log(
          `${scene.id} body ${bodyId}: initial-to-final xz drift ` +
            `${initialToFinalDrift.toFixed(5)}, settled-to-final xz drift ` +
            `${settledToFinalDrift.toFixed(5)}, horizontal speed ` +
            `${impactSpeed.toFixed(5)} -> ${settledSpeed.toFixed(5)} -> ` +
            finalSpeed.toFixed(5),
        );
      }
      await recordScenePerformance(page, testInfo, scene.id);
    });
  }
});

async function requireHardwareWebGPU(
  page: Page,
  testInfo: TestInfo,
): Promise<readonly string[]> {
  const support = await page.evaluate(async () => {
    const gpu = navigator.gpu;

    if (!gpu) {
      return {
        available: false,
        secureContext: window.isSecureContext,
        adapter: null,
      };
    }

    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });

    if (!adapter) {
      return {
        available: true,
        secureContext: window.isSecureContext,
        adapter: null,
      };
    }

    return {
      available: true,
      secureContext: window.isSecureContext,
      adapter: {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
        device: adapter.info.device,
        description: adapter.info.description,
        isFallbackAdapter: adapter.info.isFallbackAdapter,
        features: [...adapter.features].sort(),
      },
    };
  });

  if (!support.available) {
    throw new Error(
      `WebGPU is unavailable in Google Chrome (secure context: ${support.secureContext}). ` +
        "The deterministic physics tests require a hardware WebGPU adapter.",
    );
  }

  if (!support.adapter) {
    throw new Error(
      "Google Chrome could not create a hardware WebGPU adapter. " +
        "Software fallback does not validate the real-time GPU implementation.",
    );
  }

  const adapterDescription = [
    support.adapter.description,
    support.adapter.vendor,
    support.adapter.architecture,
    support.adapter.device,
  ]
    .filter((detail, index, all) => detail && all.indexOf(detail) === index)
    .join(" / ");

  if (
    support.adapter.isFallbackAdapter ||
    /swiftshader/i.test(adapterDescription)
  ) {
    throw new Error(
      `Chrome selected a software WebGPU adapter (${adapterDescription || "unknown adapter"}). ` +
        "The JGS2 scene tests require hardware WebGPU and never accept SwiftShader.",
    );
  }

  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapterDescription || "hardware adapter (details unavailable)",
  });
  console.log(
    `Hardware WebGPU adapter: ${adapterDescription || "details unavailable"}`,
  );
  return support.adapter.features;
}

async function waitForTestHarness(page: Page): Promise<void> {
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.profileCpuFrames === "function" &&
      typeof window.__jgs2Test?.profileGpuFrames === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function",
  );
  await page.evaluate(async () => {
    await window.__jgs2Test!.ready;
  });
}

async function recordScenePerformance(
  page: Page,
  testInfo: TestInfo,
  workloadId: string,
): Promise<void> {
  const options = DEFAULT_JGS2_E2E_PERFORMANCE_OPTIONS;

  await page.reload({ waitUntil: "domcontentloaded" });
  await waitForTestHarness(page);
  const cpuProfile = await page.evaluate(
    async (profileOptions) =>
      window.__jgs2Test!.profileCpuFrames(profileOptions),
    options,
  );

  await page.reload({ waitUntil: "domcontentloaded" });
  await waitForTestHarness(page);
  const gpuProfile = await page.evaluate(
    async (profileOptions) =>
      window.__jgs2Test!.profileGpuFrames(profileOptions),
    options,
  );

  const report = buildJGS2PerformanceBenchmark(cpuProfile, gpuProfile);
  expect(report.workloadId).toBe(workloadId);
  expect(report.measuredFrameCount).toBe(options.measuredFrameCount);
  expect(report.gpuTimestamp.featureEnabled).toBe(
    report.gpuTimestamp.supported,
  );
  expect(report.gpuTimestamp.timestampMapCount).toBe(
    report.gpuTimestamp.supported ? 1 : 0,
  );
  expect(report.stateEquivalent, "timestamp profiling must not change state").toBe(
    true,
  );

  const computeBudget = assessJGS2ComputeBudget(report);
  const artifact = {
    ...report,
    budgetAssessment: {
      ...computeBudget,
      enforcement:
        "informational-scene-log; the isolated performance-baseline test owns the enforced hardware budget",
    },
  };
  const artifactPath = testInfo.outputPath("scene-performance.json");
  await writeFile(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  await testInfo.attach("scene-performance.json", {
    path: artifactPath,
    contentType: "application/json",
  });

  const gpuSummary = report.gpuFrame
    ? `GPU ${report.gpuFrame.meanMilliseconds.toFixed(3)} ms/frame, ` +
      `${report.gpuSimulationStep!.meanMilliseconds.toFixed(3)} ms/step, ` +
      `${report.gpuRender!.meanMilliseconds.toFixed(3)} ms/render`
    : `GPU timestamp-query unavailable (${report.gpuTimestamp.reason})`;
  console.log(
    `[scene-performance] ${workloadId}: ` +
      `serialized ${report.endToEndFrame.averageFramesPerSecond!.toFixed(1)} FPS average, ` +
      `${report.endToEndFrame.onePercentLowFramesPerSecond!.toFixed(1)} FPS 1% low; ` +
      `wall ${report.endToEndFrame.meanMilliseconds.toFixed(3)} ms/frame ` +
      `(p95 ${report.endToEndFrame.p95Milliseconds.toFixed(3)}); ` +
      `CPU submit ${report.cpuFrameSubmission.meanMilliseconds.toFixed(3)} ms/frame, ` +
      `${report.cpuSimulationSubmission.meanMilliseconds.toFixed(3)} ms/step; ` +
      `${gpuSummary}; serialized step-compute budget ` +
      `${computeBudget.serializedStepBudgetMilliseconds.toFixed(3)} ms ` +
      `(assessment ${computeBudget.passesNecessaryComputeBudget ? "PASS" : "FAIL"}; ` +
      `wall mean ${computeBudget.meetsSerializedWallMeanBudget ? "PASS" : "FAIL"}, ` +
      `GPU p95 ${computeBudget.meetsGpuFrameP95Budget === null ? "N/A" : computeBudget.meetsGpuFrameP95Budget ? "PASS" : "FAIL"})`,
  );
}

async function waitForCanvasPresentation(page: Page): Promise<void> {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }),
  );
}

async function readDiagnostics(page: Page): Promise<JGS2Diagnostics> {
  return page.evaluate(async () => window.__jgs2Test!.diagnostics());
}

function assertDiagnostics(
  diagnostics: JGS2Diagnostics,
  expectedFrame: number,
  sceneId: string,
): void {
  expect(diagnostics.frame, `${sceneId} frame counter`).toBe(expectedFrame);
  expect(Number.isInteger(diagnostics.frame), `${sceneId} frame must be integral`).toBe(
    true,
  );
  expect(diagnostics.finite, `${sceneId} GPU state must remain finite`).toBe(true);
  expect(
    Number.isFinite(diagnostics.pinnedMaxError),
    `${sceneId} pinned error must be finite`,
  ).toBe(true);
  expect(typeof diagnostics.pinnedMaxErrorValid).toBe("boolean");
  if (diagnostics.pinnedMaxErrorValid) {
    expect(
      diagnostics.pinnedMaxError,
      `${sceneId} pinned vertices must remain fixed`,
    ).toBeLessThanOrEqual(1e-4);
  } else {
    expect(
      diagnostics.pinnedMaxError,
      `${sceneId} no-pin sentinel must be zero`,
    ).toBe(0);
  }
  expect(
    Number.isFinite(diagnostics.minTetDeterminant),
    `${sceneId} minimum tetrahedron determinant must be finite`,
  ).toBe(true);
  expect(
    diagnostics.minTetDeterminantValid,
    `${sceneId} tetrahedron determinant metric must be valid`,
  ).toBe(true);
  expect(
    diagnostics.minTetDeterminant,
    `${sceneId} must not contain inverted tetrahedra`,
  ).toBeGreaterThan(0);
  expect(
    Number.isFinite(diagnostics.floorHeight),
    `${sceneId} floor height must be finite`,
  ).toBe(true);
  expect(
    Number.isFinite(diagnostics.timestep),
    `${sceneId} timestep must be finite`,
  ).toBe(true);
  expect(diagnostics.timestep, `${sceneId} timestep must be positive`).toBeGreaterThan(
    0,
  );

  assertVec3(diagnostics.bounds.min, `${sceneId} bounds.min`);
  assertVec3(diagnostics.bounds.max, `${sceneId} bounds.max`);
  assertVec3(diagnostics.landmark, `${sceneId} landmark`);
  if (diagnostics.comparisonLandmark) {
    assertVec3(
      diagnostics.comparisonLandmark,
      `${sceneId} comparisonLandmark`,
    );
  }
  assertBodyDiagnostics(diagnostics.bodies, sceneId);

  const diagonal = boundsDiagonal(diagnostics);
  expect(diagonal, `${sceneId} bounds must be nondegenerate`).toBeGreaterThan(
    1e-6,
  );

  const tolerance = diagonal * 1e-4 + 1e-6;
  for (let axis = 0; axis < 3; axis += 1) {
    expect(
      diagnostics.bounds.max[axis],
      `${sceneId} bounds must be ordered on axis ${axis}`,
    ).toBeGreaterThan(diagnostics.bounds.min[axis]);
    expect(
      diagnostics.landmark[axis],
      `${sceneId} landmark must lie inside the dynamic bounds on axis ${axis}`,
    ).toBeGreaterThanOrEqual(diagnostics.bounds.min[axis] - tolerance);
    expect(
      diagnostics.landmark[axis],
      `${sceneId} landmark must lie inside the dynamic bounds on axis ${axis}`,
    ).toBeLessThanOrEqual(diagnostics.bounds.max[axis] + tolerance);
    if (diagnostics.comparisonLandmark) {
      expect(
        diagnostics.comparisonLandmark[axis],
        `${sceneId} comparison landmark must lie inside the dynamic bounds on axis ${axis}`,
      ).toBeGreaterThanOrEqual(diagnostics.bounds.min[axis] - tolerance);
      expect(
        diagnostics.comparisonLandmark[axis],
        `${sceneId} comparison landmark must lie inside the dynamic bounds on axis ${axis}`,
      ).toBeLessThanOrEqual(diagnostics.bounds.max[axis] + tolerance);
    }
  }
}

function assertBodyDiagnostics(
  bodies: readonly JGS2BodyDiagnostics[],
  sceneId: string,
): void {
  expect(Array.isArray(bodies), `${sceneId} bodies must be an array`).toBe(true);
  expect(bodies.length, `${sceneId} must expose at least one body`).toBeGreaterThan(
    0,
  );

  const ids = bodies.map((body) => body.bodyId);
  expect(ids, `${sceneId} body IDs must be unique and sorted`).toEqual(
    [...new Set(ids)].sort((left, right) => left - right),
  );
  for (const body of bodies) {
    expect(
      Number.isSafeInteger(body.bodyId) && body.bodyId >= 0,
      `${sceneId} body ID must be a nonnegative integer`,
    ).toBe(true);
    assertVec3(body.centerOfMass, `${sceneId} body ${body.bodyId} centerOfMass`);
    assertVec3(body.linearVelocity, `${sceneId} body ${body.bodyId} linearVelocity`);
    expect(
      Number.isFinite(body.minY),
      `${sceneId} body ${body.bodyId} minY must be finite`,
    ).toBe(true);
  }
}

function assertBodyCount(
  diagnostics: JGS2Diagnostics,
  expectedBodyCount: number,
  sceneId: string,
): void {
  expect(diagnostics.bodies, `${sceneId} body count`).toHaveLength(
    expectedBodyCount,
  );
}

function frameAtSeconds(
  timestep: number,
  seconds: number,
  sceneId: string,
): number {
  const frame = Math.round(seconds / timestep);
  expect(
    Math.abs(frame * timestep - seconds),
    `${sceneId} must represent ${seconds} seconds with an integral frame count`,
  ).toBeLessThan(1e-8);
  return frame;
}

async function stepToFrame(
  page: Page,
  currentFrame: number,
  targetFrame: number,
): Promise<void> {
  expect(targetFrame, "target frame must not precede current frame").toBeGreaterThanOrEqual(
    currentFrame,
  );
  await page.evaluate(async (frameCount) => {
    await window.__jgs2Test!.stepFrames(frameCount);
  }, targetFrame - currentFrame);
}

function indexBodies(
  diagnostics: JGS2Diagnostics,
): ReadonlyMap<number, JGS2BodyDiagnostics> {
  return new Map(diagnostics.bodies.map((body) => [body.bodyId, body]));
}

function requireBody(
  bodies: ReadonlyMap<number, JGS2BodyDiagnostics>,
  bodyId: number,
  sceneId: string,
  checkpoint: string,
): JGS2BodyDiagnostics {
  const body = bodies.get(bodyId);
  expect(
    body,
    `${sceneId} body ${bodyId} must exist at ${checkpoint}`,
  ).toBeDefined();
  if (!body) {
    throw new Error(`${sceneId} body ${bodyId} is missing at ${checkpoint}`);
  }
  return body;
}

function horizontalDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(right[0] - left[0], right[2] - left[2]);
}

function horizontalSpeed(velocity: Vec3): number {
  return Math.hypot(velocity[0], velocity[2]);
}

function vectorNorm(value: Vec3): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function vectorDifferenceNorm(left: Vec3, right: Vec3): number {
  return Math.hypot(
    left[0] - right[0],
    left[1] - right[1],
    left[2] - right[2],
  );
}

function assertStiffnessComparison(
  start: JGS2Diagnostics,
  end: JGS2Diagnostics,
  initialDiagonal: number,
): void {
  const startComparison = start.comparisonLandmark;
  const endComparison = end.comparisonLandmark;

  expect(
    startComparison,
    "stiffness diagnostics must include the stiff-tip comparison landmark at frame 0",
  ).toBeDefined();
  expect(
    endComparison,
    "stiffness diagnostics must include the stiff-tip comparison landmark after stepping",
  ).toBeDefined();
  if (!startComparison || !endComparison) {
    throw new Error(
      "stiffness diagnostics did not include both comparison landmarks",
    );
  }

  const softDownwardDeflection = start.landmark[1] - end.landmark[1];
  const stiffDownwardDeflection = startComparison[1] - endComparison[1];

  expect(
    stiffDownwardDeflection,
    "the stiff cantilever tip must still respond to gravity",
  ).toBeGreaterThan(0);
  expect(
    softDownwardDeflection,
    "the soft tip must deflect at least 1.5x farther downward than the stiff tip",
  ).toBeGreaterThan(stiffDownwardDeflection * 1.5);
  expect(
    (softDownwardDeflection - stiffDownwardDeflection) / initialDiagonal,
    "soft and stiff tip deflections must differ by at least 3% of the scene diagonal",
  ).toBeGreaterThan(0.03);
}

function assertVec3(value: Vec3, label: string): void {
  expect(Array.isArray(value), `${label} must be a tuple`).toBe(true);
  expect(value, `${label} must contain three coordinates`).toHaveLength(3);

  for (const coordinate of value) {
    expect(Number.isFinite(coordinate), `${label} must contain finite values`).toBe(
      true,
    );
  }
}

function boundsDiagonal(diagnostics: JGS2Diagnostics): number {
  return distance(diagnostics.bounds.min, diagnostics.bounds.max);
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(
    right[0] - left[0],
    right[1] - left[1],
    right[2] - left[2],
  );
}

function maximumBoundsMotion(
  start: JGS2Diagnostics,
  end: JGS2Diagnostics,
): number {
  let maximum = 0;

  for (let axis = 0; axis < 3; axis += 1) {
    maximum = Math.max(
      maximum,
      Math.abs(end.bounds.min[axis] - start.bounds.min[axis]),
      Math.abs(end.bounds.max[axis] - start.bounds.max[axis]),
    );
  }

  return maximum;
}

async function captureCanvas(
  page: Page,
  testInfo: TestInfo,
  artifactName: "start.png" | "end.png" | "settled-20s.png",
): Promise<DecodedPng> {
  const canvas = page.getByTestId("gpu-canvas");
  await expect(canvas).toBeVisible();

  const screenshot = await canvas.screenshot({
    path: testInfo.outputPath(artifactName),
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });

  expect(screenshot.byteLength, `${artifactName} must not be empty`).toBeGreaterThan(
    256,
  );
  await testInfo.attach(artifactName, {
    body: screenshot,
    contentType: "image/png",
  });

  const decoded = decodePng(screenshot);
  expect(decoded.width, `${artifactName} width`).toBeGreaterThanOrEqual(128);
  expect(decoded.height, `${artifactName} height`).toBeGreaterThanOrEqual(128);
  return decoded;
}

function comparePngs(
  start: DecodedPng,
  end: DecodedPng,
): { readonly changedPixelRatio: number; readonly meanRgbDelta: number } {
  expect(end.width, "start/end screenshot widths must match").toBe(start.width);
  expect(end.height, "start/end screenshot heights must match").toBe(start.height);

  const pixelCount = start.width * start.height;
  let changedPixels = 0;
  let absoluteRgbDelta = 0;

  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const startOffset = pixel * start.channels;
    const endOffset = pixel * end.channels;
    let largestChannelDelta = 0;

    for (let channel = 0; channel < 3; channel += 1) {
      const delta = Math.abs(
        end.pixels[endOffset + channel] - start.pixels[startOffset + channel],
      );
      absoluteRgbDelta += delta;
      largestChannelDelta = Math.max(largestChannelDelta, delta);
    }

    if (largestChannelDelta > 6) {
      changedPixels += 1;
    }
  }

  return {
    changedPixelRatio: changedPixels / pixelCount,
    meanRgbDelta: absoluteRgbDelta / (pixelCount * 3),
  };
}

function decodePng(encoded: Buffer): DecodedPng {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (!encoded.subarray(0, signature.length).equals(signature)) {
    throw new Error("Playwright returned an artifact that is not a PNG image.");
  }

  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = -1;
  let compressionMethod = -1;
  let filterMethod = -1;
  let interlaceMethod = -1;
  const imageDataChunks: Buffer[] = [];

  let offset = signature.length;
  while (offset + 12 <= encoded.length) {
    const length = encoded.readUInt32BE(offset);
    const type = encoded.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;

    if (dataEnd + 4 > encoded.length) {
      throw new Error(`PNG chunk ${type} extends past the end of the image.`);
    }

    if (type === "IHDR") {
      if (length !== 13) {
        throw new Error("PNG IHDR chunk has an invalid length.");
      }
      width = encoded.readUInt32BE(dataStart);
      height = encoded.readUInt32BE(dataStart + 4);
      bitDepth = encoded[dataStart + 8];
      colorType = encoded[dataStart + 9];
      compressionMethod = encoded[dataStart + 10];
      filterMethod = encoded[dataStart + 11];
      interlaceMethod = encoded[dataStart + 12];
    } else if (type === "IDAT") {
      imageDataChunks.push(encoded.subarray(dataStart, dataEnd));
    } else if (type === "IEND") {
      break;
    }

    offset = dataEnd + 4;
  }

  if (width < 1 || height < 1 || imageDataChunks.length === 0) {
    throw new Error("PNG is missing dimensions or image data.");
  }
  if (
    bitDepth !== 8 ||
    (colorType !== 2 && colorType !== 6) ||
    compressionMethod !== 0 ||
    filterMethod !== 0 ||
    interlaceMethod !== 0
  ) {
    throw new Error(
      `Unsupported Playwright PNG format: depth=${bitDepth}, color=${colorType}, ` +
        `compression=${compressionMethod}, filter=${filterMethod}, interlace=${interlaceMethod}.`,
    );
  }

  const channels: 3 | 4 = colorType === 2 ? 3 : 4;
  const stride = width * channels;
  const filtered = inflateSync(Buffer.concat(imageDataChunks));
  const expectedLength = (stride + 1) * height;

  if (filtered.length < expectedLength) {
    throw new Error(
      `PNG pixel stream is truncated (${filtered.length} < ${expectedLength}).`,
    );
  }

  const pixels = new Uint8Array(stride * height);
  let sourceOffset = 0;

  for (let row = 0; row < height; row += 1) {
    const filter = filtered[sourceOffset];
    sourceOffset += 1;

    if (filter > 4) {
      throw new Error(`PNG row ${row} uses unknown filter ${filter}.`);
    }

    for (let columnByte = 0; columnByte < stride; columnByte += 1) {
      const destination = row * stride + columnByte;
      const left = columnByte >= channels ? pixels[destination - channels] : 0;
      const up = row > 0 ? pixels[destination - stride] : 0;
      const upperLeft =
        row > 0 && columnByte >= channels
          ? pixels[destination - stride - channels]
          : 0;
      const predictor = pngFilterPredictor(filter, left, up, upperLeft);
      pixels[destination] = (filtered[sourceOffset] + predictor) & 0xff;
      sourceOffset += 1;
    }
  }

  return { width, height, channels, pixels };
}

function pngFilterPredictor(
  filter: number,
  left: number,
  up: number,
  upperLeft: number,
): number {
  switch (filter) {
    case 0:
      return 0;
    case 1:
      return left;
    case 2:
      return up;
    case 3:
      return Math.floor((left + up) / 2);
    case 4:
      return paethPredictor(left, up, upperLeft);
    default:
      throw new Error(`Unsupported PNG filter ${filter}.`);
  }
}

function paethPredictor(left: number, up: number, upperLeft: number): number {
  const estimate = left + up - upperLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const upperLeftDistance = Math.abs(estimate - upperLeft);

  if (leftDistance <= upDistance && leftDistance <= upperLeftDistance) {
    return left;
  }
  if (upDistance <= upperLeftDistance) {
    return up;
  }
  return upperLeft;
}
