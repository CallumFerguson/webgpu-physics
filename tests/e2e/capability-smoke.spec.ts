import { expect, test, type Page, type TestInfo } from "@playwright/test";

type Vec3 = readonly [number, number, number];
type CapabilitySceneId =
  | "minimal"
  | "contact"
  | "trough"
  | "cloth"
  | "stress";

interface CapabilityDiagnostics {
  readonly frame: number;
  readonly finite: boolean;
  readonly landmark: Vec3;
}

interface CapabilityHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  positions(): Promise<Float32Array>;
  diagnostics(): Promise<CapabilityDiagnostics>;
}

interface CapabilitySnapshot {
  readonly diagnostics: CapabilityDiagnostics;
  readonly positions: readonly number[];
}

interface CapabilityCase {
  readonly scene: CapabilitySceneId;
  readonly frameCount: number;
  readonly expectedDirection: Vec3;
  readonly minimumProjectedMotion: number;
}

declare global {
  interface Window {
    __jgs2Test?: CapabilityHarness;
  }
}

const CAPABILITY_CASES: readonly CapabilityCase[] = [
  {
    scene: "minimal",
    frameCount: 60,
    expectedDirection: [0, -1, 0],
    minimumProjectedMotion: 0.005,
  },
  {
    scene: "contact",
    frameCount: 24,
    expectedDirection: [1, -1, 0],
    minimumProjectedMotion: 0.01,
  },
  {
    scene: "trough",
    frameCount: 24,
    expectedDirection: [0, -1, 0],
    minimumProjectedMotion: 0.01,
  },
  {
    scene: "cloth",
    frameCount: 48,
    expectedDirection: [0, -1, 0],
    minimumProjectedMotion: 0.001,
  },
  {
    scene: "stress",
    frameCount: 60,
    expectedDirection: [0, -1, 0],
    minimumProjectedMotion: 0.005,
  },
];

test.use({ trace: "off" });

test("roadmap item 5: integrated public capability smoke", async ({
  browser,
}, testInfo) => {
  test.setTimeout(300_000);
  const browserErrors: string[] = [];

  let adapterChecked = false;
  for (const capability of CAPABILITY_CASES) {
    const context = await browser.newContext({
      baseURL: "http://127.0.0.1:4173",
      viewport: { width: 800, height: 600 },
      deviceScaleFactor: 1,
      colorScheme: "dark",
      locale: "en-US",
      timezoneId: "UTC",
      reducedMotion: "reduce",
    });
    const page = await context.newPage();
    page.on("pageerror", (error) => browserErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text());
    });
    try {
      await openCapabilityScene(page, capability.scene);
      if (!adapterChecked) {
        await requireHardwareWebGPU(page, testInfo);
        adapterChecked = true;
      }

      const start = await readSnapshot(page);
      assertFiniteSnapshot(capability.scene, start, 0);
      await captureScene(page, testInfo, capability.scene, "start");

      await stepCapabilityFrames(page, capability.frameCount);
      const end = await readSnapshot(page);
      assertFiniteSnapshot(capability.scene, end, capability.frameCount);
      await captureScene(page, testInfo, capability.scene, "end");

      const landmarkMotion = projectedMotion(
        start.diagnostics.landmark,
        end.diagnostics.landmark,
        capability.expectedDirection,
      );
      expect(
        landmarkMotion,
        `${capability.scene} primary landmark should move in its advertised direction`,
      ).toBeGreaterThan(capability.minimumProjectedMotion);

      console.log(
        `Item 5 ${capability.scene}: ${capability.frameCount} frames, ` +
          `landmark motion=${landmarkMotion.toFixed(6)}.`,
      );
    } finally {
      await context.close();
    }
  }

  expect(browserErrors, "the integrated public smoke must not emit browser errors")
    .toEqual([]);
});

async function stepCapabilityFrames(
  page: Page,
  frameCount: number,
): Promise<void> {
  let remaining = frameCount;
  while (remaining > 0) {
    const batch = Math.min(12, remaining);
    await page.evaluate(
      async (batchFrameCount) =>
        window.__jgs2Test!.stepFrames(batchFrameCount),
      batch,
    );
    remaining -= batch;
  }
}

async function openCapabilityScene(
  page: Page,
  scene: CapabilitySceneId,
): Promise<void> {
  await page.goto(`/?scene=${scene}&test=1`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.positions === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);
}

async function readSnapshot(page: Page): Promise<CapabilitySnapshot> {
  return page.evaluate(async () => ({
    diagnostics: await window.__jgs2Test!.diagnostics(),
    positions: Array.from(await window.__jgs2Test!.positions()),
  }));
}

function assertFiniteSnapshot(
  scene: CapabilitySceneId,
  snapshot: CapabilitySnapshot,
  expectedFrame: number,
): void {
  expect(snapshot.diagnostics.frame, `${scene} diagnostic frame`).toBe(
    expectedFrame,
  );
  expect(snapshot.diagnostics.finite, `${scene} diagnostics`).toBe(true);
  expect(
    snapshot.diagnostics.landmark.every(Number.isFinite),
    `${scene} landmark`,
  ).toBe(true);
  expect(snapshot.positions.length, `${scene} packed positions`).toBeGreaterThan(
    0,
  );
  expect(snapshot.positions.length % 4, `${scene} vec4 position stride`).toBe(0);
  expect(
    snapshot.positions.every(Number.isFinite),
    `${scene} positions`,
  ).toBe(true);
}

function projectedMotion(start: Vec3, end: Vec3, direction: Vec3): number {
  const directionNorm = Math.hypot(...direction);
  if (!(directionNorm > 0)) {
    throw new Error("Expected landmark direction must be nonzero.");
  }
  return (
    ((end[0] - start[0]) * direction[0] +
      (end[1] - start[1]) * direction[1] +
      (end[2] - start[2]) * direction[2]) /
    directionNorm
  );
}

async function captureScene(
  page: Page,
  testInfo: TestInfo,
  scene: CapabilitySceneId,
  phase: "start" | "end",
): Promise<void> {
  const name = `${scene}-${phase}.png`;
  const outputPath = testInfo.outputPath(name);
  const screenshot = await page.locator(".canvas-shell").screenshot({
    path: outputPath,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });
  expect(screenshot.byteLength, `${scene} ${phase} screenshot`).toBeGreaterThan(
    256,
  );
  await testInfo.attach(name, { body: screenshot, contentType: "image/png" });
}

async function requireHardwareWebGPU(
  page: Page,
  testInfo: TestInfo,
): Promise<void> {
  const adapter = await page.evaluate(async () => {
    const selected = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });
    return selected
      ? {
          fallback: selected.info.isFallbackAdapter,
          description: [
            selected.info.description,
            selected.info.vendor,
            selected.info.architecture,
            selected.info.device,
          ]
            .filter(Boolean)
            .join(" / "),
        }
      : null;
  });
  expect(adapter, "Chrome must expose a hardware WebGPU adapter").not.toBeNull();
  expect(adapter!.fallback, "software WebGPU does not validate public scenes").toBe(
    false,
  );
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapter!.description || "hardware adapter",
  });
}
