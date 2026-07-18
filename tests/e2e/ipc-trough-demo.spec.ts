import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { buildScene, toJGS2GpuInput } from "../../src/scenes";
import { minimumStaticIpcContactDistance } from "../../src/simulation/cpu";

const BOX_COUNT = 10;
const DMIN = 0.003;
const scene = buildScene("trough", { troughBoxCount: BOX_COUNT });
const candidates = toJGS2GpuInput(scene).contactCandidates;
if (!candidates) {
  throw new Error("The IPC trough scene must provide static contact candidates.");
}

interface TroughDiagnostics {
  readonly frame: number;
  readonly finite: boolean;
  readonly pinnedMaxError: number;
  readonly minTetDeterminant: number;
  readonly bodies: readonly {
    readonly bodyId: number;
    readonly centerOfMass: readonly [number, number, number];
    readonly minY: number;
  }[];
}

interface TroughHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  positions(): Promise<Float32Array>;
  diagnostics(): Promise<TroughDiagnostics>;
  configuration(): {
    readonly floorStiffness: number;
    readonly ipcActivationDistance: number;
    readonly ipcMinimumDistance: number;
    readonly ipcBarrierStiffness: number;
    readonly ipcFrictionCoefficient: number;
    readonly contactCandidateCount: number;
  };
}

declare global {
  interface Window {
    __jgs2Test?: TroughHarness;
  }
}

test.use({ trace: "off" });

test("IPC trough box count configures bodies and remains shareable", async ({
  page,
}, testInfo) => {
  test.setTimeout(120_000);
  await page.goto("/?scene=trough&boxes=3&test=1", {
    waitUntil: "domcontentloaded",
  });
  await requireHardwareWebGPU(page, testInfo);
  await page.waitForFunction(
    () => typeof window.__jgs2Test?.diagnostics === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);

  await expect(page.getByTestId("trough-box-count")).toHaveValue("3");
  await expect(page.getByRole("link", { name: /IPC trough/ })).toHaveAttribute(
    "href",
    "?scene=trough&boxes=3",
  );
  const diagnostics = await page.evaluate(() =>
    window.__jgs2Test!.diagnostics(),
  );
  expect(diagnostics.bodies).toHaveLength(4);
  expect(diagnostics.finite).toBe(true);
});

test("IPC trough keeps a configurable box pile finite and feasible", async ({
  page,
}, testInfo) => {
  test.setTimeout(180_000);
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  await page.goto(`/?scene=trough&boxes=${BOX_COUNT}&test=1`, {
    waitUntil: "domcontentloaded",
  });
  await requireHardwareWebGPU(page, testInfo);
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);

  await expect(page.getByTestId("trough-box-count")).toHaveValue(
    String(BOX_COUNT),
  );
  const configuration = await page.evaluate(() =>
    window.__jgs2Test!.configuration(),
  );
  expect(configuration).toMatchObject({
    floorStiffness: 0,
    ipcActivationDistance: 0.08,
    ipcMinimumDistance: DMIN,
    ipcBarrierStiffness: 100_000,
    ipcFrictionCoefficient: 0.45,
  });
  expect(configuration.contactCandidateCount).toBe(
    candidates.vertexTriangleCount + candidates.edgeEdgeCount,
  );

  const initial = await readSnapshot(page);
  assertSnapshot(initial.diagnostics, 0);
  expect(initial.diagnostics.bodies).toHaveLength(BOX_COUNT + 1);
  const initialBoxCenterY = initial.diagnostics.bodies
    .slice(1)
    .map((body) => body.centerOfMass[1]);
  let minimumDistance = minimumStaticIpcContactDistance(
    initial.positions,
    candidates,
    4,
  );

  for (let frame = 12; frame <= 96; frame += 12) {
    await page.evaluate(async () => window.__jgs2Test!.stepFrames(12));
    const snapshot = await readSnapshot(page);
    assertSnapshot(snapshot.diagnostics, frame);
    const frameMinimum = minimumStaticIpcContactDistance(
      snapshot.positions,
      candidates,
      4,
    );
    expect(frameMinimum, `frame ${frame} IPC distance`).toBeGreaterThan(
      DMIN - 5e-4,
    );
    minimumDistance = Math.min(minimumDistance, frameMinimum);
  }

  const final = await readSnapshot(page);
  for (let vertex = 0; vertex < scene.mesh.bodyIds.length; vertex += 1) {
    if (scene.mesh.bodyIds[vertex] === 0) continue;
    const offset = vertex * 4;
    expect(
      Math.abs(final.positions[offset]!),
      `vertex ${vertex} remains inside the trough width`,
    ).toBeLessThan(2.2);
    expect(
      Math.abs(final.positions[offset + 2]!),
      `vertex ${vertex} remains inside the trough depth`,
    ).toBeLessThan(1.5);
  }
  for (let box = 0; box < BOX_COUNT; box += 1) {
    const body = final.diagnostics.bodies[box + 1]!;
    expect(body.bodyId).toBe(box + 1);
    expect(body.centerOfMass[1]).toBeLessThan(initialBoxCenterY[box]! - 0.1);
    expect(body.centerOfMass[1], `box ${box + 1} reaches the trough pile`)
      .toBeLessThan(2);
    expect(body.minY, `box ${box + 1} remains in the trough`).toBeGreaterThan(
      0.05,
    );
  }
  expect(minimumDistance, "at least one IPC pair enters the activation band")
    .toBeLessThan(configuration.ipcActivationDistance);
  expect(browserErrors, "the trough demo must not emit browser errors").toEqual(
    [],
  );
  const screenshotPath = testInfo.outputPath("trough-pile.png");
  const screenshot = await page.locator(".canvas-shell").screenshot({
    path: screenshotPath,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });
  expect(screenshot.byteLength).toBeGreaterThan(256);
  await testInfo.attach("trough-pile.png", {
    body: screenshot,
    contentType: "image/png",
  });
});

async function readSnapshot(page: Page): Promise<{
  readonly diagnostics: TroughDiagnostics;
  readonly positions: readonly number[];
}> {
  return page.evaluate(async () => ({
    diagnostics: await window.__jgs2Test!.diagnostics(),
    positions: Array.from(await window.__jgs2Test!.positions()),
  }));
}

function assertSnapshot(
  diagnostics: TroughDiagnostics,
  expectedFrame: number,
): void {
  expect(diagnostics.frame).toBe(expectedFrame);
  expect(diagnostics.finite).toBe(true);
  expect(diagnostics.pinnedMaxError).toBeLessThan(2e-5);
  expect(diagnostics.minTetDeterminant).toBeGreaterThan(0);
  expect(
    diagnostics.bodies.every((body) =>
      [
        body.centerOfMass[0],
        body.centerOfMass[1],
        body.centerOfMass[2],
        body.minY,
      ].every(Number.isFinite),
    ),
  ).toBe(true);
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
          ].filter(Boolean).join(" / "),
        }
      : null;
  });
  expect(adapter, "Chrome must expose a hardware WebGPU adapter").not.toBeNull();
  expect(adapter!.fallback, "software WebGPU does not validate IPC").toBe(false);
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapter!.description,
  });
}
