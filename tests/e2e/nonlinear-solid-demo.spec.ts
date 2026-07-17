import { expect, test, type Page, type TestInfo } from "@playwright/test";

type Vec3 = readonly [number, number, number];

interface DemoDiagnostics {
  readonly frame: number;
  readonly finite: boolean;
  readonly lastStepIterations: number;
  readonly minTetDeterminant: number;
  readonly minTetDeterminantValid: boolean;
  readonly relativeResidual: number;
  readonly relativeResidualValid: boolean;
  readonly maximumUpdate: number;
  readonly maximumUpdateValid: boolean;
  readonly landmark: Vec3;
  readonly bodies: readonly {
    readonly linearVelocity: Vec3;
  }[];
}

interface DemoTargetState {
  readonly vertex: number;
  readonly phase: "waiting" | "pulling" | "holding" | "released";
  readonly active: boolean;
  readonly position: Vec3;
  readonly stiffness: number;
  readonly revision: number;
  readonly solverTargetActive: boolean;
  readonly objectiveRevision: number;
}

interface NonlinearDemoHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  diagnostics(): Promise<DemoDiagnostics>;
  scriptedTarget(): DemoTargetState | null;
}

declare global {
  interface Window {
    __jgs2Test?: NonlinearDemoHarness;
  }
}

test("roadmap item 1: stable nonlinear cantilever pulls, releases, and reports finite convergence", async ({
  page,
}, testInfo) => {
  test.setTimeout(90_000);
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });

  await page.goto("/?scene=minimal&test=1", {
    waitUntil: "domcontentloaded",
  });
  await requireHardwareWebGPU(page, testInfo);
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function" &&
      typeof window.__jgs2Test?.scriptedTarget === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);

  const solverHud = page.getByTestId("solver-hud");
  await expect(solverHud).toBeVisible();
  await expect(solverHud).toHaveAttribute("data-material", "Stable Neo-Hookean");

  const start = await readState(page);
  expect(start.diagnostics.frame).toBe(0);
  expect(start.diagnostics.finite).toBe(true);
  expect(start.target).toMatchObject({
    phase: "waiting",
    active: false,
    solverTargetActive: false,
    objectiveRevision: 0,
  });
  await captureDemo(page, testInfo, "start.png");

  await page.evaluate(async () => window.__jgs2Test!.stepFrames(47));
  const gravityOnly = await readState(page);
  expect(gravityOnly.diagnostics.frame).toBe(47);
  expect(gravityOnly.target).toMatchObject({
    phase: "waiting",
    active: false,
    solverTargetActive: false,
  });
  expect(
    start.diagnostics.landmark[1] - gravityOnly.diagnostics.landmark[1],
    "the untargeted cantilever must first sag under gravity",
  ).toBeGreaterThan(0.01);
  assertFiniteNonlinearState(gravityOnly.diagnostics);

  await page.evaluate(async () => window.__jgs2Test!.stepFrames(60));
  const pulled = await readState(page);
  expect(pulled.diagnostics.frame).toBe(107);
  expect(pulled.target).toMatchObject({
    phase: "holding",
    active: true,
    solverTargetActive: true,
  });
  expect(distance(pulled.target!.position, start.target!.position)).toBeGreaterThan(
    0.5,
  );
  expect(
    distance(pulled.diagnostics.landmark, start.diagnostics.landmark),
    "the stable Neo-Hookean cantilever must visibly deform under gravity and the target",
  ).toBeGreaterThan(0.05);
  assertFiniteNonlinearState(pulled.diagnostics);
  expect(
    pulled.diagnostics.lastStepIterations,
    "the settled target hold should trip the nonlinear convergence gate",
  ).toBeLessThan(7);
  console.log(
    `Item 1 pull checkpoint: ${pulled.diagnostics.lastStepIterations} effective iterations, ` +
      `relative residual ${pulled.diagnostics.relativeResidual}, ` +
      `maximum update ${pulled.diagnostics.maximumUpdate}`,
  );

  const revisionBeforeRelease = pulled.target!.objectiveRevision;
  await page.evaluate(async () => window.__jgs2Test!.stepFrames(1));
  const released = await readState(page);
  expect(released.diagnostics.frame).toBe(108);
  expect(released.target).toMatchObject({
    phase: "released",
    active: false,
    solverTargetActive: false,
    stiffness: 0,
    objectiveRevision: revisionBeforeRelease + 1,
  });
  expect(
    distance(released.diagnostics.landmark, pulled.diagnostics.landmark),
    "releasing the target must not teleport the body",
  ).toBeLessThan(0.35);
  expect(
    Math.hypot(...released.diagnostics.bodies[0]!.linearVelocity),
    "releasing the target must retain a finite, nonzero velocity state",
  ).toBeGreaterThan(1e-5);

  await page.evaluate(async () => window.__jgs2Test!.stepFrames(18));
  const end = await readState(page);
  expect(end.diagnostics.frame).toBe(126);
  expect(end.target).toMatchObject({
    phase: "released",
    active: false,
    solverTargetActive: false,
  });
  assertFiniteNonlinearState(end.diagnostics);
  console.log(
    `Item 1 released endpoint: ${end.diagnostics.lastStepIterations} effective iterations, ` +
      `relative residual ${end.diagnostics.relativeResidual}, ` +
      `maximum update ${end.diagnostics.maximumUpdate}`,
  );

  await expect.poll(async () => solverHud.getAttribute("data-iterations")).toBe(
    String(end.diagnostics.lastStepIterations),
  );
  await expect(solverHud).toHaveAttribute("data-convergence-finite", "true");
  await expect(solverHud).toHaveAttribute("data-target-phase", "released");
  await expect(solverHud).toHaveAttribute("data-target-active", "false");
  for (const attribute of ["data-relative-residual", "data-normalized-update"]) {
    const value = Number(await solverHud.getAttribute(attribute));
    expect(Number.isFinite(value), `${attribute} must be finite`).toBe(true);
  }

  await captureDemo(page, testInfo, "end.png");
  expect(browserErrors, "the demo must not emit browser errors").toEqual([]);
});

async function readState(page: Page): Promise<{
  readonly diagnostics: DemoDiagnostics;
  readonly target: DemoTargetState | null;
}> {
  return page.evaluate(async () => ({
    diagnostics: await window.__jgs2Test!.diagnostics(),
    target: window.__jgs2Test!.scriptedTarget(),
  }));
}

function assertFiniteNonlinearState(diagnostics: DemoDiagnostics): void {
  expect(diagnostics.finite).toBe(true);
  expect(diagnostics.lastStepIterations).toBeGreaterThan(0);
  expect(diagnostics.lastStepIterations).toBeLessThanOrEqual(7);
  expect(diagnostics.minTetDeterminantValid).toBe(true);
  expect(diagnostics.minTetDeterminant).toBeGreaterThan(0);
  expect(diagnostics.relativeResidualValid).toBe(true);
  expect(Number.isFinite(diagnostics.relativeResidual)).toBe(true);
  expect(diagnostics.maximumUpdateValid).toBe(true);
  expect(Number.isFinite(diagnostics.maximumUpdate)).toBe(true);
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(
    right[0] - left[0],
    right[1] - left[1],
    right[2] - left[2],
  );
}

async function captureDemo(
  page: Page,
  testInfo: TestInfo,
  name: "start.png" | "end.png",
): Promise<void> {
  const outputPath = testInfo.outputPath(name);
  const screenshot = await page.locator(".canvas-shell").screenshot({
    path: outputPath,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });
  expect(screenshot.byteLength).toBeGreaterThan(256);
  await testInfo.attach(name, { body: screenshot, contentType: "image/png" });
  console.log(`Item 1 screenshot: ${outputPath}`);
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
  expect(adapter!.fallback, "software WebGPU does not validate the demo").toBe(
    false,
  );
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapter!.description || "hardware adapter",
  });
}
