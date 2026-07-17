import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { buildScene } from "../../src/scenes";
import { evaluateStableNeoHookeanMesh } from "../../src/simulation/cpu";

type Schedule = "jacobi" | "graph-colored-gauss-seidel";
type Vec3 = readonly [number, number, number];

interface ScheduleDiagnostics {
  readonly frame: number;
  readonly lastStepIterations: number;
  readonly finite: boolean;
  readonly pinnedMaxError: number;
  readonly pinnedMaxErrorValid: boolean;
  readonly minTetDeterminant: number;
  readonly minTetDeterminantValid: boolean;
  readonly landmark: Vec3;
  readonly bounds: {
    readonly min: Vec3;
    readonly max: Vec3;
  };
}

interface ScheduleConfiguration {
  readonly schedule: Schedule;
  readonly scheduleColorCount: number;
}

interface ScheduleHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  positions(): Promise<Float32Array>;
  diagnostics(): Promise<ScheduleDiagnostics>;
  oracleEnergy(): Promise<number>;
  configuration(): ScheduleConfiguration;
}

interface ScheduleSnapshot {
  readonly diagnostics: ScheduleDiagnostics;
  readonly positions: readonly number[];
  readonly materialEnergy: number;
  readonly implicitEnergy: number;
}

interface ScheduleRun {
  readonly start: ScheduleSnapshot;
  readonly end: ScheduleSnapshot;
}

declare global {
  interface Window {
    __jgs2Test?: ScheduleHarness;
  }
}

const FRAME_COUNT = 12;
const CONFIGURED_ITERATIONS = 7;
const comparisonScene = buildScene("minimal");

test("roadmap item 4: graph-colored Gauss-Seidel remains comparable to Jacobi", async ({
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

  await openSchedule(page, "jacobi");
  await requireHardwareWebGPU(page, testInfo);
  const jacobi = await runLoadedSchedule(page, "jacobi", "Jacobi");

  await openSchedule(page, "graph-colored-gauss-seidel");
  const colored = await runLoadedSchedule(
    page,
    "graph-colored-gauss-seidel",
    "Colored GS",
  );

  expectMaximumDifference(
    jacobi.start.positions,
    colored.start.positions,
    1e-7,
    "fresh loads must start from the same configuration",
  );

  const sceneScale = Math.max(boundsDiagonal(jacobi.end.diagnostics), 1e-6);
  const rmsDifference = rootMeanSquarePositionDifference(
    jacobi.end.positions,
    colored.end.positions,
  );
  const maximumDifference = maximumPositionDifference(
    jacobi.end.positions,
    colored.end.positions,
  );
  const landmarkDifference = distance(
    jacobi.end.diagnostics.landmark,
    colored.end.diagnostics.landmark,
  );
  const relativeMaterialEnergyDifference =
    Math.abs(jacobi.end.materialEnergy - colored.end.materialEnergy) /
    Math.max(
      1,
      Math.abs(jacobi.end.materialEnergy),
      Math.abs(colored.end.materialEnergy),
    );
  const relativeImplicitEnergyDifference =
    Math.abs(jacobi.end.implicitEnergy - colored.end.implicitEnergy) /
    Math.max(
      1,
      Math.abs(jacobi.end.implicitEnergy),
      Math.abs(colored.end.implicitEnergy),
    );

  expect(
    rmsDifference / sceneScale,
    "the two schedules should settle to comparable full-mesh configurations",
  ).toBeLessThan(5e-4);
  expect(
    maximumDifference / sceneScale,
    "no vertex should diverge materially between the two schedules",
  ).toBeLessThan(1e-3);
  expect(
    landmarkDifference / sceneScale,
    "the public free-tip landmark should agree across schedules",
  ).toBeLessThan(1e-3);
  expect(
    relativeMaterialEnergyDifference,
    "stable Neo-Hookean material energy should be comparable across schedules",
  ).toBeLessThan(2e-3);
  expect(
    relativeImplicitEnergyDifference,
    "exact all-element implicit energy should be comparable across schedules",
  ).toBeLessThan(2e-3);

  console.log(
    `Item 4 schedule comparison after ${FRAME_COUNT} frames: ` +
      `RMS/scale=${(rmsDifference / sceneScale).toExponential(4)}, ` +
      `max/scale=${(maximumDifference / sceneScale).toExponential(4)}, ` +
      `landmark/scale=${(landmarkDifference / sceneScale).toExponential(4)}, ` +
      `relative material energy=${relativeMaterialEnergyDifference.toExponential(4)}, ` +
      `relative implicit energy=${relativeImplicitEnergyDifference.toExponential(4)}.`,
  );
  expect(browserErrors, "the schedule comparison must not emit browser errors")
    .toEqual([]);
});

async function openSchedule(page: Page, schedule: Schedule): Promise<void> {
  const suffix =
    schedule === "jacobi" ? "" : `&schedule=${schedule}`;
  await page.goto(`/?scene=minimal&test=1${suffix}`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.positions === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function" &&
      typeof window.__jgs2Test?.oracleEnergy === "function" &&
      typeof window.__jgs2Test?.configuration === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);
}

async function runLoadedSchedule(
  page: Page,
  schedule: Schedule,
  displayLabel: string,
): Promise<ScheduleRun> {
  const configuration = await page.evaluate(() =>
    window.__jgs2Test!.configuration(),
  );
  expect(configuration.schedule).toBe(schedule);
  expect(configuration.scheduleColorCount).toBeGreaterThan(1);
  await expect(page.getByTestId("solver-hud")).toHaveAttribute(
    "data-schedule",
    schedule,
  );
  await expect(page.getByTestId("active-schedule")).toHaveText(displayLabel);
  await expect(page.getByTestId("schedule-badge")).toHaveAttribute(
    "data-schedule",
    schedule,
  );
  await expect(page.getByTestId("actual-iterations")).toContainText(
    `${CONFIGURED_ITERATIONS} actual`,
  );

  const start = await readSnapshot(page);
  assertFiniteSnapshot(start, 0);
  await page.evaluate(
    async (frameCount) => window.__jgs2Test!.stepFrames(frameCount),
    FRAME_COUNT,
  );
  const end = await readSnapshot(page);
  assertFiniteSnapshot(end, FRAME_COUNT);
  expect(end.diagnostics.lastStepIterations).toBeGreaterThan(0);
  expect(end.diagnostics.lastStepIterations).toBeLessThanOrEqual(
    CONFIGURED_ITERATIONS,
  );
  expect(
    maximumPositionDifference(start.positions, end.positions),
    `${schedule} should advance the physical state`,
  ).toBeGreaterThan(1e-4);
  return { start, end };
}

async function readSnapshot(page: Page): Promise<ScheduleSnapshot> {
  const snapshot = await page.evaluate(async () => ({
    diagnostics: await window.__jgs2Test!.diagnostics(),
    positions: Array.from(await window.__jgs2Test!.positions()),
    implicitEnergy: await window.__jgs2Test!.oracleEnergy(),
  }));
  return {
    ...snapshot,
    materialEnergy: stableMaterialEnergy(snapshot.positions),
  };
}

function stableMaterialEnergy(positions: readonly number[]): number {
  const xyz = new Float64Array((positions.length / 4) * 3);
  for (let vertex = 0; vertex < positions.length / 4; vertex += 1) {
    xyz.set(positions.slice(vertex * 4, vertex * 4 + 3), vertex * 3);
  }
  return evaluateStableNeoHookeanMesh(
    comparisonScene.mesh,
    comparisonScene.restTetraData,
    comparisonScene.materials,
    xyz,
  ).energy;
}

function assertFiniteSnapshot(
  snapshot: ScheduleSnapshot,
  expectedFrame: number,
): void {
  expect(snapshot.diagnostics.frame).toBe(expectedFrame);
  expect(snapshot.diagnostics.finite).toBe(true);
  expect(snapshot.diagnostics.pinnedMaxErrorValid).toBe(true);
  expect(snapshot.diagnostics.pinnedMaxError).toBeLessThanOrEqual(2e-5);
  expect(snapshot.diagnostics.minTetDeterminantValid).toBe(true);
  expect(snapshot.diagnostics.minTetDeterminant).toBeGreaterThan(0);
  expect(snapshot.positions.length).toBeGreaterThan(0);
  expect(snapshot.positions.length % 4).toBe(0);
  expect(snapshot.positions.every(Number.isFinite)).toBe(true);
  expect(Number.isFinite(snapshot.materialEnergy)).toBe(true);
  expect(snapshot.materialEnergy).toBeGreaterThanOrEqual(0);
  expect(Number.isFinite(snapshot.implicitEnergy)).toBe(true);
}

function boundsDiagonal(diagnostics: ScheduleDiagnostics): number {
  return distance(diagnostics.bounds.min, diagnostics.bounds.max);
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(
    left[0] - right[0],
    left[1] - right[1],
    left[2] - right[2],
  );
}

function rootMeanSquarePositionDifference(
  left: readonly number[],
  right: readonly number[],
): number {
  expect(right).toHaveLength(left.length);
  let sum = 0;
  const vertexCount = left.length / 4;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const offset = vertex * 4;
    const difference = Math.hypot(
      left[offset]! - right[offset]!,
      left[offset + 1]! - right[offset + 1]!,
      left[offset + 2]! - right[offset + 2]!,
    );
    sum += difference * difference;
  }
  return Math.sqrt(sum / vertexCount);
}

function maximumPositionDifference(
  left: readonly number[],
  right: readonly number[],
): number {
  expect(right).toHaveLength(left.length);
  let maximum = 0;
  for (let offset = 0; offset < left.length; offset += 4) {
    maximum = Math.max(
      maximum,
      Math.hypot(
        left[offset]! - right[offset]!,
        left[offset + 1]! - right[offset + 1]!,
        left[offset + 2]! - right[offset + 2]!,
      ),
    );
  }
  return maximum;
}

function expectMaximumDifference(
  left: readonly number[],
  right: readonly number[],
  tolerance: number,
  message: string,
): void {
  expect(maximumPositionDifference(left, right), message).toBeLessThanOrEqual(
    tolerance,
  );
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
  expect(adapter!.fallback, "software WebGPU does not validate scheduling").toBe(
    false,
  );
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapter!.description || "hardware adapter",
  });
}
