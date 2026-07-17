import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { buildScene, toJGS2GpuInput } from "../../src/scenes";
import {
  evaluateIpcEdgeEdgeDistance,
  evaluateIpcVertexTriangleDistance,
  type StaticIpcContactCandidates,
} from "../../src/simulation/cpu";

type Vec3 = readonly [number, number, number];

interface ContactBodyDiagnostics {
  readonly bodyId: number;
  readonly centerOfMass: Vec3;
  readonly linearVelocity: Vec3;
  readonly minY: number;
}

interface ContactDiagnostics {
  readonly frame: number;
  readonly finite: boolean;
  readonly minTetDeterminant: number;
  readonly minTetDeterminantValid: boolean;
  readonly bodies: readonly ContactBodyDiagnostics[];
}

interface ContactConfiguration {
  readonly floorStiffness: number;
  readonly contactTangentialDamping: number;
  readonly horizontalBodyCorrection: boolean;
  readonly ipcActivationDistance: number;
  readonly ipcMinimumDistance: number;
  readonly ipcBarrierStiffness: number;
  readonly ipcFrictionCoefficient: number;
  readonly ipcFrictionVelocityEpsilon: number;
  readonly ipcStepSafety: number;
  readonly contactCandidateCount: number;
}

interface ContactDemoHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  positions(): Promise<Float32Array>;
  diagnostics(): Promise<ContactDiagnostics>;
  configuration(): ContactConfiguration;
}

interface ContactSnapshot {
  readonly diagnostics: ContactDiagnostics;
  readonly configuration: ContactConfiguration;
  readonly positions: readonly number[];
  readonly distances: ContactDistanceSummary;
}

interface ContactDistanceSummary {
  readonly minimum: number;
  readonly byBodyPair: ReadonlyMap<string, number>;
}

interface ContactRun {
  readonly start: ContactSnapshot;
  readonly samples: readonly ContactSnapshot[];
  readonly end: ContactSnapshot;
}

const CONTACT_FRAME_DELTAS = [2, 2, 4, 12, 12, 4] as const;
const DISTANCE_TOLERANCE = 5e-4;

declare global {
  interface Window {
    __jgs2Test?: ContactDemoHarness;
  }
}

const scene = buildScene("contact");
const gpuInput = toJGS2GpuInput(scene);
const candidates = gpuInput.contactCandidates;
if (!candidates) {
  throw new Error("The public contact scene must provide static IPC candidates.");
}
const bodyIds = Array.from(scene.mesh.bodyIds);
const expectedCandidateCount =
  candidates.vertexTriangleCount + candidates.edgeEdgeCount;

test("roadmap item 2: IPC contact remains feasible and friction opposes sliding", async ({
  page,
}, testInfo) => {
  test.setTimeout(120_000);
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });

  await openContactDemo(page, "/?scene=contact&test=1");
  await requireHardwareWebGPU(page, testInfo);
  const frictionRun = await runContactTrajectory(
    page,
    candidates,
    bodyIds,
    testInfo,
    true,
  );

  assertIpcConfiguration(frictionRun.start.configuration, true);
  assertFiniteFeasibleTrajectory(frictionRun);
  logContactRun("friction", frictionRun);
  assertFallCollisionAndSettlement(frictionRun);

  await openContactDemo(page, "/?scene=contact&test=1&friction=0");
  const frictionlessRun = await runContactTrajectory(
    page,
    candidates,
    bodyIds,
    testInfo,
    false,
  );

  assertIpcConfiguration(frictionlessRun.start.configuration, false);
  assertFiniteFeasibleTrajectory(frictionlessRun);
  logContactRun("frictionless", frictionlessRun);

  const frictionTravel = body(frictionRun.end.diagnostics, 2).centerOfMass[0] -
    body(frictionRun.start.diagnostics, 2).centerOfMass[0];
  const frictionlessTravel =
    body(frictionlessRun.end.diagnostics, 2).centerOfMass[0] -
    body(frictionlessRun.start.diagnostics, 2).centerOfMass[0];
  console.log(
    `Item 2 upper-body x travel: friction=${frictionTravel}, ` +
      `frictionless=${frictionlessTravel}`,
  );
  expect(
    frictionlessTravel,
    "without friction the upper block should retain its prescribed +x slide",
  ).toBeGreaterThan(0.05);
  expect(
    frictionlessTravel - frictionTravel,
    "IPC friction should measurably reduce the upper block's x travel",
  ).toBeGreaterThan(0.001);

  expect(browserErrors, "the contact demo must not emit browser errors").toEqual(
    [],
  );
});

async function openContactDemo(page: Page, url: string): Promise<void> {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.positions === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function" &&
      typeof window.__jgs2Test?.configuration === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);
}

async function runContactTrajectory(
  page: Page,
  staticCandidates: StaticIpcContactCandidates,
  vertexBodyIds: readonly number[],
  testInfo: TestInfo,
  captureScreenshots: boolean,
): Promise<ContactRun> {
  const start = await readSnapshot(page, staticCandidates, vertexBodyIds);
  expect(start.diagnostics.frame).toBe(0);
  if (captureScreenshots) {
    await captureContactDemo(page, testInfo, "start.png");
  }

  const samples: ContactSnapshot[] = [start];
  for (const frameDelta of CONTACT_FRAME_DELTAS) {
    await page.evaluate(
      async (frames) => window.__jgs2Test!.stepFrames(frames),
      frameDelta,
    );
    samples.push(await readSnapshot(page, staticCandidates, vertexBodyIds));
  }
  const end = samples.at(-1)!;
  expect(end.diagnostics.frame).toBe(
    CONTACT_FRAME_DELTAS.reduce((sum, value) => sum + value, 0),
  );
  if (captureScreenshots) {
    await captureContactDemo(page, testInfo, "end.png");
  }
  return { start, samples, end };
}

async function readSnapshot(
  page: Page,
  staticCandidates: StaticIpcContactCandidates,
  vertexBodyIds: readonly number[],
): Promise<ContactSnapshot> {
  const state = await page.evaluate(async () => ({
    diagnostics: await window.__jgs2Test!.diagnostics(),
    configuration: window.__jgs2Test!.configuration(),
    positions: Array.from(await window.__jgs2Test!.positions()),
  }));
  return {
    ...state,
    distances: contactDistances(
      state.positions,
      staticCandidates,
      vertexBodyIds,
    ),
  };
}

function assertIpcConfiguration(
  configuration: ContactConfiguration,
  frictionEnabled: boolean,
): void {
  expect(configuration.floorStiffness).toBe(0);
  expect(configuration.contactTangentialDamping).toBe(0);
  expect(configuration.horizontalBodyCorrection).toBe(false);
  expect(configuration.contactCandidateCount).toBe(expectedCandidateCount);
  expect(configuration.contactCandidateCount).toBeGreaterThan(0);
  expect(configuration.ipcActivationDistance).toBeGreaterThan(
    configuration.ipcMinimumDistance,
  );
  expect(configuration.ipcMinimumDistance).toBeGreaterThan(0);
  expect(configuration.ipcBarrierStiffness).toBeGreaterThan(0);
  expect(configuration.ipcFrictionVelocityEpsilon).toBeGreaterThan(0);
  expect(configuration.ipcStepSafety).toBeGreaterThan(0);
  expect(configuration.ipcStepSafety).toBeLessThan(1);
  if (frictionEnabled) {
    expect(configuration.ipcFrictionCoefficient).toBeGreaterThan(0);
  } else {
    expect(configuration.ipcFrictionCoefficient).toBe(0);
  }
}

function assertFiniteFeasibleTrajectory(run: ContactRun): void {
  const minimumDistance = run.start.configuration.ipcMinimumDistance;
  for (const sample of run.samples) {
    expect(sample.diagnostics.finite).toBe(true);
    expect(sample.diagnostics.minTetDeterminantValid).toBe(true);
    expect(sample.diagnostics.minTetDeterminant).toBeGreaterThan(0);
    expect(
      sample.positions.every(Number.isFinite),
      `frame ${sample.diagnostics.frame} positions must remain finite`,
    ).toBe(true);
    expect(
      sample.distances.minimum,
      `frame ${sample.diagnostics.frame} exact static-candidate distance ` +
        `must stay above the IPC minimum`,
    ).toBeGreaterThanOrEqual(minimumDistance - DISTANCE_TOLERANCE);
  }
}

function assertFallCollisionAndSettlement(run: ContactRun): void {
  const activationDistance = run.start.configuration.ipcActivationDistance;
  for (const bodyId of [1, 2]) {
    const startBody = body(run.start.diagnostics, bodyId);
    const lowestCenter = Math.min(
      ...run.samples.map(
        (sample) => body(sample.diagnostics, bodyId).centerOfMass[1],
      ),
    );
    const peakDownwardSpeed = Math.max(
      ...run.samples.map((sample) =>
        Math.max(0, -body(sample.diagnostics, bodyId).linearVelocity[1]),
      ),
    );
    const settledBody = body(run.end.diagnostics, bodyId);
    expect(
      startBody.centerOfMass[1] - lowestCenter,
      `dynamic body ${bodyId} should fall before settling`,
    ).toBeGreaterThan(0.005);
    expect(
      peakDownwardSpeed,
      `dynamic body ${bodyId} should acquire downward velocity`,
    ).toBeGreaterThan(0.02);
    expect(
      Math.abs(settledBody.linearVelocity[1]),
      `dynamic body ${bodyId} should settle vertically`,
    ).toBeLessThan(0.25);
  }

  const dynamicPairMinimum = Math.min(
    ...run.samples.map(
      (sample) => sample.distances.byBodyPair.get(bodyPairKey(1, 2)) ?? Infinity,
    ),
  );
  const slabLowerMinimum = Math.min(
    ...run.samples.map(
      (sample) => sample.distances.byBodyPair.get(bodyPairKey(0, 1)) ?? Infinity,
    ),
  );
  const selfContactMinimum = Math.min(
    ...run.samples.map(
      (sample) => sample.distances.byBodyPair.get(bodyPairKey(3, 3)) ?? Infinity,
    ),
  );
  expect(
    dynamicPairMinimum,
    "the two deformable bodies should enter the IPC activation distance",
  ).toBeLessThan(activationDistance);
  expect(
    slabLowerMinimum,
    "the lower deformable body should contact the pinned slab",
  ).toBeLessThan(activationDistance);
  expect(
    selfContactMinimum,
    "the same-body pair should enter the IPC activation distance without crossing",
  ).toBeLessThan(activationDistance);

  const penultimate = run.samples.at(-2)!;
  for (const bodyId of [1, 2]) {
    expect(
      Math.abs(
        body(run.end.diagnostics, bodyId).centerOfMass[1] -
          body(penultimate.diagnostics, bodyId).centerOfMass[1],
      ),
      `dynamic body ${bodyId} should have little late vertical drift`,
    ).toBeLessThan(0.05);
  }
  expect(body(run.end.diagnostics, 2).centerOfMass[1]).toBeGreaterThan(
    body(run.end.diagnostics, 1).centerOfMass[1],
  );
}

function logContactRun(label: string, run: ContactRun): void {
  const finalDynamicBodies = [1, 2].map((bodyId) => {
    const result = body(run.end.diagnostics, bodyId);
    return {
      id: bodyId,
      center: result.centerOfMass,
      velocity: result.linearVelocity,
    };
  });
  console.log(
    `Item 2 ${label}: checkpoint minimum distances ` +
      `${run.samples.map((sample) => sample.distances.minimum).join(", ")}; ` +
      `final dynamic bodies ${JSON.stringify(finalDynamicBodies)}`,
  );
}

function body(
  diagnostics: ContactDiagnostics,
  bodyId: number,
): ContactBodyDiagnostics {
  const result = diagnostics.bodies.find(
    (candidate) => candidate.bodyId === bodyId,
  );
  if (!result) {
    throw new Error(`Missing diagnostics for body ${bodyId}.`);
  }
  return result;
}

function contactDistances(
  positions: readonly number[],
  staticCandidates: StaticIpcContactCandidates,
  vertexBodyIds: readonly number[],
): ContactDistanceSummary {
  let minimum = Infinity;
  const byBodyPair = new Map<string, number>();
  const indices = staticCandidates.packedIndices;
  const candidateCount =
    staticCandidates.vertexTriangleCount + staticCandidates.edgeEdgeCount;
  for (let candidate = 0; candidate < candidateCount; candidate += 1) {
    const offset = candidate * 4;
    const i0 = indices[offset]!;
    const i1 = indices[offset + 1]!;
    const i2 = indices[offset + 2]!;
    const i3 = indices[offset + 3]!;
    const tuple = [i0, i1, i2, i3] as const;
    let distance: number;
    let firstBody: number;
    let secondBody: number;
    if (candidate < staticCandidates.vertexTriangleCount) {
      distance = evaluateIpcVertexTriangleDistance(positions, tuple);
      firstBody = vertexBodyIds[i0]!;
      secondBody = vertexBodyIds[i1]!;
    } else {
      distance = evaluateIpcEdgeEdgeDistance(positions, tuple);
      firstBody = vertexBodyIds[i0]!;
      secondBody = vertexBodyIds[i2]!;
    }
    minimum = Math.min(minimum, distance);
    const key = bodyPairKey(firstBody, secondBody);
    byBodyPair.set(key, Math.min(byBodyPair.get(key) ?? Infinity, distance));
  }
  return { minimum, byBodyPair };
}

function bodyPairKey(first: number, second: number): string {
  return first <= second ? `${first}:${second}` : `${second}:${first}`;
}

async function captureContactDemo(
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
  console.log(`Item 2 screenshot: ${outputPath}`);
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
  expect(adapter!.fallback, "software WebGPU does not validate IPC contact").toBe(
    false,
  );
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapter!.description || "hardware adapter",
  });
}
