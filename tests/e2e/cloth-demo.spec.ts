import { expect, test, type Page, type TestInfo } from "@playwright/test";

import { buildScene, toJGS2GpuInput } from "../../src/scenes";
import {
  evaluateIpcEdgeEdgeDistance,
  evaluateIpcVertexTriangleDistance,
  type StaticIpcContactCandidates,
} from "../../src/simulation/cpu";

type Vec3 = readonly [number, number, number];

interface ClothDiagnostics {
  readonly frame: number;
  readonly finite: boolean;
  readonly minTetDeterminant: number;
  readonly minTetDeterminantValid: boolean;
}

interface ClothConfiguration {
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

interface ClothDemoHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  positions(): Promise<Float32Array>;
  diagnostics(): Promise<ClothDiagnostics>;
  configuration(): ClothConfiguration;
}

interface TriangleQuality {
  readonly minimumAreaRatio: number;
  readonly minimumNormalDotRatio: number;
}

interface ContactDistances {
  readonly minimum: number;
  readonly colliderCloth: number;
  readonly clothSelf: number;
}

interface ClothSnapshot {
  readonly diagnostics: ClothDiagnostics;
  readonly configuration: ClothConfiguration;
  readonly positions: readonly number[];
  readonly triangleQuality: TriangleQuality;
  readonly contactDistances: ContactDistances;
}

declare global {
  interface Window {
    __jgs2Test?: ClothDemoHarness;
  }
}

const scene = buildScene("cloth");
const cloth = scene.cloth;
if (!cloth) {
  throw new Error("The public cloth scene must provide triangle-cloth data.");
}
const gpuInput = toJGS2GpuInput(scene);
const candidateInput = gpuInput.contactCandidates;
if (!candidateInput) {
  throw new Error("The public cloth scene must provide static IPC candidates.");
}
const candidates: StaticIpcContactCandidates = candidateInput;

const vertexCount = scene.mesh.positions.length / 3;
const bodyIds = Array.from(scene.mesh.bodyIds);
const clothTriangles = Array.from(cloth.triangles);
const clothVertices = [...new Set(clothTriangles)];
const pinnedColliderVertices = Array.from(
  { length: vertexCount },
  (_unused, vertex) => vertex,
).filter(
  (vertex) =>
    scene.mesh.bodyIds[vertex] === 0 && scene.mesh.fixed[vertex] !== 0,
);
const pinnedClothVertices = clothVertices.filter(
  (vertex) => scene.mesh.fixed[vertex] !== 0,
);
const freeClothVertices = clothVertices.filter(
  (vertex) => scene.mesh.fixed[vertex] === 0,
);
const centerVertex = scene.landmark.vertex;
const expectedCandidateCount =
  candidates.vertexTriangleCount + candidates.edgeEdgeCount;
const selfContactCandidateCount = countBodyPairCandidates(
  candidates,
  bodyIds,
  1,
  1,
);

const FRAME_DELTAS = [8, 8, 8, 8, 8, 8] as const;
const POSITION_TOLERANCE = 2e-5;
const DISTANCE_TOLERANCE = 5e-4;

test("roadmap item 3: pinned StVK cloth drapes without inversion or IPC crossing", async ({
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

  await openClothDemo(page);
  await requireHardwareWebGPU(page, testInfo);

  const start = await readSnapshot(page);
  expect(start.diagnostics.frame).toBe(0);
  assertClothConfiguration(start.configuration);
  expect(pinnedColliderVertices).toHaveLength(8);
  expect(pinnedClothVertices).toHaveLength(2);
  expect(freeClothVertices).toHaveLength(23);
  expect(
    selfContactCandidateCount,
    "the topology-filtered IPC buffer must include nonlocal cloth self-contact",
  ).toBeGreaterThan(0);
  await captureClothDemo(page, testInfo, "start.png");

  const samples: ClothSnapshot[] = [start];
  for (const frameDelta of FRAME_DELTAS) {
    await page.evaluate(
      async (frames) => window.__jgs2Test!.stepFrames(frames),
      frameDelta,
    );
    samples.push(await readSnapshot(page));
  }
  const end = samples.at(-1)!;
  expect(end.diagnostics.frame).toBe(
    FRAME_DELTAS.reduce((sum, frameCount) => sum + frameCount, 0),
  );

  for (const sample of samples) {
    assertFiniteFeasibleSnapshot(sample, start.positions);
  }

  expect(
    position(start.positions, centerVertex)[1] -
      position(end.positions, centerVertex)[1],
    "the free cloth center should fall while its two corners stay pinned",
  ).toBeGreaterThan(0.01);
  expect(
    maximumDisplacement(start.positions, end.positions, freeClothVertices),
    "the pinned sheet should undergo a visible non-rigid deformation",
  ).toBeGreaterThan(0.02);
  expect(
    end.contactDistances.colliderCloth,
    "the falling sheet should reach the collider's IPC activation band",
  ).toBeLessThanOrEqual(
    end.configuration.ipcActivationDistance + DISTANCE_TOLERANCE,
  );

  console.log(
    `Item 3 cloth: ${selfContactCandidateCount} topology-filtered self-contact ` +
      `candidates; minimum distances ` +
      `${samples.map((sample) => sample.contactDistances.minimum).join(", ")}; ` +
      `collider distances ` +
      `${samples.map((sample) => sample.contactDistances.colliderCloth).join(", ")}; ` +
      `minimum area ratios ` +
      `${samples.map((sample) => sample.triangleQuality.minimumAreaRatio).join(", ")}`,
  );
  await captureClothDemo(page, testInfo, "end.png");
  expect(browserErrors, "the cloth demo must not emit browser errors").toEqual(
    [],
  );
});

async function openClothDemo(page: Page): Promise<void> {
  await page.goto("/?scene=cloth&test=1", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () =>
      typeof window.__jgs2Test?.stepFrames === "function" &&
      typeof window.__jgs2Test?.positions === "function" &&
      typeof window.__jgs2Test?.diagnostics === "function" &&
      typeof window.__jgs2Test?.configuration === "function",
  );
  await page.evaluate(async () => window.__jgs2Test!.ready);
}

async function readSnapshot(page: Page): Promise<ClothSnapshot> {
  const state = await page.evaluate(async () => ({
    diagnostics: await window.__jgs2Test!.diagnostics(),
    configuration: window.__jgs2Test!.configuration(),
    positions: Array.from(await window.__jgs2Test!.positions()),
  }));
  return {
    ...state,
    triangleQuality: evaluateTriangleQuality(state.positions),
    contactDistances: evaluateContactDistances(state.positions),
  };
}

function assertClothConfiguration(configuration: ClothConfiguration): void {
  expect(configuration.floorStiffness).toBe(0);
  expect(configuration.contactTangentialDamping).toBe(0);
  expect(configuration.horizontalBodyCorrection).toBe(false);
  expect(configuration.contactCandidateCount).toBe(expectedCandidateCount);
  expect(configuration.ipcActivationDistance).toBeGreaterThan(
    configuration.ipcMinimumDistance,
  );
  expect(configuration.ipcMinimumDistance).toBeGreaterThan(0);
  expect(configuration.ipcBarrierStiffness).toBeGreaterThan(0);
  expect(configuration.ipcFrictionCoefficient).toBeGreaterThan(0);
  expect(configuration.ipcFrictionVelocityEpsilon).toBeGreaterThan(0);
  expect(configuration.ipcStepSafety).toBeGreaterThan(0);
  expect(configuration.ipcStepSafety).toBeLessThan(1);
}

function assertFiniteFeasibleSnapshot(
  sample: ClothSnapshot,
  startPositions: readonly number[],
): void {
  const frame = sample.diagnostics.frame;
  expect(sample.diagnostics.finite).toBe(true);
  expect(sample.diagnostics.minTetDeterminantValid).toBe(true);
  expect(sample.diagnostics.minTetDeterminant).toBeGreaterThan(0);
  expect(sample.positions).toHaveLength(vertexCount * 4);
  expect(
    sample.positions.every(Number.isFinite),
    `frame ${frame} positions must remain finite`,
  ).toBe(true);
  expect(
    maximumDisplacement(
      startPositions,
      sample.positions,
      pinnedColliderVertices,
    ),
    `frame ${frame} pinned collider vertices must stay fixed`,
  ).toBeLessThanOrEqual(POSITION_TOLERANCE);
  expect(
    maximumDisplacement(startPositions, sample.positions, pinnedClothVertices),
    `frame ${frame} pinned cloth corners must stay fixed`,
  ).toBeLessThanOrEqual(POSITION_TOLERANCE);
  expect(
    sample.triangleQuality.minimumAreaRatio,
    `frame ${frame} cloth triangles must remain nondegenerate`,
  ).toBeGreaterThan(1e-4);
  expect(
    sample.triangleQuality.minimumNormalDotRatio,
    `frame ${frame} cloth normals must retain their start orientation`,
  ).toBeGreaterThan(0);
  expect(
    sample.contactDistances.minimum,
    `frame ${frame} topology-filtered static IPC distance must remain above dmin`,
  ).toBeGreaterThan(
    sample.configuration.ipcMinimumDistance - DISTANCE_TOLERANCE,
  );
  expect(
    sample.contactDistances.clothSelf,
    `frame ${frame} cloth self-contact distance must remain above dmin`,
  ).toBeGreaterThan(
    sample.configuration.ipcMinimumDistance - DISTANCE_TOLERANCE,
  );
  expect(
    sample.contactDistances.colliderCloth,
    `frame ${frame} cloth/collider distance must remain above dmin`,
  ).toBeGreaterThan(
    sample.configuration.ipcMinimumDistance - DISTANCE_TOLERANCE,
  );
}

function evaluateTriangleQuality(positions: readonly number[]): TriangleQuality {
  let minimumAreaRatio = Infinity;
  let minimumNormalDotRatio = Infinity;
  for (let triangle = 0; triangle < clothTriangles.length; triangle += 3) {
    const i0 = clothTriangles[triangle]!;
    const i1 = clothTriangles[triangle + 1]!;
    const i2 = clothTriangles[triangle + 2]!;
    const startNormal = triangleNormal(scene.mesh.positions, 3, i0, i1, i2);
    const currentNormal = triangleNormal(positions, 4, i0, i1, i2);
    const startSquared = dot(startNormal, startNormal);
    const currentSquared = dot(currentNormal, currentNormal);
    minimumAreaRatio = Math.min(
      minimumAreaRatio,
      Math.sqrt(currentSquared / startSquared),
    );
    minimumNormalDotRatio = Math.min(
      minimumNormalDotRatio,
      dot(startNormal, currentNormal) / startSquared,
    );
  }
  return { minimumAreaRatio, minimumNormalDotRatio };
}

function evaluateContactDistances(
  positions: readonly number[],
): ContactDistances {
  let minimum = Infinity;
  let colliderCloth = Infinity;
  let clothSelf = Infinity;
  for (const candidate of candidates.vertexTriangleCandidates) {
    const distance = evaluateIpcVertexTriangleDistance(positions, candidate, 4);
    minimum = Math.min(minimum, distance);
    const firstBody = bodyIds[candidate[0]]!;
    const secondBody = bodyIds[candidate[1]]!;
    if (isBodyPair(firstBody, secondBody, 0, 1)) {
      colliderCloth = Math.min(colliderCloth, distance);
    } else if (firstBody === 1 && secondBody === 1) {
      clothSelf = Math.min(clothSelf, distance);
    }
  }
  for (const candidate of candidates.edgeEdgeCandidates) {
    const distance = evaluateIpcEdgeEdgeDistance(positions, candidate, 4);
    minimum = Math.min(minimum, distance);
    const firstBody = bodyIds[candidate[0]]!;
    const secondBody = bodyIds[candidate[2]]!;
    if (isBodyPair(firstBody, secondBody, 0, 1)) {
      colliderCloth = Math.min(colliderCloth, distance);
    } else if (firstBody === 1 && secondBody === 1) {
      clothSelf = Math.min(clothSelf, distance);
    }
  }
  return { minimum, colliderCloth, clothSelf };
}

function countBodyPairCandidates(
  staticCandidates: StaticIpcContactCandidates,
  vertexBodyIds: readonly number[],
  firstBody: number,
  secondBody: number,
): number {
  let count = 0;
  for (const candidate of staticCandidates.vertexTriangleCandidates) {
    if (
      isBodyPair(
        vertexBodyIds[candidate[0]]!,
        vertexBodyIds[candidate[1]]!,
        firstBody,
        secondBody,
      )
    ) {
      count += 1;
    }
  }
  for (const candidate of staticCandidates.edgeEdgeCandidates) {
    if (
      isBodyPair(
        vertexBodyIds[candidate[0]]!,
        vertexBodyIds[candidate[2]]!,
        firstBody,
        secondBody,
      )
    ) {
      count += 1;
    }
  }
  return count;
}

function isBodyPair(
  first: number,
  second: number,
  expectedFirst: number,
  expectedSecond: number,
): boolean {
  return (
    (first === expectedFirst && second === expectedSecond) ||
    (first === expectedSecond && second === expectedFirst)
  );
}

function triangleNormal(
  positions: ArrayLike<number>,
  stride: 3 | 4,
  i0: number,
  i1: number,
  i2: number,
): Vec3 {
  const a = position(positions, i0, stride);
  const b = position(positions, i1, stride);
  const c = position(positions, i2, stride);
  const ab: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  return [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
}

function dot(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function position(
  positions: ArrayLike<number>,
  vertex: number,
  stride: 3 | 4 = 4,
): Vec3 {
  const offset = vertex * stride;
  return [positions[offset]!, positions[offset + 1]!, positions[offset + 2]!];
}

function maximumDisplacement(
  start: readonly number[],
  end: readonly number[],
  vertices: readonly number[],
): number {
  let maximum = 0;
  for (const vertex of vertices) {
    const initial = position(start, vertex);
    const current = position(end, vertex);
    maximum = Math.max(
      maximum,
      Math.hypot(
        current[0] - initial[0],
        current[1] - initial[1],
        current[2] - initial[2],
      ),
    );
  }
  return maximum;
}

async function captureClothDemo(
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
  console.log(`Item 3 screenshot: ${outputPath}`);
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
  expect(adapter!.fallback, "software WebGPU does not validate the cloth demo").toBe(
    false,
  );
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: adapter!.description || "hardware adapter",
  });
}
