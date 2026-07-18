import { expect, test } from "@playwright/test";

const PROFILE_ENABLED = process.env.JGS2_IPC_PROFILE === "1";
const PROFILE_OPTIONS = {
  warmupFrameCount: 10,
  measuredFrameCount: 30,
} as const;

const PRODUCTION_CASES = [
  { id: "contact", scene: "contact", timestep: 1 / 60, minimumSamples: 120 },
  { id: "trough", scene: "trough", timestep: 1 / 60, minimumSamples: 120 },
  { id: "cloth", scene: "cloth", timestep: 1 / 120, minimumSamples: 240 },
  {
    id: "cloth-60hz-batched",
    scene: "cloth",
    timestep: 1 / 120,
    minimumSamples: 120,
    emulatedFrameIntervalMilliseconds: 1000 / 60,
  },
] as const;

const PROFILE_CASES = [
  { id: "minimal", query: "scene=minimal" },
  { id: "stress", query: "scene=stress" },
  {
    id: "contact-friction",
    query: "scene=contact",
    expectedConfiguration: {
      ipcActivationDistance: 0.08,
      maxStep: 0.075,
      lastStepIterations: 7,
    },
  },
  { id: "contact-frictionless", query: "scene=contact&friction=0" },
  { id: "contact-barrier-zero", query: "scene=contact&barrier=0" },
  {
    id: "contact-narrow-band",
    query: "scene=contact&activation=0.004&friction=0",
  },
  {
    id: "trough-friction",
    query: "scene=trough",
    preprofileFrames: 96,
    expectedConfiguration: {
      ipcActivationDistance: 0.08,
      maxStep: 0.075,
      lastStepIterations: 3,
    },
  },
  {
    id: "cloth-friction",
    query: "scene=cloth",
    expectedConfiguration: {
      ipcActivationDistance: 0.01,
      maxStep: 0.01,
      lastStepIterations: 5,
    },
  },
  { id: "cloth-settled", query: "scene=cloth", preprofileFrames: 240 },
  {
    id: "cloth-settled-frictionless",
    query: "scene=cloth&friction=0",
    preprofileFrames: 240,
  },
  { id: "cloth-frictionless", query: "scene=cloth&friction=0" },
  { id: "cloth-barrier-zero", query: "scene=cloth&barrier=0" },
  {
    id: "cloth-narrow-band",
    query: "scene=cloth&activation=0.004&friction=0",
  },
] as const;

interface ProfileHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  profileCombinedFrames(options: typeof PROFILE_OPTIONS): Promise<{
    readonly cpuProfile: {
      readonly samples: {
        readonly endToEndFrameMilliseconds: readonly number[];
        readonly cpuSimulationSubmissionMilliseconds: readonly number[];
        readonly cpuRenderSubmissionMilliseconds: readonly number[];
      };
    };
    readonly gpuProfile: {
      readonly timestamp: {
        readonly supported: boolean;
        readonly reason: string | null;
        readonly timestampMapCount: 0 | 1;
      };
      readonly samples: {
        readonly gpuFrameMilliseconds: readonly number[];
        readonly gpuSimulationStepMilliseconds: readonly number[];
        readonly gpuRenderMilliseconds: readonly number[];
      };
    };
  }>;
  configuration(): {
    readonly contactCandidateCount: number;
    readonly ipcFrictionCoefficient: number;
    readonly ipcActivationDistance: number;
    readonly schedule: string;
    readonly scheduleColorCount: number;
    readonly maxStep: number;
  };
  diagnostics(): Promise<{
    readonly finite: boolean;
    readonly lastStepIterations: number;
    readonly activeContactCount: number;
    readonly activeContactCountValid: boolean;
    readonly minimumContactDistance: number;
    readonly minimumContactDistanceValid: boolean;
  }>;
  submissionPolicy(): {
    readonly solverSubmissions: number;
    readonly renderSubmissions: number;
    readonly readbackSubmissions: number;
    readonly maximumOutstanding: number;
  };
}

declare global {
  interface Window {
    __jgs2Test?: ProfileHarness;
  }
}

// Browser tracing is useful for correctness failures but perturbs wall-time
// samples, so this opt-in diagnostic profile disables it explicitly.
test.use({ trace: "off" });

test.describe("opt-in IPC performance diagnosis", () => {
  test.skip(
    !PROFILE_ENABLED,
    "Set JGS2_IPC_PROFILE=1 to run the serialized hardware profiles.",
  );

  for (const profileCase of PROFILE_CASES) {
    test(profileCase.id, async ({ page }, testInfo) => {
      test.setTimeout(180_000);
      await page.goto(`/?${profileCase.query}&test=1`, {
        waitUntil: "domcontentloaded",
      });
      const adapter = await requireHardwareWebGPU(page);
      await page.waitForFunction(
        () => typeof window.__jgs2Test?.profileCombinedFrames === "function",
      );
      await page.evaluate(async () => window.__jgs2Test!.ready);
      if ("preprofileFrames" in profileCase) {
        await page.evaluate(
          async (frameCount) => window.__jgs2Test!.stepFrames(frameCount),
          profileCase.preprofileFrames,
        );
      }

      const result = await page.evaluate(async (options) => {
        const harness = window.__jgs2Test!;
        const profile = await harness.profileCombinedFrames(options);
        const diagnostics = await harness.diagnostics();
        return {
          cpu: profile.cpuProfile.samples,
          gpu: profile.gpuProfile.samples,
          timestamp: profile.gpuProfile.timestamp,
          configuration: harness.configuration(),
          diagnostics: {
            finite: diagnostics.finite,
            lastStepIterations: diagnostics.lastStepIterations,
            activeContactCount: diagnostics.activeContactCount,
            activeContactCountValid: diagnostics.activeContactCountValid,
            minimumContactDistance: diagnostics.minimumContactDistance,
            minimumContactDistanceValid:
              diagnostics.minimumContactDistanceValid,
          },
          submissions: harness.submissionPolicy(),
        };
      }, PROFILE_OPTIONS);

      const report = {
        scene: profileCase.id,
        adapter,
        options: PROFILE_OPTIONS,
        configuration: result.configuration,
        diagnostics: result.diagnostics,
        submissions: result.submissions,
        wallFrame: summarize(result.cpu.endToEndFrameMilliseconds),
        cpuSimulationSubmission: summarize(
          result.cpu.cpuSimulationSubmissionMilliseconds,
        ),
        cpuRenderSubmission: summarize(
          result.cpu.cpuRenderSubmissionMilliseconds,
        ),
        gpuFrame: summarize(result.gpu.gpuFrameMilliseconds),
        gpuSimulation: summarize(result.gpu.gpuSimulationStepMilliseconds),
        gpuRender: summarize(result.gpu.gpuRenderMilliseconds),
        timestamp: result.timestamp,
        samples: result,
      };

      expect(result.diagnostics.finite).toBe(true);
      expect(typeof result.diagnostics.activeContactCountValid).toBe(
        "boolean",
      );
      expect(typeof result.diagnostics.minimumContactDistanceValid).toBe(
        "boolean",
      );
      if ("expectedConfiguration" in profileCase) {
        expect(result.configuration.ipcActivationDistance).toBe(
          profileCase.expectedConfiguration.ipcActivationDistance,
        );
        expect(result.configuration.maxStep).toBe(
          profileCase.expectedConfiguration.maxStep,
        );
        expect(result.diagnostics.lastStepIterations).toBe(
          profileCase.expectedConfiguration.lastStepIterations,
        );
      }
      expect(result.cpu.endToEndFrameMilliseconds).toHaveLength(
        PROFILE_OPTIONS.measuredFrameCount,
      );
      expect(result.submissions.maximumOutstanding).toBeLessThanOrEqual(2);
      expect(result.timestamp.timestampMapCount).toBe(
        result.timestamp.supported ? 1 : 0,
      );
      if (result.timestamp.supported) {
        expect(result.gpu.gpuSimulationStepMilliseconds).toHaveLength(
          PROFILE_OPTIONS.measuredFrameCount,
        );
        if (
          profileCase.id === "contact-friction" ||
          profileCase.id === "trough-friction" ||
          profileCase.id === "cloth-friction" ||
          profileCase.id === "cloth-settled"
        ) {
          const timestepBudgetMilliseconds = profileCase.id.startsWith("cloth")
            ? 1000 / 120
            : 1000 / 60;
          expect(report.gpuSimulation.mean).not.toBeNull();
          expect(report.gpuSimulation.mean!).toBeLessThan(
            timestepBudgetMilliseconds,
          );
        }
      }

      console.log(
        `[ipc-profile] ${profileCase.id}: wall ${format(report.wallFrame.mean)} ms, ` +
          `CPU submit ${format(report.cpuSimulationSubmission.mean)} ms, ` +
          `GPU simulation ${format(report.gpuSimulation.mean)} ms, ` +
          `${result.configuration.contactCandidateCount} candidates, ` +
          `${result.diagnostics.lastStepIterations} iterations`,
      );
      await testInfo.attach(`${profileCase.id}-performance.json`, {
        body: Buffer.from(`${JSON.stringify(report, null, 2)}\n`),
        contentType: "application/json",
      });
    });
  }

  for (const productionCase of PRODUCTION_CASES) {
    test(`${productionCase.id}-production-rate`, async ({ page }, testInfo) => {
      test.setTimeout(90_000);
      if ("emulatedFrameIntervalMilliseconds" in productionCase) {
        await page.addInitScript((frameIntervalMilliseconds) => {
          let nextFrameRequest = 1;
          let nextVsync: number | null = null;
          let timer: number | null = null;
          const pendingFrames = new Map<number, FrameRequestCallback>();
          const schedule = () => {
            if (timer !== null || pendingFrames.size === 0) {
              return;
            }
            const now = performance.now();
            nextVsync ??= now + frameIntervalMilliseconds;
            while (nextVsync <= now) {
              nextVsync += frameIntervalMilliseconds;
            }
            const target = nextVsync;
            const fire = () => {
              const actual = performance.now();
              if (actual < target) {
                timer = window.setTimeout(fire, target - actual);
                return;
              }
              timer = null;
              let delivered = target;
              while (
                delivered + frameIntervalMilliseconds <= actual
              ) {
                delivered += frameIntervalMilliseconds;
              }
              nextVsync = delivered + frameIntervalMilliseconds;
              const callbacks = [...pendingFrames.values()];
              pendingFrames.clear();
              for (const pendingCallback of callbacks) {
                pendingCallback(delivered);
              }
              schedule();
            };
            timer = window.setTimeout(fire, Math.max(0, target - now));
          };
          window.requestAnimationFrame = (callback: FrameRequestCallback) => {
            const request = nextFrameRequest;
            nextFrameRequest += 1;
            pendingFrames.set(request, callback);
            schedule();
            return request;
          };
          window.cancelAnimationFrame = (request: number) => {
            pendingFrames.delete(request);
            if (pendingFrames.size === 0 && timer !== null) {
              window.clearTimeout(timer);
              timer = null;
            }
          };
        }, productionCase.emulatedFrameIntervalMilliseconds);
      }
      await page.goto(`/?scene=${productionCase.scene}`, {
        waitUntil: "domcontentloaded",
      });
      const adapter = await requireHardwareWebGPU(page);
      const performanceHud = page.getByTestId("live-performance");
      await expect
        .poll(
          async () =>
            Number(
              (await performanceHud.getAttribute("data-sample-count")) ?? 0,
            ),
          { timeout: 30_000 },
        )
        .toBeGreaterThanOrEqual(productionCase.minimumSamples);

      const simulationRate = Number(
        await page.getByTestId("live-simulation-rate").textContent(),
      );
      const producedFps = Number(
        await page.getByTestId("live-fps").textContent(),
      );
      await expect
        .poll(async () => performanceHud.getAttribute("data-gpu-timing"), {
          timeout: 15_000,
        })
        .toBe("available");
      const gpuStepMilliseconds = Number(
        await page.getByTestId("live-gpu-step-ms").textContent(),
      );
      const gpuFrameMilliseconds = Number(
        await page.getByTestId("live-gpu-frame-ms").textContent(),
      );
      const frameMilliseconds = Number(
        await page.getByTestId("live-frame-ms").textContent(),
      );
      const report = {
        adapter,
        producedFps,
        simulationRate,
        frameMilliseconds,
        gpuStepMilliseconds,
        gpuFrameMilliseconds,
      };
      expect(Number.isFinite(simulationRate)).toBe(true);
      expect(simulationRate).toBeGreaterThanOrEqual(0.95);
      expect(gpuStepMilliseconds).toBeLessThan(
        productionCase.timestep * 1000,
      );
      if ("emulatedFrameIntervalMilliseconds" in productionCase) {
        expect(producedFps).toBeGreaterThanOrEqual(57);
        expect(producedFps).toBeLessThanOrEqual(63);
        const simulationStepsPerProducedFrame =
          simulationRate / (productionCase.timestep * producedFps);
        expect(simulationStepsPerProducedFrame).toBeGreaterThan(1.9);
        expect(simulationStepsPerProducedFrame).toBeLessThan(2.1);
        expect(gpuFrameMilliseconds).toBeLessThan(
          productionCase.emulatedFrameIntervalMilliseconds,
        );
      }
      console.log(
        `[ipc-profile] ${productionCase.id} production: ` +
          `${producedFps.toFixed(1)} fps, ` +
          `${simulationRate.toFixed(2)}x real time, ` +
          `${gpuStepMilliseconds.toFixed(3)} ms GPU/step, ` +
          `${gpuFrameMilliseconds.toFixed(3)} ms GPU/frame`,
      );
      await testInfo.attach(`${productionCase.id}-production-rate.json`, {
        body: Buffer.from(`${JSON.stringify(report, null, 2)}\n`),
        contentType: "application/json",
      });
    });
  }
});

function summarize(values: readonly number[]) {
  if (values.length === 0) {
    return { count: 0, mean: null, median: null, p95: null, min: null, max: null };
  }
  const sorted = [...values].sort((a, b) => a - b);
  return {
    count: values.length,
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    median: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    min: sorted[0]!,
    max: sorted[sorted.length - 1]!,
  };
}

function percentile(sorted: readonly number[], quantile: number): number {
  return sorted[Math.ceil(sorted.length * quantile) - 1]!;
}

function format(value: number | null): string {
  return value === null ? "N/A" : value.toFixed(3);
}

async function requireHardwareWebGPU(page: import("@playwright/test").Page) {
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
  expect(adapter).not.toBeNull();
  expect(adapter!.fallback).toBe(false);
  expect(adapter!.description).not.toMatch(/swiftshader/i);
  return adapter!.description || "hardware adapter";
}
