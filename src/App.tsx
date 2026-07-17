import { useEffect, useMemo, useRef, useState } from "react";

import { requestWebGPUDevice } from "./renderer";
import { SceneRenderer, type SceneCamera } from "./rendering/scene-renderer";
import {
  validateJGS2PerformanceProfileOptions,
  type JGS2CpuFrameProfile,
  type JGS2GpuFrameProfile,
  type JGS2PerformanceProfileOptions,
  type JGS2PerformanceState,
} from "./performance";
import {
  DEFAULT_SCENE_ID,
  FORCE_FREE_CONSERVATION_FIXTURE_ID,
  PHASE0_FORCE_FREE_CORPUS_CASE_COUNT,
  PHASE0_FORCE_FREE_FRAME_COUNT,
  PHASE0_FORCE_FREE_ITERATIONS,
  SCENE_IDS,
  buildForceFreeConservationDefinition,
  buildForceFreeConservationScene,
  buildScene,
  buildSceneDefinition,
  generatePhase0ForceFreeInitialStateCorpus,
  toForceFreeConservationGpuInput,
  toJGS2GpuInput,
  type SceneId,
} from "./scenes";
import {
  JGS2_MAX_BATCH_FRAMES,
  GpuTimestampFrameProfiler,
  JGS2GpuSolver,
  type GpuTimestampMeasurement,
} from "./simulation/gpu";
import {
  diagnosticsFromState,
  type JGS2Diagnostics,
} from "./simulation/diagnostics";

interface JGS2TestHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  /** Advance one complete frame with exactly iterationCount nonlinear solves. */
  stepIterations(iterationCount: number): Promise<void>;
  /** Advance one frame and explicitly time only its GPU simulation commands. */
  timedStepFrame(): Promise<GpuTimestampMeasurement>;
  /** Profile serialized production-shaped frames on a freshly loaded scene. */
  profileCpuFrames(
    options: JGS2PerformanceProfileOptions,
  ): Promise<JGS2CpuFrameProfile>;
  /** Profile GPU simulation and rendering with one timestamp-buffer map. */
  profileGpuFrames(
    options: JGS2PerformanceProfileOptions,
  ): Promise<JGS2GpuFrameProfile>;
  waitForGpu(): Promise<void>;
  diagnostics(): Promise<JGS2Diagnostics>;
  diagnosticReadbackCount(): number;
  configuration(): JGS2TestConfiguration;
  submissionPolicy(): JGS2SubmissionPolicy;
  runForceFreeCorpus(): Promise<readonly Phase0ForceFreeCorpusResult[]>;
  /** Test-only camera follow used to make a translated final pose visible. */
  focusOnPrimaryBody(): Promise<readonly [number, number, number]>;
}

interface JGS2TestConfiguration {
  readonly fixtureId: typeof FORCE_FREE_CONSERVATION_FIXTURE_ID | null;
  readonly gravity: readonly [number, number, number];
  readonly floorStiffness: number;
  readonly parityMode: boolean;
  readonly velocityDamping: number;
  readonly contactTangentialDamping: number;
  readonly horizontalBodyCorrection: boolean;
}

interface JGS2SubmissionPolicy {
  readonly maximumOutstanding: number;
  readonly currentOutstanding: number;
  readonly solverSubmissions: number;
  readonly renderSubmissions: number;
  readonly readbackSubmissions: number;
  readonly testBatchFrameLimit: number;
  readonly solverBatchFrameLimit: number;
  readonly productionStepsPerSubmission: 1;
}

interface Phase0ForceFreeCorpusResult {
  readonly id: string;
  readonly finite: boolean;
  readonly minimumDeterminant: number;
  readonly linearMomentumError: number;
  readonly angularMomentumError: number;
}

const TEST_BATCH_FRAME_LIMIT = 120;

declare global {
  interface Window {
    __jgs2Test?: JGS2TestHarness;
  }
}

type AppPhase = "preparing" | "gpu" | "ready" | "error";

interface RuntimeStats {
  readonly vertices: number;
  readonly tetrahedra: number;
  readonly precomputeMilliseconds: number;
  readonly adapter: string;
  readonly iterations: number;
  readonly cubatureSamples: number;
}

const sceneLabels: Record<SceneId, string> = {
  minimal: "Minimal beam",
  stiffness: "Soft / stiff",
  drop: "Floor impact",
  stress: "Stress test",
};

function requestedSceneId(): SceneId {
  const requested = new URLSearchParams(window.location.search).get("scene");
  return SCENE_IDS.includes(requested as SceneId)
    ? (requested as SceneId)
    : DEFAULT_SCENE_ID;
}

function adapterDescription(info: GPUAdapterInfo): string {
  return [info.vendor, info.architecture, info.device, info.description]
    .filter((value, index, all) => value && all.indexOf(value) === index)
    .join(" / ") || "Hardware WebGPU adapter";
}

function phaseLabel(phase: AppPhase): string {
  switch (phase) {
    case "preparing":
      return "Building rest Hessian and Cubature samples";
    case "gpu":
      return "Compiling WebGPU compute pipelines";
    case "ready":
      return "Running on the GPU";
    case "error":
      return "WebGPU initialization failed";
  }
}

function vectorNorm(vector: readonly number[]): number {
  return Math.hypot(...vector);
}

function vectorDifferenceNorm(
  left: readonly number[],
  right: readonly number[],
): number {
  return Math.hypot(
    (left[0] ?? 0) - (right[0] ?? 0),
    (left[1] ?? 0) - (right[1] ?? 0),
    (left[2] ?? 0) - (right[2] ?? 0),
  );
}

function boundsDiagonal(diagnostics: JGS2Diagnostics): number {
  return Math.hypot(
    diagnostics.bounds.max[0] - diagnostics.bounds.min[0],
    diagnostics.bounds.max[1] - diagnostics.bounds.min[1],
    diagnostics.bounds.max[2] - diagnostics.bounds.min[2],
  );
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneId = useMemo(requestedSceneId, []);
  const testMode = useMemo(
    () => new URLSearchParams(window.location.search).get("test") === "1",
    [],
  );
  const conservationFixture = useMemo(
    () =>
      testMode &&
      new URLSearchParams(window.location.search).get("fixture") ===
        FORCE_FREE_CONSERVATION_FIXTURE_ID,
    [testMode],
  );
  const sceneDefinition = useMemo(
    () =>
      conservationFixture
        ? buildForceFreeConservationDefinition()
        : buildSceneDefinition(sceneId),
    [conservationFixture, sceneId],
  );
  const parityMode = useMemo(
    () =>
      conservationFixture ||
      new URLSearchParams(window.location.search).get("parity") === "1",
    [conservationFixture],
  );
  const [phase, setPhase] = useState<AppPhase>("preparing");
  const [error, setError] = useState("");
  const [frame, setFrame] = useState(0);
  const [stats, setStats] = useState<RuntimeStats | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    let active = true;
    let animationFrame = 0;
    let submissionsEnabled = true;
    let inFlightBatches = 0;
    let currentOutstandingSubmissions = 0;
    let maximumOutstandingSubmissions = 0;
    let solverSubmissions = 0;
    let renderSubmissions = 0;
    let readbackSubmissions = 0;
    let gpuDevice: GPUDevice | undefined;
    let solver: JGS2GpuSolver | undefined;
    let renderer: SceneRenderer | undefined;
    let frameProfiler: GpuTimestampFrameProfiler | undefined;
    let installedHarness: JGS2TestHarness | undefined;

    const recordSubmission = (
      kind: "solver" | "render" | "readback",
    ): void => {
      currentOutstandingSubmissions += 1;
      maximumOutstandingSubmissions = Math.max(
        maximumOutstandingSubmissions,
        currentOutstandingSubmissions,
      );
      if (kind === "solver") {
        solverSubmissions += 1;
      } else if (kind === "render") {
        renderSubmissions += 1;
      } else {
        readbackSubmissions += 1;
      }
      if (currentOutstandingSubmissions > 2) {
        throw new Error(
          "The App queued more than two GPU submissions without a drain.",
        );
      }
    };

    const recordQueueDrained = (): void => {
      currentOutstandingSubmissions = 0;
    };

    const initialize = async (): Promise<void> => {
      const precomputeStart = performance.now();
      const scene = conservationFixture
        ? buildForceFreeConservationScene()
        : buildScene(sceneId);
      const gpuInput = conservationFixture
        ? toForceFreeConservationGpuInput(scene)
        : toJGS2GpuInput(scene);
      const precomputeMilliseconds = performance.now() - precomputeStart;
      if (!active) {
        return;
      }

      setPhase("gpu");
      const { device, gpu, adapter } = await requestWebGPUDevice();
      gpuDevice = device;
      if (!active) {
        device.destroy();
        return;
      }
      const format = gpu.getPreferredCanvasFormat();
      const solverSettings = {
        timestep: scene.settings.timestep,
        gravity: scene.settings.gravity,
        iterations: scene.settings.solverIterations,
        floorHeight: scene.settings.floorY,
        floorStiffness: conservationFixture ? 0 : 250_000,
        velocityDamping: 0.997,
        contactTangentialDamping: 12,
        contactMargin: 0.01,
        horizontalBodyCorrection: true,
        parityMode,
        regularization: 1e-6,
        rotationEpsilon: 1e-7,
        maxStep: 0.075,
      } as const;
      solver = await JGS2GpuSolver.create(device, gpuInput, solverSettings);
      frameProfiler = new GpuTimestampFrameProfiler(device, "jgs2-e2e-frame");
      if (!active) {
        frameProfiler.destroy();
        solver.destroy();
        device.destroy();
        return;
      }

      const inferredExtent = Math.max(
        scene.camera.eye[0] - scene.camera.target[0],
        scene.camera.eye[1] - scene.settings.floorY,
        scene.camera.eye[2] - scene.camera.target[2],
        2,
      );
      const camera: SceneCamera = {
        eye: scene.camera.eye,
        target: scene.camera.target,
        floorCenter: [scene.camera.target[0], scene.camera.target[2]],
        floorScale: inferredExtent * 1.15,
      };
      renderer = SceneRenderer.create(
        device,
        canvas,
        format,
        {
          restPositions: gpuInput.vertexRest,
          vertexColors: gpuInput.vertexColors,
          surfaceTriangles: scene.surface.triangles,
          surfaceEdges: scene.surface.edges,
          // The canonical force-free manifest has no floor. Keep its decorative
          // plane disabled while the solver uses exactly zero contact.
          floorHeight: scene.settings.floorY,
          showFloor: !conservationFixture,
        },
        solver.currentPositionBuffer,
        solver.currentPositionByteOffset / 16,
        camera,
      );
      recordSubmission("render");
      await renderer.renderAndWait(0);
      recordQueueDrained();
      if (!active) {
        renderer.destroy();
        frameProfiler.destroy();
        solver.destroy();
        device.destroy();
        return;
      }

      let simulationFrame = 0;
      const readDiagnostics = async (): Promise<JGS2Diagnostics> => {
        recordSubmission("readback");
        const positionsPromise = solver!.readPositions();
        recordSubmission("readback");
        const velocitiesPromise = solver!.readVelocities();
        const [positions, velocities] = await Promise.all([
          positionsPromise,
          velocitiesPromise,
        ]);
        recordQueueDrained();
        const submittedSettings = solver!.lastSubmittedSettings;
        return diagnosticsFromState(
          scene,
          positions,
          velocities,
          {
            frame: simulationFrame,
            lastStepIterations: solver!.lastSubmittedIterationCount,
            runtime: {
              parityMode: submittedSettings.parityMode,
              velocityDamping: submittedSettings.velocityDamping,
              contactTangentialDamping:
                submittedSettings.contactTangentialDamping,
              horizontalBodyCorrection:
                submittedSettings.horizontalBodyCorrection,
            },
          },
        );
      };

      const readPerformanceState = async (): Promise<JGS2PerformanceState> => {
        recordSubmission("readback");
        const positionsPromise = solver!.readPositions();
        recordSubmission("readback");
        const velocitiesPromise = solver!.readVelocities();
        const [positions, velocities] = await Promise.all([
          positionsPromise,
          velocitiesPromise,
        ]);
        recordQueueDrained();
        return {
          positions: Array.from(positions),
          velocities: Array.from(velocities),
        };
      };

      if (testMode) {
        const renderAndDrain = async (): Promise<void> => {
          recordSubmission("render");
          await renderer!.renderAndWait(simulationFrame);
          recordQueueDrained();
          setFrame(simulationFrame);
        };

        const submitOneSimulationStep = (): void => {
          if (conservationFixture) {
            solver!.stepFramesExactIterations(
              1,
              PHASE0_FORCE_FREE_ITERATIONS,
            );
          } else {
            solver!.step();
          }
        };

        const advanceSerializedFrame = async (): Promise<{
          readonly endToEndMilliseconds: number;
          readonly cpuSimulationSubmissionMilliseconds: number;
          readonly cpuRenderSubmissionMilliseconds: number;
        }> => {
          const frameStart = performance.now();
          const simulationStart = performance.now();
          submitOneSimulationStep();
          const cpuSimulationSubmissionMilliseconds =
            performance.now() - simulationStart;
          recordSubmission("solver");
          simulationFrame += 1;

          const renderStart = performance.now();
          renderer!.render(simulationFrame);
          const cpuRenderSubmissionMilliseconds =
            performance.now() - renderStart;
          recordSubmission("render");
          await solver!.awaitIdle();
          recordQueueDrained();
          return {
            endToEndMilliseconds: performance.now() - frameStart,
            cpuSimulationSubmissionMilliseconds,
            cpuRenderSubmissionMilliseconds,
          };
        };

        const assertFreshPerformanceScene = (): void => {
          if (simulationFrame !== 0) {
            throw new Error(
              "Performance profiles require a freshly loaded scene at simulation frame zero.",
            );
          }
        };

        const profileIdentity = () => ({
          workloadId: conservationFixture
            ? FORCE_FREE_CONSERVATION_FIXTURE_ID
            : scene.id,
          timestepSeconds: scene.settings.timestep,
          iterationsPerStep: conservationFixture
            ? PHASE0_FORCE_FREE_ITERATIONS
            : scene.settings.solverIterations,
        });

        const runForceFreeCorpus = async (): Promise<
          readonly Phase0ForceFreeCorpusResult[]
        > => {
          if (!conservationFixture) {
            throw new Error(
              "The force-free corpus is only available on its canonical fixture.",
            );
          }
          const states = generatePhase0ForceFreeInitialStateCorpus();
          if (states.length !== PHASE0_FORCE_FREE_CORPUS_CASE_COUNT) {
            throw new Error("The frozen force-free corpus is incomplete.");
          }
          const results: Phase0ForceFreeCorpusResult[] = [];
          for (const state of states) {
            const corpusInput = toForceFreeConservationGpuInput(scene, state);
            const initial = diagnosticsFromState(
              scene,
              corpusInput.positions,
              corpusInput.velocities!,
              {
                frame: 0,
                lastStepIterations: 0,
                runtime: {
                  parityMode: true,
                  velocityDamping: 1,
                  contactTangentialDamping: 0,
                  horizontalBodyCorrection: false,
                },
              },
            );
            const corpusSolver = await JGS2GpuSolver.create(
              device,
              corpusInput,
              solverSettings,
            );
            let minimumDeterminant = initial.minTetDeterminant;
            let finite = initial.finite;
            let finalDiagnostics: JGS2Diagnostics | undefined;
            try {
              for (
                let completedFrames = 0;
                completedFrames < PHASE0_FORCE_FREE_FRAME_COUNT;
              ) {
                const batchFrames = Math.min(
                  TEST_BATCH_FRAME_LIMIT,
                  PHASE0_FORCE_FREE_FRAME_COUNT - completedFrames,
                );
                corpusSolver.stepFramesExactIterations(
                  batchFrames,
                  PHASE0_FORCE_FREE_ITERATIONS,
                );
                recordSubmission("solver");
                completedFrames += batchFrames;
                await corpusSolver.awaitIdle();
                recordQueueDrained();

                if (completedFrames === PHASE0_FORCE_FREE_FRAME_COUNT) {
                  recordSubmission("readback");
                  const positionsPromise = corpusSolver.readPositions();
                  recordSubmission("readback");
                  const velocitiesPromise = corpusSolver.readVelocities();
                  const [positions, velocities] = await Promise.all([
                    positionsPromise,
                    velocitiesPromise,
                  ]);
                  recordQueueDrained();
                  finalDiagnostics = diagnosticsFromState(
                    scene,
                    positions,
                    velocities,
                    {
                      frame: completedFrames,
                      lastStepIterations: PHASE0_FORCE_FREE_ITERATIONS,
                      runtime: {
                        parityMode: true,
                        velocityDamping: 1,
                        contactTangentialDamping: 0,
                        horizontalBodyCorrection: false,
                      },
                    },
                  );
                  minimumDeterminant = Math.min(
                    minimumDeterminant,
                    finalDiagnostics.minTetDeterminant,
                  );
                  finite &&= finalDiagnostics.finite;
                } else {
                  recordSubmission("readback");
                  const positions = await corpusSolver.readPositions();
                  recordQueueDrained();
                  const checkpoint = diagnosticsFromState(
                    scene,
                    positions,
                    corpusInput.velocities!,
                    {
                      frame: completedFrames,
                      lastStepIterations: PHASE0_FORCE_FREE_ITERATIONS,
                      runtime: {
                        parityMode: true,
                        velocityDamping: 1,
                        contactTangentialDamping: 0,
                        horizontalBodyCorrection: false,
                      },
                    },
                  );
                  minimumDeterminant = Math.min(
                    minimumDeterminant,
                    checkpoint.minTetDeterminant,
                  );
                  finite &&= checkpoint.finite;
                }
              }
            } finally {
              corpusSolver.destroy();
            }
            if (!finalDiagnostics) {
              throw new Error(`Corpus case ${state.id} did not reach its end.`);
            }
            const totalMass = initial.bodies.reduce(
              (sum, body) => sum + body.mass,
              0,
            );
            const sceneScale = boundsDiagonal(initial);
            results.push({
              id: state.id,
              finite,
              minimumDeterminant,
              linearMomentumError:
                vectorDifferenceNorm(
                  finalDiagnostics.totalLinearMomentum,
                  initial.totalLinearMomentum,
                ) /
                Math.max(
                  vectorNorm(initial.totalLinearMomentum),
                  totalMass * sceneScale,
                ),
              angularMomentumError:
                vectorDifferenceNorm(
                  finalDiagnostics.totalAngularMomentum,
                  initial.totalAngularMomentum,
                ) /
                Math.max(
                  vectorNorm(initial.totalAngularMomentum),
                  totalMass * sceneScale * sceneScale,
                ),
            });
          }
          return results;
        };

        installedHarness = {
          ready: Promise.resolve(),
          stepFrames: async (frameCount: number) => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            if (!Number.isSafeInteger(frameCount) || frameCount < 0) {
              throw new RangeError("frameCount must be a nonnegative integer.");
            }
            let remaining = frameCount;
            while (remaining > 0) {
              const batchFrames = Math.min(TEST_BATCH_FRAME_LIMIT, remaining);
              if (conservationFixture) {
                solver!.stepFramesExactIterations(
                  batchFrames,
                  PHASE0_FORCE_FREE_ITERATIONS,
                );
              } else {
                solver!.stepFrames(batchFrames);
              }
              recordSubmission("solver");
              simulationFrame += batchFrames;
              remaining -= batchFrames;
              if (remaining > 0) {
                await solver!.awaitIdle();
                recordQueueDrained();
              }
            }
            await renderAndDrain();
          },
          stepIterations: async (iterationCount: number) => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            solver!.stepExactIterations(iterationCount);
            recordSubmission("solver");
            simulationFrame += 1;
            await renderAndDrain();
          },
          timedStepFrame: async () => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            recordSubmission("solver");
            const measurement = await solver!.stepWithGpuTimestamp();
            recordQueueDrained();
            simulationFrame += 1;
            await renderAndDrain();
            return measurement;
          },
          profileCpuFrames: async (options) => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            validateJGS2PerformanceProfileOptions(options);
            assertFreshPerformanceScene();
            const initialSimulationFrame = simulationFrame;
            const diagnosticReadbacksBefore =
              solver!.explicitDiagnosticReadbackCount;
            for (
              let frameIndex = 0;
              frameIndex < options.warmupFrameCount;
              frameIndex += 1
            ) {
              await advanceSerializedFrame();
            }

            const endToEndFrameMilliseconds: number[] = [];
            const cpuSimulationSubmissionMilliseconds: number[] = [];
            const cpuRenderSubmissionMilliseconds: number[] = [];
            const cpuFrameSubmissionMilliseconds: number[] = [];
            for (
              let frameIndex = 0;
              frameIndex < options.measuredFrameCount;
              frameIndex += 1
            ) {
              const sample = await advanceSerializedFrame();
              endToEndFrameMilliseconds.push(sample.endToEndMilliseconds);
              cpuSimulationSubmissionMilliseconds.push(
                sample.cpuSimulationSubmissionMilliseconds,
              );
              cpuRenderSubmissionMilliseconds.push(
                sample.cpuRenderSubmissionMilliseconds,
              );
              cpuFrameSubmissionMilliseconds.push(
                sample.cpuSimulationSubmissionMilliseconds +
                  sample.cpuRenderSubmissionMilliseconds,
              );
            }
            setFrame(simulationFrame);
            if (
              solver!.explicitDiagnosticReadbackCount !==
              diagnosticReadbacksBefore
            ) {
              throw new Error(
                "CPU performance interval performed an unexpected diagnostic readback.",
              );
            }
            const finalState = await readPerformanceState();
            return {
              ...profileIdentity(),
              initialSimulationFrame,
              finalSimulationFrame: simulationFrame,
              warmupFrameCount: options.warmupFrameCount,
              measuredFrameCount: options.measuredFrameCount,
              diagnosticReadbacksBefore,
              diagnosticReadbacksAfter:
                solver!.explicitDiagnosticReadbackCount,
              finalState,
              samples: {
                endToEndFrameMilliseconds,
                cpuSimulationSubmissionMilliseconds,
                cpuRenderSubmissionMilliseconds,
                cpuFrameSubmissionMilliseconds,
              },
            } satisfies JGS2CpuFrameProfile;
          },
          profileGpuFrames: async (options) => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            validateJGS2PerformanceProfileOptions(options);
            assertFreshPerformanceScene();
            const initialSimulationFrame = simulationFrame;
            const diagnosticReadbacksBefore =
              solver!.explicitDiagnosticReadbackCount;
            for (
              let frameIndex = 0;
              frameIndex < options.warmupFrameCount;
              frameIndex += 1
            ) {
              await advanceSerializedFrame();
            }

            const measurement = await frameProfiler!.measureFrames(
              options.measuredFrameCount,
              async (writes) => {
                if (writes) {
                  if (conservationFixture) {
                    solver!.stepExactIterationsWithGpuTimestampWrites(
                      PHASE0_FORCE_FREE_ITERATIONS,
                      writes.simulation,
                    );
                  } else {
                    solver!.stepWithGpuTimestampWrites(writes.simulation);
                  }
                } else {
                  submitOneSimulationStep();
                }
                recordSubmission("solver");
                simulationFrame += 1;
                renderer!.render(simulationFrame, writes?.render);
                recordSubmission("render");
                await solver!.awaitIdle();
                recordQueueDrained();
              },
            );
            recordQueueDrained();
            setFrame(simulationFrame);
            if (
              solver!.explicitDiagnosticReadbackCount !==
              diagnosticReadbacksBefore
            ) {
              throw new Error(
                "GPU performance interval performed an unexpected diagnostic readback.",
              );
            }
            const finalState = await readPerformanceState();
            return {
              ...profileIdentity(),
              initialSimulationFrame,
              finalSimulationFrame: simulationFrame,
              warmupFrameCount: options.warmupFrameCount,
              measuredFrameCount: options.measuredFrameCount,
              diagnosticReadbacksBefore,
              diagnosticReadbacksAfter:
                solver!.explicitDiagnosticReadbackCount,
              finalState,
              timestamp: {
                feature: measurement.feature,
                supported: measurement.supported,
                featureEnabled: measurement.featureEnabled,
                reason: measurement.reason,
                timestampMapCount: measurement.timestampMapCount,
              },
              samples: {
                gpuFrameMilliseconds:
                  measurement.gpuFrameMilliseconds ?? [],
                gpuSimulationStepMilliseconds:
                  measurement.gpuSimulationStepMilliseconds ?? [],
                gpuRenderMilliseconds:
                  measurement.gpuRenderMilliseconds ?? [],
              },
            } satisfies JGS2GpuFrameProfile;
          },
          waitForGpu: async () => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            await solver!.awaitIdle();
            recordQueueDrained();
          },
          diagnostics: readDiagnostics,
          diagnosticReadbackCount: () =>
            solver!.explicitDiagnosticReadbackCount,
          configuration: () => {
            const submittedSettings = solver!.lastSubmittedSettings;
            return {
              fixtureId: conservationFixture
                ? FORCE_FREE_CONSERVATION_FIXTURE_ID
                : null,
              gravity: submittedSettings.gravity,
              floorStiffness: submittedSettings.floorStiffness,
              parityMode: submittedSettings.parityMode,
              velocityDamping: submittedSettings.velocityDamping,
              contactTangentialDamping:
                submittedSettings.contactTangentialDamping,
              horizontalBodyCorrection:
                submittedSettings.horizontalBodyCorrection,
            };
          },
          submissionPolicy: () => ({
            maximumOutstanding: maximumOutstandingSubmissions,
            currentOutstanding: currentOutstandingSubmissions,
            solverSubmissions,
            renderSubmissions,
            readbackSubmissions,
            testBatchFrameLimit: TEST_BATCH_FRAME_LIMIT,
            solverBatchFrameLimit: JGS2_MAX_BATCH_FRAMES,
            productionStepsPerSubmission: 1,
          }),
          focusOnPrimaryBody: async () => {
            if (!conservationFixture) {
              throw new Error(
                "The follow camera is only exposed for the force-free fixture.",
              );
            }
            const diagnostics = await readDiagnostics();
            const primary = diagnostics.bodies[0];
            if (!primary) {
              throw new Error("The force-free fixture has no primary body.");
            }
            const target = primary.centerOfMass;
            const eyeOffset = [
              camera.eye[0] - camera.target[0],
              camera.eye[1] - camera.target[1],
              camera.eye[2] - camera.target[2],
            ] as const;
            renderer!.setCamera({
              eye: [
                target[0] + eyeOffset[0],
                target[1] + eyeOffset[1],
                target[2] + eyeOffset[2],
              ],
              target,
              floorCenter: [target[0], target[2]],
              floorScale: camera.floorScale,
            });
            await renderAndDrain();
            return target;
          },
          runForceFreeCorpus,
        };
        window.__jgs2Test = installedHarness;
      } else {
        let previousTime = performance.now();
        let accumulator = 0;
        const frameDuration = scene.settings.timestep * 1000;
        const animate = (now: number) => {
          if (!active || !submissionsEnabled) {
            return;
          }
          accumulator = Math.min(
            accumulator + Math.min(now - previousTime, 100),
            frameDuration * 3,
          );
          previousTime = now;
          if (accumulator >= frameDuration && inFlightBatches === 0) {
            solver!.step();
            recordSubmission("solver");
            simulationFrame += 1;
            accumulator -= frameDuration;
            renderer!.render(simulationFrame);
            recordSubmission("render");
            inFlightBatches = 1;
            void solver!.awaitIdle().then(
              () => {
                inFlightBatches = 0;
                recordQueueDrained();
              },
              (reason: unknown) => {
                inFlightBatches = 0;
                recordQueueDrained();
                if (active && submissionsEnabled) {
                  submissionsEnabled = false;
                  cancelAnimationFrame(animationFrame);
                  setError(reason instanceof Error ? reason.message : String(reason));
                  setPhase("error");
                }
              },
            );
            if (simulationFrame % 12 === 0) {
              setFrame(simulationFrame);
            }
          }
          animationFrame = requestAnimationFrame(animate);
        };
        animationFrame = requestAnimationFrame(animate);
      }

      device.lost.then((info) => {
        if (active) {
          submissionsEnabled = false;
          cancelAnimationFrame(animationFrame);
          setError(`The WebGPU device was lost: ${info.message || info.reason}`);
          setPhase("error");
        }
      }).catch(() => undefined);
      setStats({
        vertices: gpuInput.vertexCount,
        tetrahedra: gpuInput.tetCount,
        precomputeMilliseconds,
        adapter: adapterDescription(adapter.info),
        iterations: solver.lastSubmittedIterationCount || scene.settings.solverIterations,
        cubatureSamples: gpuInput.cubatureK,
      });
      setPhase("ready");
    };

    void initialize().catch((reason: unknown) => {
      if (active) {
        submissionsEnabled = false;
        renderer?.destroy();
        renderer = undefined;
        frameProfiler?.destroy();
        frameProfiler = undefined;
        solver?.destroy();
        solver = undefined;
        gpuDevice?.destroy();
        gpuDevice = undefined;
        setError(reason instanceof Error ? reason.message : String(reason));
        setPhase("error");
      }
    });

    return () => {
      active = false;
      submissionsEnabled = false;
      cancelAnimationFrame(animationFrame);
      if (installedHarness && window.__jgs2Test === installedHarness) {
        delete window.__jgs2Test;
      }
      renderer?.destroy();
      frameProfiler?.destroy();
      solver?.destroy();
      gpuDevice?.destroy();
    };
  }, [conservationFixture, parityMode, sceneId, testMode]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href={`?scene=${DEFAULT_SCENE_ID}`}>
          <span className="brand-mark" aria-hidden="true">J²</span>
          <span>
            <strong>JGS2</strong>
            <small>WebGPU elastodynamics</small>
          </span>
        </a>
        <div className={`runtime-pill runtime-pill--${phase}`} role="status">
          <span className="runtime-dot" aria-hidden="true" />
          {phaseLabel(phase)}
        </div>
        <a
          className="paper-link"
          href="https://arxiv.org/abs/2506.06494"
          target="_blank"
          rel="noreferrer"
        >
          Paper ↗
        </a>
      </header>

      <div className="workspace">
        <aside className="scene-rail" aria-label="Demo scenes">
          <p className="rail-label">Scenes</p>
          <nav>
            {SCENE_IDS.map((id, index) => (
              <a
                className={id === sceneId ? "scene-link scene-link--active" : "scene-link"}
                href={`?scene=${id}`}
                key={id}
                aria-current={id === sceneId ? "page" : undefined}
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                {sceneLabels[id]}
              </a>
            ))}
          </nav>

          <div className="rail-note">
            <span>Runtime path</span>
            <strong>CPU precompute</strong>
            <i aria-hidden="true" />
            <strong>GPU solve + draw</strong>
            <small>No WASM or per-frame readback</small>
          </div>
        </aside>

        <section className="content">
          <div className="scene-heading">
            <div>
              <p className="eyebrow">Demo {String(SCENE_IDS.indexOf(sceneId) + 1).padStart(2, "0")} · Deterministic</p>
              <h1>{sceneDefinition.title}</h1>
            </div>
            <p>{sceneDefinition.description}</p>
          </div>

          <div className="canvas-shell">
            <canvas
              ref={canvasRef}
              data-testid="gpu-canvas"
              width={960}
              height={540}
              aria-label={`Real-time WebGPU simulation: ${sceneDefinition.title}`}
            />
            <div className="canvas-badge canvas-badge--top">
              <span>Implicit Euler</span>
              <span>JGS2 Jacobi</span>
            </div>
            <div className="canvas-legend" aria-label="Rendering legend">
              <span><i className="legend-live" />Live FEM surface</span>
              <span><i className="legend-rest" />Rest-shape reference</span>
            </div>
            {phase === "error" && (
              <div className="error-panel" role="alert">
                <strong>WebGPU error</strong>
                <p>{error}</p>
              </div>
            )}
          </div>

          <div className="metrics" aria-label="Simulation metrics">
            <div>
              <span>Frame</span>
              <strong>{frame.toLocaleString("en-US")}</strong>
            </div>
            <div>
              <span>Tetrahedra</span>
              <strong>{stats?.tetrahedra.toLocaleString("en-US") ?? "—"}</strong>
            </div>
            <div>
              <span>Vertex solves</span>
              <strong>{stats?.vertices.toLocaleString("en-US") ?? "—"}<small> parallel</small></strong>
            </div>
            <div>
              <span>JGS2 iterations</span>
              <strong>{stats?.iterations ?? sceneDefinition.settings.solverIterations}<small> / step</small></strong>
            </div>
            <div>
              <span>Cubature</span>
              <strong>{stats?.cubatureSamples ?? sceneDefinition.settings.cubatureSamples}<small> / vertex</small></strong>
            </div>
            <div>
              <span>CPU setup</span>
              <strong>{stats ? stats.precomputeMilliseconds.toFixed(1) : "—"}<small> ms</small></strong>
            </div>
          </div>

          <div className="details-grid">
            <article>
              <p className="detail-number">01</p>
              <div>
                <h2>Global awareness, local work</h2>
                <p>
                  Each three-coordinate vertex solve includes the response of remote
                  tetrahedra through a co-rotated equilibrium basis. The cyan outline
                  makes drift, sag, and impact immediately visible.
                </p>
              </div>
            </article>
            <article>
              <p className="detail-number">02</p>
              <div>
                <h2>Built for the browser GPU</h2>
                <p>
                  WGSL gathers incident elements, evaluates four or six Cubature
                  samples, solves a regularized 3 × 3 system, preserves each free
                  body's horizontal center of mass, and renders from the same buffer.
                </p>
              </div>
            </article>
          </div>

          {stats && <p className="adapter-line">Hardware adapter · {stats.adapter}</p>}
        </section>
      </div>
    </main>
  );
}
