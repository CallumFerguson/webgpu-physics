import { useEffect, useMemo, useRef, useState } from "react";

import { requestWebGPUDevice } from "./renderer";
import { SceneRenderer, type SceneCamera } from "./rendering/scene-renderer";
import {
  DEFAULT_SCENE_ID,
  SCENE_IDS,
  buildScene,
  buildSceneDefinition,
  toJGS2GpuInput,
  type SceneId,
} from "./scenes";
import { JGS2GpuSolver } from "./simulation/gpu";
import {
  diagnosticsFromState,
  type JGS2Diagnostics,
} from "./simulation/diagnostics";

interface JGS2TestHarness {
  readonly ready: Promise<void>;
  stepFrames(frameCount: number): Promise<void>;
  /** Advance one complete frame with exactly iterationCount nonlinear solves. */
  stepIterations(iterationCount: number): Promise<void>;
  waitForGpu(): Promise<void>;
  diagnostics(): Promise<JGS2Diagnostics>;
}

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

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneId = useMemo(requestedSceneId, []);
  const sceneDefinition = useMemo(() => buildSceneDefinition(sceneId), [sceneId]);
  const testMode = useMemo(
    () => new URLSearchParams(window.location.search).get("test") === "1",
    [],
  );
  const parityMode = useMemo(
    () => new URLSearchParams(window.location.search).get("parity") === "1",
    [],
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
    let gpuDevice: GPUDevice | undefined;
    let solver: JGS2GpuSolver | undefined;
    let renderer: SceneRenderer | undefined;
    let installedHarness: JGS2TestHarness | undefined;

    const initialize = async (): Promise<void> => {
      const precomputeStart = performance.now();
      const scene = buildScene(sceneId);
      const gpuInput = toJGS2GpuInput(scene);
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
      solver = await JGS2GpuSolver.create(device, gpuInput, {
        timestep: scene.settings.timestep,
        gravity: scene.settings.gravity,
        iterations: scene.settings.solverIterations,
        floorHeight: scene.settings.floorY,
        floorStiffness: 250_000,
        velocityDamping: 0.997,
        contactTangentialDamping: 12,
        contactMargin: 0.01,
        horizontalBodyCorrection: true,
        parityMode,
        regularization: 1e-6,
        rotationEpsilon: 1e-7,
        maxStep: 0.075,
      });
      if (!active) {
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
          floorHeight: scene.settings.floorY,
        },
        solver.currentPositionBuffer,
        solver.currentPositionByteOffset / 16,
        camera,
      );
      await renderer.renderAndWait(0);
      if (!active) {
        renderer.destroy();
        solver.destroy();
        device.destroy();
        return;
      }

      let simulationFrame = 0;
      const readDiagnostics = async (): Promise<JGS2Diagnostics> => {
        const [positions, velocities] = await Promise.all([
          solver!.readPositions(),
          solver!.readVelocities(),
        ]);
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

      if (testMode) {
        installedHarness = {
          ready: Promise.resolve(),
          stepFrames: async (frameCount: number) => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            if (!Number.isSafeInteger(frameCount) || frameCount < 0) {
              throw new RangeError("frameCount must be a nonnegative integer.");
            }
            for (let step = 0; step < frameCount; step += 1) {
              solver!.step();
              simulationFrame += 1;
            }
            await renderer!.renderAndWait(simulationFrame);
            setFrame(simulationFrame);
          },
          stepIterations: async (iterationCount: number) => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            solver!.stepExactIterations(iterationCount);
            simulationFrame += 1;
            await renderer!.renderAndWait(simulationFrame);
            setFrame(simulationFrame);
          },
          waitForGpu: async () => {
            if (!submissionsEnabled) {
              throw new Error("The WebGPU device is no longer available.");
            }
            await solver!.awaitIdle();
          },
          diagnostics: readDiagnostics,
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
          let submitted = false;
          for (
            let step = 0;
            step < 3 && accumulator >= frameDuration && inFlightBatches < 2;
            step += 1
          ) {
            solver!.step();
            simulationFrame += 1;
            accumulator -= frameDuration;
            submitted = true;
          }
          if (submitted) {
            renderer!.render(simulationFrame);
            inFlightBatches += 1;
            void solver!.awaitIdle().then(
              () => {
                inFlightBatches = Math.max(0, inFlightBatches - 1);
              },
              (reason: unknown) => {
                inFlightBatches = Math.max(0, inFlightBatches - 1);
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
      solver?.destroy();
      gpuDevice?.destroy();
    };
  }, [sceneId, testMode]);

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
