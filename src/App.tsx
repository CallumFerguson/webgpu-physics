import { useEffect, useRef, useState } from "react";

import {
  CanvasRenderTarget,
  requestWebGPUDevice,
  TriangleRenderer,
} from "./renderer";

type RenderStatus = "starting" | "complete" | "error";

async function renderTriangle(canvas: HTMLCanvasElement): Promise<void> {
  const { device, gpu } = await requestWebGPUDevice();
  const format = gpu.getPreferredCanvasFormat();
  const target = CanvasRenderTarget.create(device, canvas, format);
  const renderer = TriangleRenderer.create(device, format);

  await renderer.render(target);
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<RenderStatus>("starting");
  const [error, setError] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    let active = true;
    const completion = renderTriangle(canvas);
    window.__webgpuRenderDone = completion;

    void completion.then(
      () => {
        if (active) {
          setStatus("complete");
        }
      },
      (reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : String(reason));
          setStatus("error");
        }
      },
    );

    return () => {
      active = false;
    };
  }, []);

  return (
    <main className="app-shell">
      <section className="demo-card" aria-labelledby="demo-title">
        <p className="eyebrow">RAW WEBGPU · WGSL</p>
        <h1 id="demo-title">A deterministic triangle</h1>
        <p className="description">
          One render pipeline, one draw call, and no animation or randomness.
        </p>

        <div className="canvas-frame">
          <canvas
            ref={canvasRef}
            data-testid="gpu-canvas"
            width={320}
            height={320}
            aria-label="A WebGPU-rendered teal triangle"
          />
        </div>

        <p className={`status status--${status}`} role="status">
          {status === "starting" && "Initializing WebGPU…"}
          {status === "complete" && "WebGPU render complete"}
          {status === "error" && `WebGPU error: ${error}`}
        </p>
      </section>
    </main>
  );
}
