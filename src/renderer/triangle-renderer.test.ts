import { describe, expect, it, vi } from "vitest";

import type { RenderTarget } from "./render-target";
import { TriangleRenderer, TRIANGLE_VERTEX_COUNT } from "./triangle-renderer";

function deferredPromise() {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
}

function createHarness(workDone: Promise<void> = Promise.resolve()) {
  const pass = {
    setPipeline: vi.fn(),
    draw: vi.fn(),
    end: vi.fn(),
  };
  const commandBuffer = {} as GPUCommandBuffer;
  const encoder = {
    beginRenderPass: vi.fn(() => pass),
    finish: vi.fn(() => commandBuffer),
  };
  const pipeline = {} as GPURenderPipeline;
  const device = {
    createShaderModule: vi.fn(() => ({})),
    createRenderPipeline: vi.fn(() => pipeline),
    createCommandEncoder: vi.fn(() => encoder),
    queue: {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => workDone),
    },
  } as unknown as GPUDevice;
  const target = {
    kind: "texture",
    format: "rgba8unorm",
    acquireView: vi.fn(() => ({}) as GPUTextureView),
  } satisfies RenderTarget;

  return { device, encoder, pass, pipeline, target };
}

describe("TriangleRenderer", () => {
  it("encodes one triangle and submits it", async () => {
    const harness = createHarness();
    const renderer = TriangleRenderer.create(harness.device, "rgba8unorm");

    await renderer.render(harness.target);

    expect(harness.pass.setPipeline).toHaveBeenCalledWith(harness.pipeline);
    expect(harness.pass.draw).toHaveBeenCalledOnce();
    expect(harness.pass.draw).toHaveBeenCalledWith(TRIANGLE_VERTEX_COUNT);
    expect(harness.pass.end).toHaveBeenCalledOnce();
    expect(harness.device.queue.submit).toHaveBeenCalledWith([
      expect.anything(),
    ]);
    expect(harness.device.queue.onSubmittedWorkDone).toHaveBeenCalledOnce();
  });

  it("does not finish rendering before submitted GPU work is complete", async () => {
    const work = deferredPromise();
    const harness = createHarness(work.promise);
    const renderer = TriangleRenderer.create(harness.device, "rgba8unorm");
    let settled = false;

    const rendering = renderer.render(harness.target).then(() => {
      settled = true;
    });
    await Promise.resolve();

    expect(settled).toBe(false);

    work.resolve();
    await rendering;

    expect(settled).toBe(true);
  });

  it("rejects a target with a format that does not match the pipeline", async () => {
    const harness = createHarness();
    const renderer = TriangleRenderer.create(harness.device, "rgba8unorm");
    const incompatibleTarget = {
      ...harness.target,
      format: "bgra8unorm",
    } satisfies RenderTarget;

    await expect(renderer.render(incompatibleTarget)).rejects.toThrow(
      "Render target format bgra8unorm does not match pipeline format rgba8unorm.",
    );
    expect(harness.device.createCommandEncoder).not.toHaveBeenCalled();
  });
});
