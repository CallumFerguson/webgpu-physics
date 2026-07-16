# WebGPU physics

## Setup

Use a current Node.js release and an installed Google Chrome build with working
hardware WebGPU support, then install the JavaScript dependencies:

```sh
npm install
```

## Commands

```sh
npm run dev                    # start the Vite development server
npm run build                  # type-check and create a production build
npm run test:unit              # run Vitest once
npm run test:screenshot        # run the headless Chromium WebGPU screenshot test
```

The Playwright project runs the installed Google Chrome build headlessly with
hardware acceleration enabled and software rasterization disabled. The
renderer rejects SwiftShader and every adapter Chrome marks as a fallback. The
test fails directly when Chrome, `navigator.gpu`, or a hardware adapter is
unavailable; otherwise it waits for `GPUQueue.onSubmittedWorkDone()`, waits for
canvas presentation, and saves a canvas-only screenshot under `test-results`.
The test prints the full artifact path and passes without comparing the image
to a baseline, provided the page reports no errors and a non-empty screenshot
is produced.

## Renderer layout

React only owns the canvas and the visible status message. The code under
`src/renderer` owns device creation, WGSL, pipeline setup, and command
submission. `TriangleRenderer.render()` accepts a `RenderTarget`; the included
targets wrap either a visible `GPUCanvasContext` or an offscreen `GPUTexture`.
Both targets use the same pipeline and draw path.
