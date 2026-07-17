# JGS2 WebGPU elastodynamics

A small WebGPU implementation of the capabilities in
[*JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU
Elastodynamics*](https://arxiv.org/abs/2506.06494) by Lan et al. (2025).

The project targets the current development computer and current Chrome. It is
intended to demonstrate the paper's methods in small real-time scenes, not to
reproduce its million-element benchmarks or serve as a portable physics engine.

## Run

```sh
npm install
npm run dev
```

Open the URL printed by Vite in a Chrome build with hardware WebGPU support.
The selected scene starts automatically and the on-screen HUD reports live
performance metrics.

Current scene URLs are:

- `/?scene=minimal`
- `/?scene=stiffness`
- `/?scene=drop`
- `/?scene=stress`

## Useful commands

```sh
npm run build
npm run test:unit
npm run test:e2e
npm run test:e2e:full
```

Use focused Vitest or Playwright files during development. `test:e2e:full` is
the old exhaustive tier and is not part of normal roadmap work.

## Current implementation

The repository already includes implicit tetrahedral elastodynamics, the
paper's full-coordinate JGS2 basis, co-rotated subspaces, nonnegative Cubature,
co-rotated linear and stable Neo-Hookean materials, nonlinear GPU local solves,
globalization, external force/target objectives, deterministic frame stepping,
screenshots, and CPU/GPU performance metrics.

Per-frame simulation and rendering stay on the GPU. Small, one-time scene
precomputation remains in TypeScript on the CPU. WebAssembly is intentionally
not used because it would not improve the GPU-resident hot loop for these fixed
small scenes.

Mesh IPC contact, friction, cloth, and colored Gauss-Seidel remain to be added.
The current floor is a penalty plane, not full IPC contact.

See the [implementation roadmap](docs/ROADMAP.md) for the remaining work,
strict timeboxes, acceptance criteria, and minimal-testing rules.
