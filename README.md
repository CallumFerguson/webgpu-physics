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
- `/?scene=contact`
- `/?scene=trough` (use `&boxes=1` through `&boxes=20`; default `10`)
- `/?scene=cloth`
- `/?scene=stress`

Append `&schedule=graph-colored-gauss-seidel` to compare the paper's
CPU-colored GGS2 update schedule with the default parallel Jacobi schedule.
The older `stiffness` and `drop` URLs remain available as regression fixtures
but are intentionally hidden from the five-scene capability rail.

## Useful commands

```sh
npm run build
npm run test:unit
npm run test:e2e
npm run test:e2e:full
```

Use focused Vitest or Playwright files during development. `test:e2e` runs the
hardware suite with lightweight telemetry; `test:e2e:full` selects the longer
performance-qualification profile through its npm lifecycle name.

## Current implementation

The repository already includes implicit tetrahedral elastodynamics, the
paper's full-coordinate JGS2 basis, co-rotated subspaces, nonnegative Cubature,
co-rotated linear and stable Neo-Hookean materials, nonlinear GPU local solves,
globalization, external force/target objectives, topology-filtered VT/EE IPC
contact, lagged friction, StVK triangle cloth with quadratic-dihedral bending,
parallel Jacobi and graph-colored Gauss-Seidel schedules, deterministic frame
stepping, screenshots, and CPU/GPU performance metrics.

Per-frame simulation and rendering stay on the GPU. Small, one-time scene
precomputation remains in TypeScript on the CPU. WebAssembly is intentionally
not used because it would not improve the GPU-resident hot loop for these fixed
small scenes.

The contact, trough, and cloth demos use the mesh IPC path; the analytic
penalty plane remains only for the small non-IPC comparison scenes.

## Capability demos

| Public scene | Main result | Paper capabilities exercised |
| --- | --- | --- |
| `minimal` | A forced stable-Neo-Hookean cantilever follows and releases a scripted soft target. | Variational implicit Euler, full-coordinate JGS2 bases, nonnegative Cubature, nonlinear globalization, force/target objectives, Jacobi/GGS2 selection |
| `contact` | Deformable blocks collide and slide while a same-body pair exercises self-contact. | VT/EE candidates, IPC barrier and collision-safe step, lagged friction |
| `trough` | A configurable set of deformable boxes falls and piles up in a connected pinned V-trough. | Multi-body VT/EE IPC, collision-safe stepping, lagged friction, GPU-local candidate rows |
| `cloth` | A pinned pre-folded sheet drapes over a fixed collider without crossing itself. | StVK triangle stretching, quadratic-dihedral bending, IPC and friction |
| `stress` | Six alternating soft/stiff bodies fall in parallel. | Co-rotated subspaces, mixed materials/body IDs, Jacobi/GGS2 scheduling, live CPU/GPU performance HUD |

## Scope and limitations

- Scene preprocessing and greedy schedule coloring are CPU-side and static.
- IPC uses a topology-filtered static candidate superset with GPU AABB culling,
  suitable for these small fixed demos; there is no dynamic candidate
  generation or GPU BVH.
- The colored schedule is deliberately bounded and unoptimized, not a general
  dynamic scheduler for million-element meshes.
- The app targets hardware WebGPU in current Chrome. GPU timing fields report
  `N/A` when timestamp queries are unavailable.
- Publication-scale preprocessing, benchmark parity, dynamic remeshing, and a
  general-purpose physics-engine API are outside this project's scope.

See the [implementation roadmap](docs/ROADMAP.md) for completed work records,
acceptance evidence, scope decisions, and minimal-testing rules.
