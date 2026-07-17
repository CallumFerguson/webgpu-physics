# JGS2 WebGPU elastodynamics

A real-time browser implementation of the core algorithm in
[*JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU
Elastodynamics*](https://arxiv.org/abs/2506.06494) by Lan et al. (2025).

The app builds a tetrahedral implicit-Euler problem, precomputes the paper's
globally aware local perturbation bases and nonnegative Cubature weights, then
runs vertex solves entirely in WebGPU compute shaders. The runtime includes
co-rotated linear regression material and a CPU/GPU-validated stable
Neo-Hookean current material path with nonlinear current-pose Cubature. The
same GPU position buffer is rendered directly with no per-frame readback.

## Run it

Use a current Google Chrome build with hardware WebGPU support:

```sh
npm install
npm run dev
```

Then open the local URL printed by Vite. The default scene starts on its own;
no interaction is required.

```sh
npm run build             # strict TypeScript + production bundle
npm run test:unit         # FEM, basis, Cubature, and renderer tests
npm run test:baseline-manifest # verify every frozen test selector still exists
npm run test:e2e          # full hardware-WebGPU E2E, oracle, visual, and timing suite
```

Software WebGPU adapters, including SwiftShader, are deliberately rejected.
The real-time and hardware E2E tests must exercise a hardware adapter.
The E2E run prints Playwright's finalized duration for every non-skipped test
and writes every result, including skips, to the machine-readable
`test-results/playwright-results.json`, with per-attempt and total durations in
milliseconds. Listing tests does not overwrite the last real run report. Every
visual scene test also logs and attaches 600-sample serialized average FPS, 1%
low, wall-frame, CPU-submit, and (when `timestamp-query` is available) GPU
frame/step/render metrics. Serialized throughput is a necessary compute check;
it is not a measurement of production animation scheduling or simulation time
rate. Scene budget assessments are informational so unrelated correctness tests
do not become load-sensitive; the isolated performance-baseline test owns the
enforced hardware budget using sustained serialized wall throughput and GPU
frame p95 when timestamps are available. Wall p95 and 1% low remain visible
diagnostics.

## Demo scenes

| Scene | Purpose |
| --- | --- |
| Minimal beam | One tetrahedral cube with its left face fixed. The free face swings down under gravity. A persistent cyan rest outline makes correct motion obvious. |
| Soft / stiff | Identical teal and red cantilevers under the same load. The soft beam bends substantially farther. |
| Floor impact | A rotated, higher-resolution deformable block falls onto an implicit penalty floor. |
| Stress test | Six separated soft/stiff bodies advance together, exercising larger batched FEM, Cubature, and contact workloads. |

Direct URLs are `/?scene=minimal`, `/?scene=stiffness`, `/?scene=drop`, and
`/?scene=stress`.

## Paper implementation

For every movable vertex `i`, the CPU constructs the exact equilibrium
extension of its three local coordinates. It factors the rest timestep Hessian
once and evaluates

```text
Y_i    = Hbar^-1 S_i^T
Ubar_i = Y_i (S_i Y_i)^-1.
```

This is the paper's full-coordinate form of
`U_Ci = -H_CC^-1 H_Ci` (equations 19-23). For stable Neo-Hookean solids, both
signs of up to eight low-frequency rest-Hessian modes generate current-pose
training shapes. At every shape, candidates use the current material gradient,
exact tangent, and co-rotated basis `R_v Ubar_vi R_i^T`. A deterministic greedy
NNLS fit implements equation 18. The runtime retains up to four or six positive
samples per vertex in fixed-width GPU slots and discards the dense bases.

Each runtime Jacobi invocation owns one vertex and solves equation 15:

```text
(H_ii + Htilde_ii) delta_i = -(g_i + gtilde_i).
```

WGSL gathers exact incident-element terms, co-rotates each selected basis,
evaluates either co-rotated linear or stable Neo-Hookean current gradients and
Hessian-vector products, performs a regularized `3 x 3` Cholesky solve, and
writes to the opposite ping-pong region.
No floating-point atomics or materialized `12 x 12` runtime Hessians are needed.
Incident samples retain their neighbor/cross response while subtracting the
source block already included by the exact local gather. Non-parity demos may
run an optional GPU post-pass that restores each free body's mass-weighted
horizontal center of mass to its predicted value. Parity mode disables that
correction and grounded tangential damping.

The Float64 CPU reference exercises a deterministic 64-pose tetrahedron corpus
with finite-difference gradient and Hessian checks. Its exact quadratic oracle
also requires each local equilibrium-basis solve to equal the corresponding
block of the full Newton update. Hardware tests compare exact GPU energy,
gradient, and local Hessian data against that CPU reference, and compare all
64 poses times three active vertex blocks against the CPU equilibrium oracle.
Other tests check `S_i Ubar_i = I`, complementary equilibrium, ordered low
modes, nonnegative Cubature, mass conservation, surface topology, and
deterministic scene packing. The stable Cubature gate additionally requires
source plus all current candidates to reproduce an independently assembled
`B^T g` and `B^T H B`, a `<=1%` normalized training residual after `f32`
packing, and `<=2%` selected-versus-exact update RMS on both training and
held-out shapes. A production WebGPU test mirrors two exact Jacobi iterations,
including a nonzero inertial-gradient second pass, against the packed CPU
reference.

The checked-in nonlinear capability fixture has 12 full-rank candidates per
source and retains six. It proves the paper's nonlinear calculations and data
flow under genuine approximation, but it does not claim the preprocessing
scale of the paper's large examples.

## CPU, GPU, and WASM tradeoff

- CPU once per scene: mesh/topology construction, f64 rest FEM and lumped mass,
  one dense Cholesky for these small demos, low modes, three-right-hand-side
  equilibrium solves, NNLS Cubature, and GPU buffer packing.
- GPU every frame: implicit prediction, polar frames, co-rotated linear or
  stable Neo-Hookean current elasticity, Cubature projection, local solves,
  optional non-parity demo stabilization, floor penalty, optional non-parity
  grounded damping, velocity update, and rendering.
- CPU in tests only: explicitly requested synchronized diagnostic checkpoints
  for numeric invariants; normal production and benchmark stepping performs no
  synchronous per-frame readback.

WebAssembly is not used. It would not improve the GPU-resident hot loop and
would introduce another memory boundary. WASM becomes sensible only for
precomputing large arbitrary user meshes in the browser, where a threaded SIMD
sparse factorization/eigensolver could replace the small-scene TypeScript path.
Production fixed scenes would more simply ship offline-precomputed binary
assets.

See [the implementation notes](docs/jgs2-implementation.md) for equations,
data layout, numerical safeguards, and the choices required by details omitted
from the paper.

## Deterministic browser validation

In test mode (`?test=1`) the animation loop is disabled. The page exposes a
fixed-step test harness that:

1. renders and saves frame zero;
2. advances an exact number of implicit timesteps or nonlinear iterations;
3. waits for submitted GPU work and canvas presentation;
4. renders and saves the ending frame;
5. performs explicitly requested diagnostic readbacks and checks finiteness,
   fixed vertices, positive tetrahedron determinants, bounds, landmarks,
   momentum, and visible pixel change.

The 17-test Playwright gate covers all four public scenes, long-running
drop/stress behavior, and the visible WebGPU-unavailable path. It also exercises
the canonical GPU oracles, the base and 32-case force-free conservation corpus,
submission/readback invariants, and the timestamped performance baseline.
Start/end PNGs and generated JSON reports are retained under `test-results` for
inspection.

See the [reproducibility guide](docs/reproducibility.md), the checked-in
[Phase 0 performance evidence](docs/evidence/phase0-performance-baseline.md),
and the [capability-parity roadmap](docs/jgs2-capability-roadmap.md) for exact
commands, frozen manifests, criteria, and results.

## Scope

This repository implements the JGS2 basis, co-rotation, Cubature, and parallel
local-solve design. Its corrected stable Neo-Hookean CPU reference and WGSL
energy, stress, exact current tangent, material dispatch, and deformation-frame
construction pass the Phase 1 CPU/GPU oracle corpus. Nonlinear current-pose
Cubature training, held-out update validation, `f32` packing, and production
GPU update parity also pass on the private capability fixture. The four visible
demos still use the legacy co-rotated-linear path. Explicit in-app material
labels, globalization, convergence controls, forces/handles, and stable-
material demo gates remain before the project claims the paper's complete
nonlinear solid capability.

Contact here is an implicit quadratic ground penalty with simple viscous
tangential damping for grounded vertices. It is not a Coulomb friction model.
Full incremental potential contact would also require broad-phase collision
detection, continuous collision detection, barrier derivatives, Coulomb
friction, and a collision-safe line search. The six-body scene therefore keeps
bodies in separate lanes rather than implying unsupported body-body collision.

The checked-in demos favor transparent, inspectable preprocessing over scale.
For million-element scenes, use a sparse native/offline precompute and serialize
the four/six sample records into the same GPU ABI.
