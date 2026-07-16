# JGS2 WebGPU elastodynamics

A real-time browser implementation of the core algorithm in
[*JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU
Elastodynamics*](https://arxiv.org/abs/2506.06494) by Lan et al. (2025).

The app builds a tetrahedral implicit-Euler problem, precomputes the paper's
globally aware local perturbation bases and nonnegative Cubature weights, then
runs co-rotated vertex solves entirely in WebGPU compute shaders. The same GPU
position buffer is rendered directly with no per-frame readback.

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
npm run test:screenshot   # hardware-WebGPU deterministic browser tests
```

Software WebGPU adapters, including SwiftShader, are deliberately rejected.
The real-time and screenshot tests must exercise a hardware adapter.

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
`U_Ci = -H_CC^-1 H_Ci` (equations 19-23). Eight low-frequency modes of the
rest Hessian train a deterministic greedy NNLS Cubature set. The demos retain
up to four or six positive samples per vertex in fixed-width GPU slots and
discard the dense bases.

Each runtime Jacobi invocation owns one vertex and solves equation 15:

```text
(H_ii + Htilde_ii) delta_i = -(g_i + gtilde_i).
```

WGSL gathers exact incident-element terms, co-rotates each selected basis,
evaluates complementary gradient and Hessian-vector products, performs a
regularized `3 x 3` Cholesky solve, and writes to the opposite ping-pong region.
No floating-point atomics or materialized `12 x 12` runtime Hessians are needed.
Incident samples retain their neighbor/cross response while subtracting the
source block already included by the exact local gather.

The strongest CPU test is the exact quadratic oracle from equations 7-13: one
local equilibrium-basis solve must equal the corresponding block of the global
Newton update. Other tests check `S_i Ubar_i = I`, complementary equilibrium,
ordered low modes, nonnegative Cubature, mass conservation, surface topology,
and deterministic scene packing.

The four/six-sample cap is intentionally the paper's runtime budget, not a
claim that these tiny educational meshes reproduce its reported sub-1% raw
Cubature training residual. The implementation validates the more decisive
quantity too: local plus all remainder candidates matches the global Newton
block, while the selected K=4 minimal-scene basis has about 1.5% RMS update
error across the eight training modes.

## CPU, GPU, and WASM tradeoff

- CPU once per scene: mesh/topology construction, f64 rest FEM and lumped mass,
  one dense Cholesky for these small demos, low modes, three-right-hand-side
  equilibrium solves, NNLS Cubature, and GPU buffer packing.
- GPU every frame: implicit prediction, polar frames, co-rotated elasticity,
  Cubature projection, local solves, floor penalty, velocity update, and
  rendering.
- CPU in tests only: synchronized checkpoint readback for numeric invariants.

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
2. advances an exact number of implicit timesteps;
3. waits for submitted GPU work and canvas presentation;
4. renders and saves the ending frame;
5. reads positions once and checks finiteness, fixed vertices, positive
   tetrahedron determinants, bounds, landmarks, and visible pixel change.

The Playwright suite covers all four scenes and also verifies the visible
WebGPU-unavailable error path. Start/end PNGs are retained under `test-results`
for manual inspection.

## Scope

This repository implements the JGS2 basis, co-rotation, Cubature, and parallel
local-solve design for co-rotated linear tetrahedral elasticity. The paper's
largest examples use stable Neo-Hookean elasticity; this implementation does
not claim that material model.

Contact here is an implicit quadratic ground penalty. Full incremental
potential contact would also require broad-phase collision detection,
continuous collision detection, barrier derivatives, friction, and a
collision-safe line search. The six-body scene therefore keeps bodies in
separate lanes rather than implying unsupported body-body collision.

The checked-in demos favor transparent, inspectable preprocessing over scale.
For million-element scenes, use a sparse native/offline precompute and serialize
the four/six sample records into the same GPU ABI.
