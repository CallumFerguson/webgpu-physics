# JGS2 WebGPU implementation notes

This project implements the central algorithm from *JGS2: Near Second-order
Converging Jacobi/Gauss-Seidel for GPU Elastodynamics* (Lan et al., 2025) for
small procedural tetrahedral scenes. It uses implicit Euler, a vertex-sized
Jacobi subproblem, and the paper's precomputed perturbation bases and Cubature
projection. The runtime can evaluate either the original co-rotated linear
regression material or the paper's stable Neo-Hookean energy and exact current
tangent. The stable path now trains and validates nonlinear current-pose
Cubature. A CPU material-and-inertia globalization reference now exists;
composite force/target terms and the production WebGPU globalization path
remain Phase 1 work.

## Algorithm mapping

For a vertex subproblem `i`, the rest timestep Hessian is partitioned into the
three local coordinates and all complementary coordinates. The CPU
precomputation forms the exact equilibrium extension

```text
U_Ci = -H_CC^-1 H_Ci
U_i  = [I; U_Ci].
```

The implementation uses the paper's full-coordinate equivalent: factor the
rest Hessian once, solve `Y_i = H^-1 S_i^T`, and normalize with
`U_i = Y_i (S_i Y_i)^-1`. Unit tests verify `S_i U_i = I`, the complementary
equilibrium residual, and the exact quadratic Newton-block identity.

At runtime, the basis is co-rotated using current per-vertex deformation
frames. Each GPU invocation owns one vertex, gathers its incident tetrahedra,
evaluates the selected material's current energy derivatives, projects the
selected remainder elements into the three-dimensional basis, and solves

```text
(H_ii + H_tilde_ii) delta_i = -(g_i + g_tilde_i).
```

The GPU ABI has four or six fixed sample slots per movable vertex; zero-weight
slots remain empty when NNLS retains fewer positive tetrahedra. The legacy
co-rotated-linear path trains candidates from the constant quadratic rest
system. The stable path instead evaluates every candidate from the current
stable Neo-Hookean gradient and tangent at deterministic low-mode nonlinear
poses while applying `R_v Ubar_vi R_i^T`. A greedy search repeatedly solves
the stacked nonnegative least-squares problem from Eq. 18. Only nonnegative
weights and the selected rest-basis blocks are uploaded; current rotations are
computed on the GPU.

Every tetrahedron may be a remainder candidate. For an incident tetrahedron,
training and WGSL subtract its source gradient and diagonal Hessian block from
the full basis projection, because those terms were already gathered exactly;
neighbor and cross terms from that same tet remain. This makes the local term
plus all candidates reproduce the exact projected system for either material.
The nonlinear CPU oracle verifies the stable path against independently
assembled `B^T g` and `B^T H B`, not a linearized `-displacement` identity.

The stable training corpus uses both signs of up to eight rest-Hessian modes,
eight held-out modal mixtures with different amplitudes and inertial targets,
component-wise rigid-mode removal for free bodies, and rigidly rotated held-out
cases when the whole mesh is unconstrained. Poses are
deformation-gradient normalized and must keep `min J >= 0.5`. Runtime-packed
`f32` data gates both the `<=1%` training residual and `<=2%` selected-versus-
exact update RMS. The production WebGPU parity test then starts from selected
training and validation shapes and compares one real solver iteration against
the packed CPU reference.

The paper does not specify several reproducibility details, including the exact
local/complementary energy split, training-pose amplitudes, the rule used to
average a vertex deformation gradient, or line-search acceptance constants.
This implementation makes those choices explicit in source and checks them
against a dense CPU oracle on tiny systems.

## CPU, GPU, and WASM split

| Location | Work | Reason |
| --- | --- | --- |
| CPU, once per scene | Procedural mesh construction, rest tetrahedral data, lumped masses, dense rest-Hessian Cholesky, equilibrium bases, nonlinear pose generation, Cubature training, and buffer packing | These tasks are irregular, use double precision, and are amortized over the scene. This mirrors the paper's CPU preprocessing. |
| GPU, every frame | Implicit prediction, polar frames, co-rotated linear or stable Neo-Hookean current gradients and tangent products, Cubature projection, regularized `3 x 3` vertex solves, per-body horizontal COM correction, floor contact and tangential damping, velocity update, and surface rendering | These are uniform data-parallel kernels. State stays GPU-resident, with no per-frame position readback. |
| CPU, tests only | Requested checkpoint readback and invariant calculation | A deliberate synchronization is useful for correctness but would distort real-time performance. |

WebAssembly is intentionally absent. It cannot improve the GPU-resident frame
loop and would add CPU/GPU synchronization and another memory representation.
It becomes worthwhile only if large arbitrary user meshes must be precomputed
inside the browser, where a threaded SIMD sparse factorization/eigensolver and
NNLS library could replace the current small-scene TypeScript preprocessing.
That path also requires a larger download and cross-origin isolation for WASM
threads. Fixed demo assets would be better precomputed offline.

## Numerical choices

- Positions and GPU calculations use `f32`; preprocessing uses JavaScript
  numbers and `Float64Array`.
- The rest Hessian includes the timestep inertia term, so a basis is specific
  to its mesh, mass, material, timestep, and fixed boundary conditions.
- A vertex deformation gradient is the incident-rest-volume-weighted average
  of current tetrahedron deformation gradients. Its frame is computed with
  seven fixed `f32` polar iterations followed by explicit orthonormalization
  and a degenerate-matrix fallback. This construction is affine-exact at both
  interior and boundary vertices and is shared with the Float64 CPU oracle.
- Local matrices are symmetrized and solved by shifted Cholesky. A scale-aware
  diagonal shift and maximum update length keep malformed `f32` systems finite.
- Vertex subproblems gather incident elements through CSR. This avoids relying
  on floating-point atomics, which are not part of core WebGPU.
- A free body's solved mass-weighted x/z center is translated to its predicted
  center before velocity finalization. This projects out net translation
  injected by finite independent local solves while preserving legitimate
  predicted motion. Bodies containing fixed vertices are excluded. The current
  `O(bodyCount * vertexCount)` pass is intended for these demo-scale scenes.
- Vertices within the floor contact margin receive viscous tangential velocity
  retention `exp(-12 * dt)`. This is timestep-independent ground drag, not a
  Coulomb static/kinetic friction law.
- The precomputed basis and weights are timestep-specific; the public solver
  rejects a runtime timestep that is not f32-equivalent to the training value.
- The solver uses an odd number of ping-pong Jacobi iterations so the final
  positions always return to a stable buffer region used directly by rendering.
- The runtime animation does not synchronously block on the queue. Completion
  promises cap work at two submitted batches so a slow GPU cannot accumulate
  unbounded latency. Deterministic tests wait after an exact number of steps.

## Scope and limitations

The implementation exercises the paper's JGS2 basis, co-rotation, Cubature,
and parallel local solve. The Phase 1 material path now contains the corrected
stable Neo-Hookean energy, stress, and exact current tangent on both CPU and
GPU, plus CPU/GPU checks over a frozen 64-pose material corpus. Its nonlinear
current-pose Cubature preprocessing, held-out local-update oracle, and
two-iteration production GPU parity path are also complete. The CPU reference
now additionally covers solve-only shifting, geometry-before-energy Armijo
trials, whole-pose feasibility/revert, and convergence normalization for its
material-and-inertia scalar. External-force and quadratic-target terms, WebGPU
globalization and reductions, and runtime convergence history remain pending.
The four public scenes continue to use the legacy co-rotated-linear material
until explicit labels, production local line search, convergence diagnostics,
and Phase 1 scenes meet their roadmap gates.
Contact is an implicit quadratic ground penalty with
simple grounded viscous damping.
Full incremental potential contact would additionally require broad-phase
collision detection, continuous collision detection, barrier derivatives,
Coulomb friction, and a collision-safe line search; none of those are silently
approximated or claimed here.

The browser demos are correctness and real-time demonstrations, not a
reproduction of the paper's million-element benchmark. The dense educational
precomputation deliberately caps scene size. A production path would generate
sparse, precomputed binary assets offline and keep the same GPU runtime layout.
The paper reports less than 1% Cubature training residual for its large meshes.
The stable capability fixture enforces the same threshold after `f32` packing
and also enforces a 2% held-out update-RMS threshold. It selects six of 12
full-rank candidates, but does not establish large-mesh preprocessing
scalability. The legacy
public scenes retain their earlier regression thresholds until they are
replaced by explicitly labeled stable scenes.
