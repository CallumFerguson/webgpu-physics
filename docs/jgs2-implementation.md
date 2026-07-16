# JGS2 WebGPU implementation notes

This project implements the central algorithm from *JGS2: Near Second-order
Converging Jacobi/Gauss-Seidel for GPU Elastodynamics* (Lan et al., 2025) for
small procedural tetrahedral scenes. It uses implicit Euler, isotropic linear
tetrahedral FEM in a co-rotated frame, a vertex-sized Jacobi subproblem, and the
paper's precomputed perturbation bases and Cubature projection.

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

At runtime, the basis is co-rotated using the current per-vertex frames. Each
GPU invocation owns one vertex, gathers its incident tetrahedra, projects the
selected remainder elements into the three-dimensional basis, and solves

```text
(H_ii + H_tilde_ii) delta_i = -(g_i + g_tilde_i).
```

The GPU ABI has four or six fixed sample slots per movable vertex; zero-weight
slots remain empty when NNLS retains fewer positive tetrahedra.
Deterministic low-frequency training deformations are normalized as in the
paper; a greedy search repeatedly solves the resulting small nonnegative least
squares problem. Only nonnegative weights and the selected four `3 x 3` basis
blocks are uploaded.

Every tetrahedron may be a remainder candidate. For an incident tetrahedron,
training and WGSL subtract its source gradient and diagonal Hessian block from
the full basis projection, because those terms were already gathered exactly;
neighbor and cross terms from that same tet remain. This makes the local term
plus all candidates reproduce the exact projected quadratic system.

The paper does not specify several reproducibility details, including the exact
local/complementary energy split, training-pose amplitudes, the rule used to
average a vertex deformation gradient, or line-search acceptance constants.
This implementation makes those choices explicit in source and checks them
against a dense CPU oracle on tiny systems.

## CPU, GPU, and WASM split

| Location | Work | Reason |
| --- | --- | --- |
| CPU, once per scene | Procedural mesh construction, rest tetrahedral data, lumped masses, dense rest-Hessian Cholesky, equilibrium bases, Cubature training, and buffer packing | These tasks are irregular, use double precision, and are amortized over the scene. This mirrors the paper's CPU preprocessing. |
| GPU, every frame | Implicit prediction, polar frames, co-rotated element gradients and Hessian products, Cubature projection, regularized `3 x 3` vertex solves, per-body horizontal COM correction, floor contact and tangential damping, velocity update, and surface rendering | These are uniform data-parallel kernels. State stays GPU-resident, with no per-frame position readback. |
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
- Tet polar decomposition is followed by explicit orthonormalization with a
  degenerate-matrix fallback. A vertex frame is the polar rotation of a
  volume-weighted average of its incident tet rotations; the paper does not
  provide its exact vertex-frame construction.
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
and parallel local solve. The material is co-rotated linear elasticity rather
than the stable Neo-Hookean model used in the paper's largest examples. Contact
is an implicit quadratic ground penalty with simple grounded viscous damping.
Full incremental potential contact would additionally require broad-phase
collision detection, continuous collision detection, barrier derivatives,
Coulomb friction, and a collision-safe line search; none of those are silently
approximated or claimed here.

The browser demos are correctness and real-time demonstrations, not a
reproduction of the paper's million-element benchmark. The dense educational
precomputation deliberately caps scene size. A production path would generate
sparse, precomputed binary assets offline and keep the same GPU runtime layout.
The paper reports less than 1% Cubature training residual for its large meshes;
these tiny scenes keep the same four/six-sample budget but use a relaxed fit.
The exact all-candidate oracle matches the global Newton block, and the selected
minimal K=4 solve is within roughly 1.5% RMS on its low-mode training set.
