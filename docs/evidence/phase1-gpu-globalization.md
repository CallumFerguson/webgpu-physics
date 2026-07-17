# Phase 1 WebGPU globalization evidence

This milestone moves the frozen material-and-inertia globalization policy from
the Float64 reference into the production stable-Neo-Hookean WebGPU path. It is
partial Phase 1 evidence, not Phase 1 exit evidence: arbitrary external forces,
quadratic kinematic targets, GPU-driven early termination, public stable
scenes, and the Phase 1 performance workload remain open.

The executable contract is frozen in
[`manifests/phase1-globalization.v2.json`](../../manifests/phase1-globalization.v2.json).
Version 1 remains the historical CPU-reference contract.

## Implemented runtime policy

For homogeneous stable-Neo-Hookean inputs, every nonlinear iteration stays on
the GPU and executes this dependency chain:

```text
exact-f32 source-feasibility preflight
  -> tetrahedron frames -> vertex frames -> local solve and Armijo trials
  -> candidate tetrahedron diagnostics -> assembled feasibility reduction
  -> accept or byte-exact whole-pose revert -> exact gradient components
  -> convergence reduction and history record
```

The local solve uses the same scale-aware symmetric `3 x 3` eigensolve,
solve-only diagonal shift, normalized shift cap, and unclamped normalized
Cholesky policy as the CPU reference. It records the raw minimum eigenvalue,
scale, shift, normalized shift, shifted residual, direction, descent product,
accepted step length, backtracks, determinant, energy delta, and status.

Armijo tests use `c1 = 1e-4`, candidates `alpha = 1, 1/2, ..., 2^-12`, and a
strict accepted determinant floor `J > 1e-4`. Geometry is evaluated before the
restricted energy. The current scalar contains exact source inertia, the
implicit floor penalty, incident stable material, weighted complementary
material and distributed inertia, and the matching source-term subtraction.
External-force and quadratic-target slots remain exactly zero because those
capabilities are not implemented yet.

After parallel local acceptance, a separate assembled pass evaluates every
tetrahedron at the candidate pose. An infeasible or non-finite candidate is
rejected and every position `vec4` is copied from the source pose; partial
vertex acceptance is never retained. Assembled material-plus-inertia/floor
energy is diagnostic only and does not accept an individual local step.
Pinned rest targets are proposed inside that same candidate. Stable
finalization keeps the accepted candidate or byte-identical reverted source;
it cannot reapply a rejected pinned displacement after the determinant gate.
This is an immutable hard rest constraint, not the still-pending arbitrary
quadratic or moving kinematic-target objective represented by the zero target
component.

The convergence pass records the exact inertia, material, external-force,
target, and contact component slots, total relative residual, maximum update,
normalized update, feasibility, local failures, and the final convergence
flag. External-force and target components are currently valid zeros. Both
residual and update thresholds are required and runtime thresholds are capped
at `1e-3`. Exact-iteration APIs still execute the requested iteration count;
the flag does not yet terminate the GPU command sequence early.

The legacy co-rotated-linear path retains its prior regularized/clamped solve
and pass graph. Mixed stable and co-rotated inputs are rejected rather than
silently applying one material's globalization semantics to the other.

## GPU residency and performance tradeoff

Solver creation performs one synchronized GPU readback to certify the uploaded
stable source pose with the exact production f32 determinant arithmetic. Normal
stepping performs no diagnostic readback. Local records, the assembled
decision, and up to 64 iteration-history records remain in storage buffers;
tests read them only through an explicit synchronized checkpoint.

Candidate and convergence reductions use one deterministic 128-lane workgroup
with fixed strided record ownership and fixed binary trees. Convergence uses
two vertex scans: one for overflow-resistant scales and one for normalized
sums. Candidate and convergence scratch consume 5,120 and 4,096 bytes,
respectively, remaining within WebGPU's portable 16 KiB workgroup-storage
floor. This removes the former single-invocation bottleneck for the Phase 1
workloads without adding atomics or subgroup assumptions. Formal Phase 1
performance qualification is still required; a multi-workgroup hierarchy may
be warranted for later, materially larger scenes.

## Hardware cases

The production shader and reduction pipelines are exercised by six
independently selectable Playwright cases in
[`tests/e2e/nonlinear-globalization-gpu.spec.ts`](../../tests/e2e/nonlinear-globalization-gpu.spec.ts).
They share one browser page, hardware adapter, device, and frozen CPU fixture to
avoid repeating setup; each case keeps its own error scope and assertions.

| Case | Required observation |
| --- | --- |
| Assembled feasibility | Two individually feasible, higher-energy shears each have `J = 1` and are accepted even though assembled energy is diagnostic-only; their Jacobi combination has `J = -0.21` and reverts byte-exactly. A finite-input arithmetic-overflow candidate is also classified nonfinite and reverts while preserving the valid source minimum. A coherent rigidly rotated `J = 1` source whose newly pinned B rest target alone would produce `J = -1` likewise reverts byte-exactly through finalization, proving the target cannot bypass the last gate. |
| Initial-source preflight | A pose that passes the host Float64-over-f32 determinant helper but falls below the floor under production GPU f32 arithmetic is rejected during solver creation. |
| Feasibility-first Armijo | A one-tetrahedron production solve backtracks to `0 < alpha < 1`; at least one infeasible geometric trial skips material-energy evaluation; the accepted trial satisfies Armijo and `J > 1e-4`. |
| Shared shift solver | GPU and CPU classifications and diagnostics agree for an indefinite accepted system, values immediately inside/outside the `1e-3` shift cap, and scales `1e-12`, `1`, and `1e12`. |
| Floor-active objective | A penetrating stable pose has nonzero floor energy and contact-gradient norm; accepted local steps satisfy Armijo and the assembled energy/residual diagnostics match the exact GPU oracle. |
| Convergence reduction | GPU and CPU agree for exact component cancellation, excessive residual, excessive update, assembled revert, and local failure; only the first case converges. Repeated prefix sizes `1`, `127`, `128`, `129`, and `257` exercise both sides of the workgroup boundary with identical repeated results. |

The nonlinear Cubature production oracle additionally runs the globalized
stable path on six routine sentinels and, with `JGS2_FULL_E2E=1`, all 24 frozen
training and validation poses. Routine sentinels retain two nonlinear
iterations; full qualification keeps two-iteration depth at the first and last
sentinels and runs the remaining poses for one iteration. The exhaustive tier
covers exactly 240 active packed local systems. It compares the production
update and GPU-resident residual, maximum update, accepted determinant, and
energy diagnostics with independent CPU oracles.

## Recorded verification

Nominated hardware: Intel Core i7-13700K, NVIDIA RTX 5090, hardware Chrome
WebGPU.

| Command | Result |
| --- | --- |
| `npm.cmd run test:e2e -- tests/e2e/nonlinear-globalization-gpu.spec.ts` | 6/6 passed in 3.2 s (4.2 s command wall). |
| `npm.cmd run test:e2e -- tests/e2e/nonlinear-cubature-gpu.spec.ts` | The selector passed in 4.4 s (5.4 s command wall), covering six two-iteration sentinels; maximum production-update relative error was `1.380e-6`. |
| `npm.cmd run test:e2e:full -- tests/e2e/nonlinear-cubature-gpu.spec.ts` | The selector passed in 5.5 s (6.6 s command wall), covering 24/24 unique frozen poses and exactly 240 active local systems; the first and last poses retained two-iteration depth and maximum production-update relative error was `9.992e-7`. |
| `npm.cmd run test:unit` | 248/248 passed across 37 files. |
| `npm.cmd run build` | Strict TypeScript and the production Vite bundle passed. |
| `npm.cmd run test:baseline-manifest` | Frozen selectors verified at 26/248 unit and 7/24 E2E. |
| `npm.cmd run test:e2e` | 24/24 passed in 50.6 s (51.6 s command wall), with zero skips or retries; the routine tier remains below one minute. |
| `npm.cmd run test:e2e:full` | 24/24 passed in 4.0 min (239.2 s command wall), including 38,400 force-free corpus frames and formal 120-warm-up/600-measured profiles. |

The complete unit/build/manifest counts and independent review outcome are
recorded in the roadmap gate row after the exact-current-tree gate is run.

## Remaining work before Phase 1 exit

- Add arbitrary external-force and quadratic-target contributions to the
  local derivatives, restricted energy, and exact convergence components.
- Add GPU-driven early termination while preserving exact-iteration test
  controls.
- Add explicitly labeled public stable scenes, scripted/pointer targets, their
  semantic screenshot gates, and the frozen Phase 1 performance workload.
- Qualify the parallel reductions and complete stable solve on that frozen
  performance workload; introduce a multi-workgroup hierarchy only if the
  recorded workload demonstrates it is needed.
- Record tiny-reference and public-runtime convergence histories and complete
  P1-EC-06 through P1-EC-10 only against the full composite objective.
