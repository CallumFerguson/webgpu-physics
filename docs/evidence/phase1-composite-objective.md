# Phase 1 composite-objective evidence

This milestone adds arbitrary per-vertex linear external-force potentials and
isotropic per-vertex quadratic targets to the Float64 reference and stable
Neo-Hookean production WebGPU solve. It is partial Phase 1 evidence, not Phase
1 exit evidence. Public scripted and pointer-driven manipulation, tiny/public
convergence histories, GPU-driven early termination, public stable scenes, and
the Phase 1 performance workload remain open.

The executable contract is frozen in
[`manifests/phase1-globalization.v3.json`](../../manifests/phase1-globalization.v3.json).
Versions 1 and 2 remain the unmodified historical CPU-reference and
material/inertia/floor GPU contracts.

## Complete objective and paper equivalence

For vertex `i`, the two new potentials are

```text
E_force_i(x_i)  = -f_i dot x_i
g_force_i       = -f_i
H_force_i       = 0

E_target_i(x_i) = 1/2 k_i ||x_i - t_i||^2
g_target_i      = k_i (x_i - t_i)
H_target_i      = k_i I,             k_i >= 0.
```

The stable runtime keeps gravity in its inertial predictor

```text
y_i = x_old_i + h v_damped_i + h^2 g
```

and adds the user force as the explicit potential `-f_i dot x_i`. This is
derivatively identical to putting the same force in the paper's Eq. 1
predictor `z_i = y_i + h^2 M_i^-1 f_i`:

```text
1/(2 h^2) ||x_i - z_i||^2_M
  = 1/(2 h^2) ||x_i - y_i||^2_M - f_i dot x_i + constant.
```

The position-independent constant cannot affect a gradient, Hessian, local
direction, Armijo energy difference, or converged pose. Keeping the user force
explicit also preserves an independently observable external-force component
in the convergence reduction.

The CPU restricted scalar evaluates the objective as a direct displacement
delta rather than subtracting two large origin-dependent absolute energies:

```text
Delta E_force  = -f dot d
Delta E_target = k (x-t) dot d + 1/2 k ||d||^2.
```

A regression translates the complete pose, predictor, and targets by world
offsets up to `2e10`, then proves this direct delta matches the quadratic form
`g dot d + 1/2 d^T H d` within relative error `1e-8` and still produces an
accepted globalized step. This prevents catastrophic cancellation from making
the explicit-force form dependent on the chosen world origin.

The shared stable restricted scalar now contains exact source inertia, linear
force, quadratic target, and incident stable Neo-Hookean terms, plus the
selected complementary Cubature terms. The production GPU scalar additionally
contains its analytic floor penalty. In the paper's Eqs. 15-17 projection,
each candidate tetrahedron receives one
incident-count share of every vertex's inertia, force, and target derivative.
The source share is subtracted for incident samples because that source term
was already gathered exactly. With unit weights, the source plus all
candidates therefore reconstructs the full projected objective without double
counting.

The hardware oracle separately checks every active vertex against the analytic
identities `g_ext = -f` and `g_target = k(x-t)`. It evaluates the same solved
pose once with objectives enabled and once with objectives disabled, and copies
the production predicted-position state into the baseline before evaluation.
Material, inertia, and contact terms therefore cancel, so the total-gradient
difference must satisfy

```text
g_oracle - g_baseline = -f + k (x-target).
```

Subtracting the two local Hessians must likewise produce exactly `k I`
(diagonal `k`, zero off-diagonal). These objective-on/off comparisons isolate
the full objective gradient and target curvature instead of inferring either
from final motion. The measured maximum total-gradient relative error is
`1.295315e-7`.

A second behavioral probe uses one production solver on the sparse canonical
beam and applies force only at driver vertex 11. At the driver, selected
incident samples have source-share leverage `1.163421`—large enough that a
missing source subtraction would materially double-count the exact local
force. The objective changes the driver direction by `1.067575e-1`, while GPU
error divided by that effect is `9.522279e-8`. At projection vertex 1, which
has no objective of its own, selected complementary samples include a
tetrahedron containing the driver. Its nonzero `1.754525e-4` direction change
therefore exercises neighbor-only Cubature projection; GPU error divided by
that effect is `5.024959e-8`. Both locals are accepted in the same solver run.

## CPU, GPU, and WASM tradeoff

The CPU retains Float64 validation, derivative oracles, deterministic
preprocessing/training, and explicit checkpoint comparisons. Those operations
are irregular, small for the checked-in fixtures, and benefit from double
precision. The per-frame objective evaluation, projected derivatives,
line-search trials, assembled feasibility decision, convergence reduction,
state update, and rendering stay on WebGPU. Normal stepping performs no
diagnostic readback.

WebAssembly is not used. It would not accelerate the GPU-resident frame loop
and would add another memory representation and transfer boundary. It should
be reconsidered only if arbitrary large meshes must be sparsely precomputed in
the browser with a mature threaded SIMD factorization/NNLS implementation.
Fixed large scenes should instead load offline-generated artifacts into the
same GPU ABI.

## Frozen GPU ABI and update semantics

Each stable-solver vertex owns one 32-byte read-only shader record at storage
binding 6:

```text
vec4(externalForce.xyz, 0)
vec4(targetPosition.xyz, isotropicStiffness)
```

The uniform buffer moves to binding 7 and remains 176 bytes. Together with the
six existing storage buffers, the objective buffer uses seven storage-buffer
bindings, below WebGPU's portable minimum of eight. Legacy regression solves
bind exactly one inactive 32-byte dummy record, regardless of vertex count, to
satisfy the shared pipeline layout. They therefore do not allocate the stable
path's `32 * vertexCount` objective array, the shader never reads the record on
the legacy branch, and all active objective creation/update APIs are rejected.

Creation accepts optional packed force and target arrays. Runtime exposes
single-vertex and bulk force writes, clearing all forces, single-vertex and
bulk quadratic-target writes, and single/all-target release. Every write is
validated completely before it is enqueued and mirrored in detached CPU state;
there is no GPU readback. Non-finite inputs, invalid lengths or indices,
negative stiffness, active objectives on a legacy solver, and nonzero
force/target conflicts on immutable pinned vertices fail closed.

`GPUQueue.writeBuffer` and command submission share one queue. A write before
`step` or `stepFrames` is therefore visible to that submission; a write after
submission is ordered after it. `stepFrames(n)` synchronously encodes all `n`
frames before returning, so one objective revision and activity mask is
constant across the batch. Callers that need a moving target must issue a
write followed by a separate frame or batch.

A target affects motion only through its quadratic objective and the same
determinant-feasible Armijo/assembled gates as every other term. It is never
snapped into canonical position storage. Releasing a target writes zero
stiffness while retaining its inert target coordinates; this removes its
energy, gradient, and Hessian exactly and does not change position or velocity.
The immutable pinned-rest proposal remains a separate hard constraint inside
the assembled-feasibility candidate.

## Acceptance criteria

The milestone is eligible for its commit only when all of these observations
hold on the exact tree:

- the explicit-force and shifted-predictor formulations agree within `1e-12`;
- complete-objective restricted gradients and Hessians meet relative errors
  `2e-6` and `2e-4`;
- all 240 canonical composite local systems are evaluated and selected-update
  RMS is at most `0.02`;
- routine/full GPU composite corpora cover exactly 60/240 active local systems
  and remain within `1e-3` of the independent CPU reference;
- every corpus local system has an objective-induced direction effect of at
  least `1e-4`, and GPU error relative to that effect is at most `1e-3`;
- accepted complete-objective steps satisfy determinant-first Armijo,
  `gradient dot direction < 0`, and normalized shift at most `1e-3`;
- convergence exposes nonzero force and target components and requires both
  residual and update criteria;
- malformed mutations and pinned conflicts produce no buffer write or
  objective-revision change;
- objective mutation/release itself adds no diagnostic readback; release
  produces a zero target component, no snap, and no velocity reset; and
- the focused unit/build gates and all three individually selectable hardware
  cases pass without skips or retries.

The representative CPU corpus currently covers 240 force-plus-target local
systems and records selected-versus-full-projection update RMS
`1.260195e-3` (`0.1260195%`), below the frozen `0.02` gate. The authoritative
source is
[`src/simulation/cpu/nonlinear-restricted-energy.test.ts`](../../src/simulation/cpu/nonlinear-restricted-energy.test.ts).

## Hardware-case provenance

All three selectors live in
[`tests/e2e/nonlinear-objectives-gpu.spec.ts`](../../tests/e2e/nonlinear-objectives-gpu.spec.ts)
and share one hardware page/device fixture. The table describes their required
observations. Focused routine/full and repository-wide milestone results are
recorded below.

| Selector | Required observation |
| --- | --- |
| `P1-COMPOSITE-GPU: force and quadratic-target derivatives match the independent CPU reference` | Production directions, accepted alpha/energy delta, and exact force/target component norms agree with the detached Float64 reference within `1e-3`; after copying the predicted state into an objective-off baseline, the exact oracle checks per-vertex `(g_oracle-g_baseline)=-f+k(x-target)` and local-Hessian difference `kI`; a one-solver sparse-beam probe separately exercises driver source-share subtraction and neighbor-only complementary projection; stepping has no diagnostic readback before the explicit checkpoint. |
| `P1-COMPOSITE-GPU: complete-objective Armijo, shift, and convergence policies hold` | Routine/full runs cover exactly 60/240 active local systems across force-only, target-only, and combined modes; each local result is compared with the independent CPU objective, the objective changes the CPU direction by at least `1e-4`, effect-relative GPU error is at most `1e-3`, accepted updates descend and satisfy Armijo and `J>1e-4`, normalized shift stays at most `1e-3`, and active convergence components are nonzero. |
| `P1-COMPOSITE-GPU: mutable objective inputs fail closed and release without hidden state` | Invalid and pinned mutations leave revision/activity state unchanged; ordered sparse writes expose the latest target; one `stepFrames` batch keeps one objective snapshot; release itself adds no readback, preserves position/velocity, zeros the target component, ignores stale zero-stiffness coordinates, and rejoins the no-target trajectory. |

## Recorded verification

Do not replace a pending cell with a result from a different tree or a partial
command.

| Command | Exact-tree result |
| --- | --- |
| `.\node_modules\.bin\vitest.cmd run src\simulation\cpu\nonlinear-objective.test.ts src\simulation\cpu\nonlinear-restricted-energy.test.ts --disableConsoleIntercept --reporter=verbose` | 15/15 passed in two files (4.29 s runner duration). The log records 240 systems and selected-update RMS `1.260195e-3`; direct objective deltas also remain accurate after a `2e10` world translation. |
| `.\node_modules\.bin\vitest.cmd run src\reproducibility\phase1-globalization-manifest.test.ts --disableConsoleIntercept --reporter=verbose` | 9/9 passed, including strict v1/v2/v3 parsing, provenance, and mutation rejection (357 ms runner duration). |
| `npm.cmd run build` | Strict TypeScript and the production Vite bundle passed. |
| `npm.cmd run test:e2e -- tests/e2e/nonlinear-objectives-gpu.spec.ts` | 3/3 passed in 4.4 s Playwright time (5.6 s command wall), covering exactly 60 active local systems. Minimum objective direction effect `5.841233e-3`; maximum direction/alpha/accepted-energy-delta relative errors `1.478227e-8` / `0` / `2.115330e-6`; maximum effect-relative direction error `1.587226e-6`. The root selector additionally records source leverage `1.163421`, driver effect/error-to-effect `1.067575e-1` / `9.522279e-8`, projection-only effect/error-to-effect `1.754525e-4` / `5.024959e-8`, and total-gradient error `1.295315e-7`. |
| `npm.cmd run test:e2e:full -- tests/e2e/nonlinear-objectives-gpu.spec.ts` | 3/3 passed in 5.8 s Playwright time (7.0 s command wall), including the sparse proof and exactly 240 active local systems. Minimum objective direction effect `4.052469e-3`; maximum direction/alpha/accepted-energy-delta relative errors `2.228256e-8` / `0` / `2.493165e-6`; maximum effect-relative direction error `2.454325e-6`. |
| `npm.cmd run test:unit` | 282/282 passed across 38 files in 5.59 s runner time. |
| `npm.cmd run test:baseline-manifest` | Frozen selectors resolved at 26/282 unit and 7/27 E2E. |
| `npm.cmd run test:e2e` | 27/27 passed in 57.3 s Playwright time (58.5 s command wall), with zero skips or retries; the complete routine tier remains below one minute. |
| `npm.cmd run test:e2e:full` | 27/27 passed in 5.0 min Playwright time (299 s command wall), including the complete 38,400-frame force-free corpus and all 240 composite-objective active local systems. |

Every acceptance item above is satisfied for this composite-objective
milestone, and both routine and full aggregate gates pass on the exact tree.
This completes the milestone evidence only; it does not close the still-open
Phase 1 scene, convergence-history, early-exit, or performance criteria.

## Remaining work before Phase 1 exit

- Add explicitly labeled public stable scenes, including scripted and
  pointer-driven moving targets with semantic start/peak/release screenshots.
- Record canonical dynamic determinant checkpoints and tiny/public convergence
  histories, then implement GPU-driven early termination while retaining exact
  iteration controls for tests.
- Freeze and qualify the 4,800-tetrahedron Phase 1 performance workload.
- Complete the cantilever, handle-tracking/release, screenshot, performance,
  and reviewer exit criteria. This milestone alone does not claim them.
