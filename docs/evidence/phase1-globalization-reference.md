# Phase 1 CPU globalization reference evidence

This report records a **CPU material-and-inertia reference foundation** for
Phase 1 nonlinear globalization. It does not qualify any of P1-EC-05 through
P1-EC-10 and does not qualify Phase 1 exit. The restricted scalar does not yet
contain external-force or quadratic-target terms, and none of this milestone's
globalization, assembled-revert, or convergence logic is wired into the
production WebGPU iteration.

## Frozen scope and policy

`manifests/phase1-globalization.v1.json` binds this reference to the versioned
stable-material and nonlinear-Cubature fixtures. Its machine-readable scope is
exactly implicit-Euler inertia plus stable Neo-Hookean material. The manifest
also records `compositeForcesAndTargets = pending`, `gpuProduction = pending`,
and `qualifiesPhase1Exit = false`.

The reference freezes these policies:

- accepted poses require `J > 1e-4`;
- the solve-only eigenvalue floor is `256 * 2^-24` of the local scale;
- normalized diagonal shift is capped at `1e-3` and an over-cap solve fails;
- Armijo uses `c1 = 1e-4`, factor `0.5`, and at most 12 backtracks;
- geometry is checked before trial material energy;
- a failed local search returns a zero update;
- convergence requires both component-aware residual and normalized update
  thresholds, with runtime thresholds capped at `1e-3`.

The validator rejects unknown fields, source-fixture reference drift, policy drift,
incomplete or reordered case inventories, and a broadened objective-term list.
Runtime constants are separately bound to the manifest in a unit test.

## Local solve and line-search reference

The CPU reference computes the minimum eigenvalue of a symmetric 3-by-3 local
matrix in closed form, applies only the documented uniform solve shift, and
uses a scale-relative 3-by-3 Cholesky solve. Tests cover matrices scaled by
`1e-12`, `1`, and `1e12`, the exact shift-cap boundary, over-cap failure,
descent, symmetry diagnostics, and shifted linear residual.

The local Armijo helper receives separate geometry and energy callbacks. Tests
cover zero, one, and multiple backtracks; the exact determinant-floor boundary;
all 13 candidate alphas failing; non-descent input; and an infeasible trial
that demonstrably never invokes material energy. Multi-tetrahedron regressions
also prove that a positive-infinity determinant cannot be hidden behind a
finite neighbor: assembled acceptance fails closed, and all non-finite
restricted trials skip their energy callback.

## Restricted selected-Cubature scalar

For this partial objective, the scalar matches the derivatives assembled from:

- exact source inertia and incident stable Neo-Hookean elements; and
- weighted projected complementary inertia and material, with incident source
  terms subtracted to avoid double counting.

Current co-rotated basis blocks are frozen from each packed runtime record at
the source pose. Finite differences match the selected local gradient within
`2e-6` relative error and Hessian within `2e-4`. Mutation tests prove that the
frozen scalar and derivatives do not change if caller-owned positions,
contexts, packed basis arrays, or exposed diagnostic views are modified.
Zero-weight GPU ABI padding records—including the empty tetrahedron sentinel
and a hostile in-range duplicate—are ignored before geometry, uniqueness, or
basis validation.

The canonical packed corpus covers ten active sources over 24 training and
held-out poses, or 240 nonzero local systems. All returned a descending
direction, accepted a feasible Armijo step, and stayed below the shift cap:

| Diagnostic | Observed |
| --- | ---: |
| Local systems | 240 |
| Maximum normalized shift | `0.000000e+0` |
| Maximum shifted linear residual | `3.801004e-16` |
| Maximum backtracks | 0 |
| Minimum accepted determinant | `8.647680e-1` |

Synthetic fixtures separately exercise nonzero diagonal shifting and Armijo
backtracking because the canonical material/inertia corpus does not require
either.

## Assembled feasibility and convergence protocol

An adversarial one-tetrahedron Jacobi fixture applies two individually feasible
vertex updates whose assembled pose has `J = -0.2100000000000002`. The complete
pose is reverted byte-for-byte. Source, candidate, and accepted assembled
energies are recorded separately; a feasible candidate is accepted even when
its recorded energy increases, proving that this diagnostic is not an
assembled energy gate. Invalid determinant arithmetic uses finite-zero
sentinels with an explicit validity flag.

The convergence helper forms the component-aware residual from inertia,
material, external-force, target, and contact gradient components and divides
maximum vertex update by `max(sceneScale, 1e-12)`. It requires matching nonempty
xyz dimensions, finite overflow-safe derived diagnostics, a feasible
non-reverted assembled pose, and zero local failures. These are protocol unit
tests only: no tiny nonlinear solve history or GPU convergence reduction is
claimed.

## Commands and result

```powershell
.\node_modules\.bin\vitest.cmd run src/simulation/cpu/nonlinear-globalization.test.ts src/simulation/cpu/nonlinear-restricted-energy.test.ts src/simulation/cpu/nonlinear-iteration.test.ts src/reproducibility/phase1-globalization-manifest.test.ts --disableConsoleIntercept --reporter=verbose
```

The focused gate passed 29/29 tests in four files. The exact-tree milestone
gate then passed 189/189 unit tests, the production build, frozen selector
verification at 26/189 unit and 7/18 E2E, and all 18 hardware-WebGPU E2E tests
in 46,802.453 ms with zero skips, retries, or failures. `test:e2e:full` was not
run because this is a non-exit CPU reference milestone, not release, formal
performance, or Phase 1 qualification evidence.

## Remaining before any related Phase 1 exit criterion closes

- add external-force and quadratic-target terms to the restricted scalar and
  its projected gradient/Hessian;
- add deterministic tiny nonlinear reference solves with recorded convergence
  histories reaching `r <= 1e-5`;
- implement the same shift, feasibility-first line search, diagnostics,
  assembled full-pose revert, and convergence reductions on WebGPU;
- prove CPU/GPU parity over canonical dynamic checkpoints;
- add public stable-material scenes, scripted and pointer handles, semantic
  screenshots, clean target release, explicit public co-rotated regression
  labels, and Phase 1 performance evidence.

Until those items pass their own gates, P1-EC-05 through P1-EC-10 and the Phase
1 completion status remain open.
