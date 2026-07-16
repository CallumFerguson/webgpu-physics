# Phase 1 nonlinear-solid design record

This record freezes the Phase 1 material, feasibility, local-solve, convergence,
performance, and CPU/GPU/WASM decisions. The exact CPU and WGSL material path,
material dispatch, deformation frames, and nonlinear Cubature preprocessing
are implemented. Globalization remains an implementation contract.

## Primary sources

- JGS2 uses stable Neo-Hookean elasticity for all volumetric deformable
  experiments and evaluates current material gradients and Hessians in its
  local/Cubature systems: [Lan et al. 2025](https://arxiv.org/abs/2506.06494).
- The material is the corrected model in equation 14, with the Hooke-law
  reparameterization in equations 15-16 and PK1 in equation 18:
  [Smith, de Goes, and Kim 2018](https://graphics.pixar.com/library/StableElasticity/paper.pdf).
- Degenerate-state analysis is in the authors'
  [stable Neo-Hookean supplement](https://graphics.pixar.com/library/StableElasticity/stable_neo_hookean_supplement.pdf).

The Pixar publication page notes that the original printed expression for
`alpha` contained a typo. This implementation uses the corrected expression.

## Material equations

For deformation gradient `F`, define

```text
I1 = F : F
J  = det(F)
G  = cofactor(F) = dJ/dF
```

Physical Young's modulus `E` and Poisson ratio `nu` first produce the usual
Hooke-law Lamé parameters:

```text
mu0     = E / (2 (1 + nu))
lambda0 = E nu / ((1 + nu) (1 - 2 nu))
```

The stable model must use the paper's reparameterized values:

```text
mu     = 4 mu0 / 3
lambda = lambda0 + 5 mu0 / 6
alpha  = 1 + 3 mu / (4 lambda)
```

The implementation evaluates the rest-normalized expanded form

```text
psi(F) = mu/2 (I1 - 3)
       - 3 mu/4 (J - 1)
       + lambda/2 (J - 1)^2
       - mu/2 log((I1 + 1) / 4).
```

It is algebraically equivalent, up to a constant, to the paper's corrected
shifted-volume form. It has `psi(I) = 0` and avoids a large irrelevant constant
in f32 WGSL. Its first Piola stress is

```text
a = mu (1 - 1 / (I1 + 1))
b = lambda (J - 1) - 3 mu / 4
P = a F + b G.
```

For a matrix direction `D`, the exact tangent action is

```text
dP[D] = a D
       + 2 mu (F : D) / (I1 + 1)^2 F
       + lambda (G : D) G
       + b dG[D].
```

The cofactor and its directional derivative are evaluated as polynomials, with
no `inverse(F)` and no division by `J`. The exact CPU/GPU oracle Hessian is not
projected or regularized; solve-only positive-definite treatment is recorded
separately so derivative parity tests cannot accidentally validate a modified
material.

## Inversion and accepted-step policy

The raw Smith density, stress, and tangent deliberately remain finite through
`J = 0` and negative `J`. The CPU oracle retains that behavior and tests it.

Production simulation has a stricter feasibility contract:

- the accepted-pose determinant floor is `J_min = 1e-4`;
- trial steps at or below that floor are rejected before their material energy
  is used for acceptance;
- an assembled Jacobi iteration that violates the floor is reverted in full to
  its feasible source pose;
- world-space update clamping is not an inversion policy;
- accepted dynamic checkpoints must report `min J >= J_min`.

The CPU helper `assertStableNeoHookeanFeasible` is the reference form of this
guard. The GPU line search and iteration-revert path remain Phase 1 runtime
work.

## Vertex deformation frames

For vertex `i`, the paper's co-rotated basis uses the polar rotation of a local
deformation gradient, not an average of already-extracted tetrahedron
rotations. With rest volume `V_t` and shape gradients `grad N_tq`, define

```text
W_i   = sum_(t incident i) V_t
F_i   = sum_(t incident i) (V_t / W_i) F_t
c_iq  = sum_(t incident i, q in t) (V_t / W_i) grad N_tq
F_i   = sum_q x_q c_iq^T
R_i   = polar(F_i).
```

This sparse stencil satisfies `sum_q c_iq = 0` and
`sum_q X_q c_iq^T = I`, so it reproduces every affine field exactly at corner,
edge, face, and interior vertices. A deterministic non-affine test also proves
that this construction is distinguishable from the old average-rotation
shortcut.

The GPU computes the same weighted average directly from current tetrahedron
deformation gradients, divides by total incident rest volume, then uses seven
fixed `f32` polar iterations. The frozen non-affine fixture is scaled to `0.01`
of its nominal dimensions to enforce scale invariance; hardware matches the
Float64 CPU frames with maximum relative error `8.702e-8`. Its production
stable-kernel diagnostic uses rest-equivalent linear basis preprocessing before
switching the packed material tag to stable; that runtime solve remains finite
with minimum `J = 0.9394`. Both observations use explicit test-only readbacks.

## Nonlinear equilibrium bases and Cubature

The stable path keeps the equilibrium basis built from the model's rest
implicit Hessian, then applies the paper's current co-rotation (Eq. 14) at each
training, validation, and runtime pose:

```text
B_vi(x) = R_v(x) Ubar_vi transpose(R_i(x)).
```

For every tetrahedron, preprocessing evaluates the current stable
Neo-Hookean gradient and exact tangent, adds that element's incident share of
the inertial gradient and Hessian, and projects both through `B`. If the source
vertex belongs to the tetrahedron, its direct 3-vector and 3-by-3 block are
subtracted from the projected candidate because the runtime gathers those
terms exactly. Consequently, source plus all candidates with unit weight
matches `B^T g` and `B^T H B`; an independent dense CPU oracle checks both to
relative error below `1e-12`.

The deterministic v1 training corpus uses both signs of up to eight
low-frequency rest-Hessian modes. A direction is normalized by the larger of
its maximum tetrahedron displacement-gradient Frobenius norm and maximum
vertex displacement divided by scene scale, then given amplitude `0.12`.
Eight held-out modal mixtures use the training modes plus four validation-only
modes, amplitudes `0.06` and `0.18`, and an implicit target at one quarter of
the displacement. Each unconstrained connected component first removes its
own rigid displacement subspace; wholly unconstrained validation meshes then
add four exact global rotations. Every corpus pose
must have `min J >= 0.5`.

Following Eq. 18, every nontrivial pose target is normalized before a
deterministic nonnegative least-squares selector retains at most the scene's
four or six candidates. Cancellation-dominated targets are rejected. The
acceptance gates use the data actually packed for the runtime: weights and
rest basis blocks are round-tripped through `f32`, and packed columns are
measured against the original Float64 target so packing drift cannot cancel.
The normalized training residual must be at most `1%`, and the unregularized selected-update RMS must
be at most `2%` separately on training, held-out, and combined poses. The
versioned inputs and expected packed selection are frozen in
`manifests/phase1-cubature.v1.json`.

The checked-in capability fixture is intentionally one 12-tetrahedron,
edge-constrained solid. Every source has 12 full-rank candidates and retains
six, so it proves nonlinear projection, training, packing, and the production
shader path under genuine approximation. It does not claim preprocessing scale.
Larger fixed scenes should consume offline artifacts; arbitrary large browser
meshes need the later sparse-preprocessing work.

## Local positive-definite treatment

The Stable Neo-Hookean tangent can legitimately be indefinite away from rest;
"stable" does not mean convex. The exact oracle remains unmodified. For an
assembled symmetric 3-by-3 local solve matrix `H`, Phase 1 will use one uniform
diagonal shift:

```text
s   = max(max_i abs(H_ii), frobenius(H) / sqrt(3), m_i / h^2)
eta = 256 * 2^-24
tau = max(0, eta s - lambda_min(H))
rho = tau / s
H_solve = H + tau I.
```

The maximum permitted normalized shift is `rho <= 1e-3`. The GPU must expose
`rho`, the unshifted minimum eigenvalue, and `gradient dot direction`.
Nonzero-gradient directions must be descending. A local system requiring a
larger shift is rejected and investigated; the cap may not be silently raised.
If the nonlinear corpus does not leave at least fourfold headroom, the next
action is constitutive projection or Cubature diagnosis, followed by a recorded
threshold decision.

## Restricted local line search

The Armijo scalar must be the same restricted JGS2 subproblem whose derivatives
form the local gradient and Hessian: exact source inertia/targets/incident
elements plus the weighted complementary Cubature energy, with the same source
terms subtracted to prevent double counting.

```text
c1 = 1e-4
alpha_0 = 1
backtrack factor = 0.5
maximum backtracks = 12
accept iff L(alpha p) <= L(0) + c1 alpha gradient_dot_p
```

The implementation records initial energy, accepted energy, Armijo bound,
accepted alpha, `gradient dot p`, determinant feasibility, and normalized
shift. The assembled Jacobi energy is a separate diagnostic and is not used to
accept an individual parallel local step. If no positive alpha passes, the
local update is zero and the failure is reported.

## Convergence normalization

The relative residual remains the Phase 0 component-aware definition:

```text
r = norm(total gradient) /
    max(1, norm(inertia) + norm(material) + norm(external force)
           + norm(target) + norm(contact)).
```

The update metric is

```text
u = max_i norm(delta x_i) / max(scene scale, 1e-12).
```

A convergence-controlled runtime solve requires both `r` and `u` below its
configured thresholds. Tiny exact reference solves require `r <= 1e-5`; demo
thresholds may not exceed `1e-3`. Exact-frame and exact-iteration test APIs
disable early termination without changing the calculations inside an
iteration.

## Phase 1 performance gate

The Phase 1 development workload is a deterministic `10 x 10 x 8` cell solid
(`4,800` tetrahedra with the current six-tet cell generator), stable
Neo-Hookean material, no contact, and a recorded high-deformation interval. On
the nominated RTX 5090 it must meet `p95 <= 16.7 ms` over at least 120 warm-up
and 600 measured complete frames.

This smaller Phase 1 gate does not revise the final roadmap target. Phase 7
must still demonstrate a 20,000-30,000-tetrahedron non-contact solid at
`p95 <= 16.7 ms` using scalable precomputed artifacts.

## CPU, GPU, and WASM

- CPU Float64: exact material/implicit/local oracles, deterministic training,
  validation, small-scene preprocessing, and artifact construction.
- WebGPU: every per-frame deformation gradient, material evaluation, Cubature
  projection, local solve, line search, convergence reduction, state update,
  and rendering operation.
- WASM: not used in Phase 1. It would add a memory boundary without improving
  the GPU-resident hot loop. Reconsider only for large arbitrary in-browser
  sparse preprocessing; fixed larger scenes should load offline artifacts.
