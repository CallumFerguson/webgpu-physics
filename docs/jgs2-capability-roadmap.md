# JGS2 paper capability-parity roadmap

This document tracks the work required to match the capabilities demonstrated
by *JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU
Elastodynamics* without reproducing every paper scene or its million-element
scale.

The roadmap is intentionally sequential. Each phase must remain runnable,
reviewable, and testable. A phase is complete only when every mandatory exit
criterion is checked and its evidence is recorded.

## Capability target

Capability parity includes:

- implicit variational elastodynamics with general external forces;
- stable Neo-Hookean tetrahedral solids;
- the paper's equilibrium-basis JGS2 local solve with Cubature;
- Jacobi and graph-colored Gauss-Seidel scheduling;
- convergence-controlled nonlinear solves and local line search;
- general penalty contact;
- incremental potential contact (IPC);
- continuous collision detection (CCD);
- deformable-deformable and self-collision;
- IPC-compatible Coulomb friction;
- StVK cloth membranes with quadratic bending;
- interactive or scripted kinematic manipulation;
- a GPU-resident real-time frame loop on moderate scenes.

The following are explicitly out of scope:

- reproducing every scene or asset from the paper;
- matching the paper's million-element scene sizes;
- matching every timing or speedup reported in the paper;
- running preprocessing in the browser unless user-loaded meshes become a
  requirement.

## Advancement policy

1. Work proceeds in phase order unless an approved decision record changes a
   dependency.
2. A later phase may be prototyped, but it may not be marked in progress until
   every mandatory criterion in the preceding phase is complete.
3. A criterion is not complete until its command, result, and artifact or log
   are recorded.
4. Screenshots are supporting evidence only. Every dynamic scene must also
   assert semantic diagnostics.
5. A threshold may be changed only through a recorded decision explaining why
   the original threshold was invalid and what protects against regression.
6. Existing unit, build, and Playwright tests must remain passing throughout
   the roadmap.
7. Parity scenes must not rely on undocumented numerical corrections that
   alter physical momentum or contact response.

## Project fields

| Field | Value |
| --- | --- |
| Roadmap owner | Codex implementation agent; user approval retained |
| Technical reviewer | TBD |
| Target branch | main |
| Baseline commit | b3893b9 |
| Start date | 2026-07-16 |
| Target completion | TBD |
| Target browser and version | Google Chrome 150.0.7871.124 |
| Target GPU and driver | NVIDIA GeForce RTX 5090, driver 596.21 |
| Target CPU | Intel Core i7-13700K |
| Operating system | Windows 11 Home x64 |
| Baseline test manifest | manifests/baseline-tests.v1.json |
| Canonical scene/corpus manifest | manifests/canonical-scenes.v1.json |
| Maximum permitted regularization ratio | 1e-3 normalized local diagonal shift (P1-D01) |
| IPC safety epsilon | TBD before Phase 4 |
| Friction numerical tolerance | TBD before Phase 5 |
| Last updated | 2026-07-16 |

## Default measurement definitions

These definitions apply unless a phase explicitly overrides them.

- Relative error is norm(actual - reference) divided by
  max(1, norm(actual), norm(reference)).
- Scene scale is the diagonal of the initial dynamic-object bounding box.
- Mean edge length is measured from the undeformed collision surface.
- Linear-momentum error uses
  norm(p - p-reference) divided by max(norm(p-reference),
  total-mass times scene-scale per second).
- Angular-momentum error uses
  norm(L - L-reference) divided by max(norm(L-reference),
  total-mass times scene-scale squared per second).
- Formal qualification performance measurements use at least 120 warm-up
  frames followed by at least 600 measured frames. The fast isolated baseline
  may use one combined 30-warm-up/120-measured CPU/GPU timestamp pass, and fast
  screenshot tests may use a 0/12 in-place combined interval for smoke
  telemetry. Neither shorter interval can be recorded as formal performance
  evidence.
- Performance measurements exclude screenshots, diagnostics readbacks, test
  traces, and preprocessing.
- Average FPS is `1000 / mean(frame milliseconds)`. The 1% low is
  `1000 / mean(the slowest ceil(1%) frame milliseconds)`; instantaneous FPS
  values are never averaged.
- FPS from the serialized test harness is compute throughput, not monitor FPS
  or production simulation time rate. The instrumentation gate requires
  serialized wall mean and, when timestamp-query is available, GPU-frame p95
  to fit the step budget; serialized wall p95 and 1% low remain diagnostics
  because every sample includes a queue drain. This gate is necessary but not
  sufficient for a real-time claim; any such claim must also exercise
  production animation scheduling and bounded catch-up behavior.
- CPU performance values mean main-thread preparation, command encoding, and
  queue submission time. They exclude GPU execution/waiting and are never
  inferred by subtracting GPU timestamps from wall time.
- GPU frame, simulation-step, and render distributions use a fresh replay of
  the wall/CPU interval, one timestamp-buffer map, and byte-identical final
  f32 state relative to the uninstrumented replay.
- Collision detection, CCD, contact assembly, solve, velocity update, and
  rendering are included in frame time.
- The runtime may have no more than two GPU submissions in flight.
- Unless a stricter phase criterion applies, no test may emit a WebGPU
  validation error, device loss, uncaught page error, NaN, or infinity.
- A diagnostic over an empty set must expose a finite value and a separate
  validity flag. Tests must assert that the flag is false rather than treating
  the finite sentinel as a measurement.

## E2E execution tiers

- `npm.cmd run test:e2e` is the under-one-minute routine hardware-WebGPU
  feedback gate on the nominated machine. It runs every collected
  test, advances all 32 frozen force-free states by one step, and continues
  frozen cases `00` and `27` through 120 steps (one simulated second). The base
  force-free trajectory remains 1,200 steps (10 seconds).
- Every fast screenshot test continues its current state in place for one
  0-warm-up/12-measured combined telemetry interval with simultaneous wall/CPU
  and GPU timestamp instrumentation when supported. It logs and attaches FPS,
  1% low, wall, CPU, and GPU metrics without reloading. CPU timing includes the
  timestamp instrumentation only when supported, and the 12-frame tail metrics
  are smoke-only. The isolated
  stress baseline separately records one combined 30/120 CPU/GPU timestamp
  pass as smoke telemetry.
- `npm.cmd run test:e2e:full` is the qualification gate. It restores all 32
  force-free cases to the canonical 1,200-step trajectory and records formal
  120-warm-up/600-measured CPU/GPU profiles for the isolated baseline and each
  unique short visual workload. Long drop/stress tests retain the combined
  0/12 smoke interval, continuing their actual settled state rather than
  reloading frame zero.
- The fast policy does not modify the canonical scene/corpus manifest or
  historical evidence. Criteria requiring the complete 32-by-1,200 corpus or
  formal performance samples must use `test:e2e:full`.

## Reproducibility and criterion evidence

- Every mandatory exit criterion has a stable ID in the form
  P-phase-EC-number.
- Every canonical scene and randomized corpus must be defined in a checked-in
  manifest containing its mesh or generator version, seed, material values,
  timestep, iteration settings, sampled frames, camera, and expected metrics.
- Phase 0 freezes a baseline-test manifest. Later gates require that manifest
  and the complete current suite to pass with zero unexpected skips.
- Evidence for a criterion is recorded in the criterion evidence log using the
  exact criterion ID, commit, command, result, artifact, date, and reviewer.
- A result marked not applicable requires a decision-log entry and cannot
  silently satisfy a capability-parity criterion.

Recommended final performance targets are:

| Workload | Required p95 frame time |
| --- | --- |
| Non-contact solid with 20,000-30,000 tetrahedra | <= 16.7 ms |
| Penalty contact with about 20,000 total tetrahedra | <= 33.3 ms |
| IPC or self-contact with 10,000-20,000 tetrahedra | <= 33.3 ms |
| Static-obstacle cloth with about 32,000 triangles | <= 33.3 ms |
| Self-colliding cloth with about 8,000 triangles | <= 33.3 ms |

If the nominated hardware cannot reasonably meet these defaults, record revised
targets in the decision log before Phase 1 begins.

## Overall progress

| Phase | Name | Depends on | Status | Exit review |
| --- | --- | --- | --- | --- |
| 0 | Reference foundation and parity-safe architecture | None | Complete | Independent gate audit passed |
| 1 | Stable Neo-Hookean solids and nonlinear JGS2 | Phase 0 | In progress | TBD |
| 2 | General collision detection and swept candidates | Phase 1 | Not started | TBD |
| 3 | Pairwise penalty contact and contact-aware JGS2 | Phase 2 | Not started | TBD |
| 4 | IPC, CCD feasibility, and local line search | Phase 3 | Not started | TBD |
| 5 | IPC-compatible Coulomb friction | Phase 4 | Not started | TBD |
| 6 | Cloth and thin-shell energy elements | Phase 5 | Not started | TBD |
| 7 | Gauss-Seidel, sparse preprocessing, and final parity | Phase 6 | Not started | TBD |

---

## Phase 0: Reference foundation and parity-safe architecture

### Objective

Create the exact reference calculations, diagnostics, test controls, and
runtime interfaces required to evaluate every later capability independently.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [ ] Not started [ ] In progress [ ] Blocked [x] Complete |
| Owner | Codex implementation agent |
| Reviewer | Independent Phase 0 gate audit (Codex sub-agent) |
| Start date | 2026-07-16 |
| Completion date | 2026-07-16 |
| Branch or PR | main; foundation commit 1453c48; Phase 0 completion milestone (this commit) |
| Design records | docs/reproducibility.md; docs/evidence/phase0-performance-baseline.md |
| Primary test command | npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e |
| Performance result | Stress scene: mean 3.832 ms, p95 5.000 ms wall; explicit GPU simulation timestamp 1.376256 ms |
| Known limitations | Phase 0 remains co-rotated and has no general body-body candidate buffer. Candidate overflow therefore reports a finite zero sentinel with valid=false until Phase 2. General contact, nonlinear material, friction, and cloth remain later-phase work. |
| Exit sign-off | Pass: independent audit plus exact-current-code unit, build, baseline-selector, and 14-test hardware suite |

### Required implementation

- [x] Add a small CPU Float64 reference path for total energy, gradient,
      Hessian blocks, and full Newton steps.
- [x] Keep an exact all-element, no-Cubature path for tiny systems.
- [x] Define shared interfaces for material, external-force, constraint, and
      contact energies.
- [x] Add deterministic APIs for stepping exact frames and exact nonlinear
      iterations.
- [x] Add GPU reductions for total energy, relative residual, maximum update,
      minimum deformation determinant, minimum contact distance, active
      contact count, candidate-buffer overflow, and finite-state status.
- [x] Add per-body linear and angular momentum diagnostics.
- [x] Add GPU timestamp instrumentation where the adapter supports it.
- [x] Define a versioned precomputation artifact schema with topology,
      materials, timestep, basis, Cubature, and solver-schedule fingerprints.
- [x] Add an explicit parity mode that disables project-specific physical
      approximations.
- [x] Freeze a checked-in baseline-test manifest with expected test IDs and
      allowed skips.
- [x] Define the Phase 0 canonical scenes and corpora in the checked-in
      scene/corpus manifest.
- [x] Record the nominated hardware and final performance targets above.

### Mandatory exit criteria

- [x] P0-EC-01: CPU analytic gradients match central finite differences with relative
      error <= 1e-5 on every reference energy available in this phase.
- [x] P0-EC-02: CPU analytic Hessians match finite differences of the gradient with
      relative error <= 1e-4.
- [x] P0-EC-03: GPU energy, gradient, and local Hessian blocks match the CPU reference
      with relative error <= 1e-3.
- [x] P0-EC-04: The exact equilibrium-basis local update matches the corresponding block
      of a full CPU Newton solve with relative error <= 1e-8 on CPU and
      <= 1e-3 on GPU.
- [x] P0-EC-05: Full-coordinate bases from equations 19-23 match direct
      complementary-coordinate solves with relative error <= 1e-8 on the
      canonical tiny systems.
- [x] P0-EC-06: The deterministic harness can pause animation, step exact frames, step
      exact nonlinear iterations, and wait for GPU completion.
- [x] P0-EC-07: Every required diagnostic is finite when valid; empty-set
      diagnostics expose a false validity flag and the documented finite
      sentinel.
- [x] P0-EC-08: Production animation performs no synchronous per-frame GPU readback.
- [x] P0-EC-09: Parity mode uses velocity damping equal to 1 and disables grounded
      tangential damping and horizontal COM restoration.
- [x] P0-EC-10: Force-free isolated-body tests conserve linear momentum within 0.5% over
      10 simulated seconds in parity mode.
- [x] P0-EC-11: The same force-free tests conserve angular momentum within
      0.5% using the roadmap's angular-momentum normalization.
- [x] P0-EC-12: The frozen baseline manifest and the complete current unit,
      build, and Playwright suites pass with zero unexpected skips and no
      WebGPU errors.
- [x] P0-EC-13: A baseline performance report is recorded on the nominated hardware.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| CPU finite-difference report | Canonical 64-pose corpus: worst gradient 1.221e-10; worst Hessian 4.389e-12 | [x] |
| GPU-versus-CPU report | Canonical sampled poses: energy 1.659e-6; gradient 1.847e-6; local Hessian 3.173e-8. Floor-active case: energy 1.237e-6; gradient 3.205e-5; local Hessian 7.120e-8 | [x] |
| Newton-block oracle result | CPU worst 1.251e-17; GPU worst 5.469e-9 over all 64 poses and three active blocks; basis 4.437e-16 | [x] |
| Deterministic test artifacts | Exact-frame/even-iteration assertions plus canonical start, fixed-camera final, and follow-camera final screenshots | [x] |
| Full regression output | 74 unit tests, build, frozen 26/74 unit and 7/14 E2E selectors, and 14/14 hardware E2E tests passed with zero skips | [x] |
| Baseline performance report | docs/evidence/phase0-performance-baseline.md | [x] |
| Reviewer approval | Independent Phase 0 criterion audit passed after the exact-current-code full suite | [x] |

---

## Phase 1: Stable Neo-Hookean solids and nonlinear JGS2

### Objective

Replace the co-rotated linear parity path with the stable Neo-Hookean solid
model used by the paper and evaluate the JGS2 local system using the current
nonlinear energy gradient and Hessian.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [ ] Not started [x] In progress [ ] Blocked [ ] Complete |
| Owner | Codex implementation agent |
| Reviewer | TBD |
| Start date | 2026-07-16 |
| Completion date | TBD |
| Branch or PR | main; Phase 0 gate ce478e6; Phase 1 material/runtime gate c0f5456 |
| Design records | docs/phase1-stable-neo-hookean.md; docs/jgs2-implementation.md; docs/evidence/phase1-nonlinear-cubature.md; manifests/phase1-scenes.v1.json; manifests/phase1-cubature.v1.json |
| Primary test command | Routine: npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e. Qualification: npm.cmd run test:e2e:full |
| Performance result | TBD |
| Known limitations | CPU/WGSL material derivatives, exact diagnostics, material dispatch, affine-exact frames, nonlinear current-pose Cubature, packed-f32 quality gates, and production update parity are implemented. Dense TypeScript preprocessing is intentionally limited to small fixtures and fails closed when K=6 misses the 1% gate. The shared ABI still carries legacy rest stiffness for stable-only meshes. Globalization and assembled-pose feasibility, explicit public-scene material labels, forces/handles, scenes, and performance evidence remain. |
| Exit sign-off | TBD |

### Required implementation

- [x] Implement stable Neo-Hookean energy, gradient, and tangent Hessian in the
      CPU reference.
- [x] Implement the same current-pose calculations in WGSL.
- [x] Generalize material dispatch so stable constitutive evaluation uses the
      current-pose tangent rather than the constant rest 12-by-12 stiffness.
- [x] Compute documented vertex-local deformation gradients and polar frames
      for the co-rotated equilibrium bases.
- [x] Evaluate current complementary gradients and Hessians in the Cubature
      projection.
- [x] Retrain the equilibrium bases and Cubature data for the nonlinear model.
- [ ] Add arbitrary per-vertex external forces.
- [ ] Add scripted and pointer-driven kinematic targets with clean release.
- [ ] Add GPU-side convergence flags based on residual and update norms.
- [ ] Add monotone local line-search infrastructure for nonlinear updates.
- [ ] Document positive-definite projection or regularization for each local
      Hessian and expose its applied shift as a diagnostic.
- [ ] Preserve co-rotated linear elasticity only as an explicitly labeled
      regression or debugging material in public scene metadata.

### Required scenes

- [ ] A single tetrahedron under rigid rotation, stretch, and compression.
- [ ] Stable Neo-Hookean soft and stiff cantilevers under identical gravity.
- [ ] A small body following a scripted grab, hold, circular path, and release.
- [ ] A high-deformation solid that exercises the nonlinear tangent.

### Mandatory exit criteria

- [x] P1-EC-01: Rest-pose energy and force are zero within CPU relative tolerance 1e-10
      and GPU relative tolerance 1e-5.
- [x] P1-EC-02: Rigid rotation changes energy by <= 1e-8 on CPU and <= 1e-4 on GPU and
      produces no material force above the same normalized tolerances.
- [x] P1-EC-03: Neo-Hookean CPU gradients and Hessians meet the Phase 0 finite-difference
      tolerances.
- [x] P1-EC-04: GPU Neo-Hookean energy, gradients, and Hessian blocks match CPU with
      relative error <= 1e-3.
- [ ] P1-EC-05: Tests at deformation determinants 0.5, 0.1, and 0.01 remain finite;
      non-positive determinants are rejected or made infeasible before
      material evaluation according to the documented inversion policy.
- [ ] P1-EC-06: Every dynamic checkpoint in the canonical Phase 1 manifest has
      minimum deformation determinant >= 1e-4.
- [ ] P1-EC-07: Every accepted local line-search step decreases its explicitly
      defined restricted JGS2 subproblem energy by at least the configured
      Armijo decrease; assembled Jacobi energy is recorded separately and is
      not used to accept an individual local step.
- [ ] P1-EC-08: For every canonical nonzero-gradient local system, the accepted
      direction satisfies gradient dot direction < 0 after the documented
      positive-definite treatment.
- [ ] P1-EC-09: The normalized local diagonal shift never exceeds the maximum
      permitted regularization ratio recorded in the project fields.
- [ ] P1-EC-10: Tiny reference solves finish with relative residual <= 1e-5; runtime
      demos finish within their configured threshold, which may not exceed
      1e-3.
- [x] P1-EC-11: Selected Cubature data reaches <= 1% normalized training residual for
      every parity scene.
- [x] P1-EC-12: Selected-Cubature local updates differ from exact projected updates by
      <= 2% RMS over the training and validation poses.
- [ ] P1-EC-13: The soft cantilever tip deflects at least 1.5 times as far as the stiff
      tip under the same load.
- [ ] P1-EC-14: The scripted handle tracks its target within 2% of body diameter and
      releases without leaving a hidden constraint.
- [ ] P1-EC-15: Start, peak-deformation, and released screenshots are saved and their
      semantic diagnostics pass.
- [ ] P1-EC-16: The canonical Phase 1 performance scene and exact sampled
      frame range are recorded in the manifest, meet the serialized
      non-contact compute gate, and pass a production-scheduler simulation
      time-rate gate.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| Neo-Hookean derivative report | CPU 60-pose corpus: worst gradient 8.975e-10; worst Hessian 1.957e-11; exact tangent symmetry and rest-stiffness parity pass | [x] |
| Objectivity and rest-state report | CPU density, stress, tangent action, translated tetrahedron, internal force/torque, and translational null-mode tests pass; f32-exact GPU quarter turns have rest and rigid energy/force errors 0/0 under the roadmap default zero-reference metric | [x] |
| CPU-versus-GPU report | Frozen 64-pose material-only hardware corpus: worst relative energy 2.234e-6, gradient 1.161e-6, local Hessian 1.089e-6; total implicit parity also <=1e-3; scaled non-affine vertex-frame error 8.702e-8 | [x] |
| Convergence history | TBD | [ ] |
| Cubature residual report | 12-tet full-rank parity fixture, K=6: packed residual maximum 0.548989%; packed selected-vs-independent-dense update RMS training 3.713075e-5, held-out 3.984377e-5, combined 3.796498e-5; two-iteration production GPU parity 2.113e-9 | [x] |
| Scene screenshots and diagnostics | TBD | [ ] |
| Performance report | TBD | [ ] |
| Reviewer approval | TBD | [ ] |

---

## Phase 2: General collision detection and swept candidates

### Objective

Build and validate the collision geometry pipeline before adding collision
forces, so candidate-generation failures can be separated from contact-solver
failures.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [x] Not started [ ] In progress [ ] Blocked [ ] Complete |
| Owner | TBD |
| Reviewer | TBD |
| Start date | TBD |
| Completion date | TBD |
| Branch or PR | TBD |
| Design records | TBD |
| Primary test command | TBD |
| Performance result | TBD |
| Known limitations | TBD |
| Exit sign-off | TBD |

### Required implementation

- [ ] Represent collision surface vertices, triangles, and unique edges.
- [ ] Store body IDs, primitive IDs, adjacency exclusions, collision thickness,
      and rigid or kinematic obstacle data.
- [ ] Implement a brute-force CPU collision-pair oracle.
- [ ] Build static acceleration topology on CPU or offline.
- [ ] Refit and traverse a GPU spatial hash or BVH each frame.
- [ ] Generate vertex-triangle and edge-edge candidate pairs on GPU.
- [ ] Generate candidates from swept bounds for trial motion.
- [ ] Implement robust closest-point, distance, normal, and mollifier
      calculations.
- [ ] Compact candidates and explicitly report buffer overflow.
- [ ] Add collision-pair and bounding-volume debug rendering.

### Required scenes

- [ ] Two triangle meshes approaching without contact forces.
- [ ] A deformable surface approaching an arbitrarily oriented obstacle.
- [ ] A folded surface producing non-adjacent self-collision candidates.
- [ ] High-speed crossing trajectories that require swept candidates.

### Mandatory exit criteria

- [ ] P2-EC-01: GPU broad-phase output has zero false negatives relative to brute force
      over at least 1,000 deterministic randomized configurations.
- [ ] P2-EC-02: Every crossing enumerated in the checked-in Phase 2 swept-motion
      corpus appears in the swept candidate set.
- [ ] P2-EC-03: Vertex-triangle and edge-edge distances match the CPU oracle within
      1e-4 times scene scale.
- [ ] P2-EC-04: Every exclusion pair enumerated by the checked-in topology
      manifest is absent from the self-collision candidate set.
- [ ] P2-EC-05: Parallel-edge, near-degenerate, and zero-relative-motion cases remain
      finite and deterministic.
- [ ] P2-EC-06: Candidate IDs are valid, duplicate handling is documented, and candidate
      overflow fails visibly instead of dropping pairs silently.
- [ ] P2-EC-07: The pair-debug scene produces saved screenshots before, during, and
      after closest approach with matching candidate-count assertions.
- [ ] P2-EC-08: The canonical Phase 2 performance scene, sampled frames, and
      candidate-count distribution are recorded in the manifest and meet
      p95 <= 33.3 ms on the nominated hardware.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| Randomized broad-phase oracle report | TBD | [ ] |
| Swept-candidate test report | TBD | [ ] |
| Distance and degeneracy report | TBD | [ ] |
| Overflow behavior evidence | TBD | [ ] |
| Debug screenshots and diagnostics | TBD | [ ] |
| Performance report | TBD | [ ] |
| Reviewer approval | TBD | [ ] |

---

## Phase 3: Pairwise penalty contact and contact-aware JGS2

### Objective

Replace the hard-coded horizontal floor with general pairwise penalty contact
and prove that multiple bodies and self-contact participate in the JGS2 solve
with physically consistent action and reaction.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [x] Not started [ ] In progress [ ] Blocked [ ] Complete |
| Owner | TBD |
| Reviewer | TBD |
| Start date | TBD |
| Completion date | TBD |
| Branch or PR | TBD |
| Design records | TBD |
| Primary test command | TBD |
| Performance result | TBD |
| Known limitations | TBD |
| Exit sign-off | TBD |

### Required implementation

- [ ] Implement pair-distance penalty energy, gradient, and Hessian for
      vertex-triangle and edge-edge contacts.
- [ ] Evaluate active contact terms exactly rather than through fixed elastic
      Cubature samples.
- [ ] Apply equal-and-opposite contact contributions to both participants.
- [ ] Implement the paper's contact-aware perturbation-basis approximation for
      the contacted body.
- [ ] Support deformable-deformable, deformable-obstacle, and non-adjacent
      self-contact.
- [ ] Support arbitrary contact normals and moving kinematic obstacles.
- [ ] Disable horizontal COM restoration in every general-contact scene.
- [ ] Disable global and grounded tangential damping in frictionless
      conservation tests.

### Required scenes

- [ ] A deformable block contacting an oblique plane.
- [ ] Two deformable blocks colliding head-on.
- [ ] Two deformable blocks colliding obliquely.
- [ ] A compressed U-shaped solid exercising self-contact.

### Mandatory exit criteria

- [ ] P3-EC-01: Penalty gradients and Hessians match CPU finite differences with relative
      error <= 1e-4 on CPU and <= 1e-3 on GPU.
- [ ] P3-EC-02: Each isolated contact pair's summed force is <= 1e-4 of the larger
      participant-force norm.
- [ ] P3-EC-03: A frictionless two-body collision changes total system linear momentum
      by < 0.5%, excluding documented external impulses.
- [ ] P3-EC-04: The same collision changes angular momentum about the system
      center of mass by < 0.5% using the roadmap normalization.
- [ ] P3-EC-05: In the canonical frictionless plane-contact case, tangential
      contact impulse magnitude is <= 1e-4 of normal contact impulse magnitude.
- [ ] P3-EC-06: Maximum penetration in the canonical penalty scenes is <= 1% of
      mean surface edge length.
- [ ] P3-EC-07: Arbitrarily oriented planes produce positions and velocities within
      1e-3 scene scale of the equivalent axis-aligned test after inverse
      transformation.
- [ ] P3-EC-08: The independent checker reports zero triangle intersections at
      every simulated frame of the canonical head-on and oblique collisions;
      both bodies have nonzero equal-and-opposite integrated contact impulse.
- [ ] P3-EC-09: The independent checker reports zero self-intersections at every
      simulated frame of the canonical U-shaped closing test.
- [ ] P3-EC-10: GPU contact-aware local updates match a CPU implementation of
      the same documented paper approximation within relative error <= 1e-3.
- [ ] P3-EC-11: Error between that approximation and the full dense contact
      Newton block is measured and recorded for the canonical tiny systems;
      it is not treated as implementation error.
- [ ] P3-EC-12: No parity scene dispatches the horizontal COM correction.
- [ ] P3-EC-13: Start, first-contact, maximum-compression, and final screenshots are
      saved with contact-count, momentum, penetration, and determinant checks.
- [ ] P3-EC-14: The canonical Phase 3 performance scene and sampled frames meet
      p95 <= 33.3 ms.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| Penalty derivative report | TBD | [ ] |
| Action-reaction and momentum report | TBD | [ ] |
| Contact-aware basis oracle result | TBD | [ ] |
| Penetration and self-contact report | TBD | [ ] |
| Scene screenshots and diagnostics | TBD | [ ] |
| Performance report | TBD | [ ] |
| Reviewer approval | TBD | [ ] |

---

## Phase 4: IPC, CCD feasibility, and local line search

### Objective

Add the paper's IPC barrier capability and ensure every accepted nonlinear
update remains collision-free, including high-speed and self-contact cases.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [x] Not started [ ] In progress [ ] Blocked [ ] Complete |
| Owner | TBD |
| Reviewer | TBD |
| Start date | TBD |
| Completion date | TBD |
| Branch or PR | TBD |
| Design records | TBD |
| Primary test command | TBD |
| Performance result | TBD |
| Known limitations | TBD |
| Exit sign-off | TBD |

### Required implementation

- [ ] Implement the IPC logarithmic barrier energy and activation distance.
- [ ] Implement analytic barrier gradients and Hessians for active primitive
      pairs.
- [ ] Require and validate a collision-free initial state.
- [ ] Compute conservative CCD time-of-impact or safe-step bounds.
- [ ] Clamp every trial step to the feasible interval before evaluating IPC.
- [ ] Add parallel per-subproblem backtracking or equivalent local line search.
- [ ] Refresh or conservatively maintain active pairs during nonlinear
      iterations.
- [ ] Include IPC terms in the contact-aware JGS2 basis projection.
- [ ] Add an independent CPU intersection checker for tests.

### Required scenes

- [ ] A vertex-triangle approach test.
- [ ] An edge-edge approach test.
- [ ] Two high-speed deformable bars colliding.
- [ ] A three-body IPC stack.
- [ ] A compressed self-colliding U-shaped solid.
- [ ] Side-by-side penalty and IPC versions of the same impact.

### Mandatory exit criteria

- [ ] P4-EC-01: IPC barrier gradients and Hessians meet relative error <= 1e-4 on CPU
      and <= 1e-3 on GPU.
- [ ] P4-EC-02: The recorded IPC safety epsilon is positive and
      <= 0.01 times activation distance; barrier evaluations at distances
      activation-distance times 0.9, 0.5, 0.25, 0.1, 0.01, and at safety
      epsilon are finite and strictly increase as distance decreases.
- [ ] P4-EC-03: Barrier energy at safety epsilon is at least 10 times its value
      at 0.1 times activation distance; the implementation never evaluates the
      barrier at non-positive distance.
- [ ] P4-EC-04: CCD reports zero missed crossings over the deterministic adversarial
      vertex-triangle and edge-edge corpus.
- [ ] P4-EC-05: Every accepted step is no larger than its conservative safe-step bound.
- [ ] P4-EC-06: Every accepted line-search step leaves all active distances positive.
- [ ] P4-EC-07: Every accepted local line-search step satisfies its documented
      Armijo decrease for the restricted contact-aware subproblem energy.
- [ ] P4-EC-08: The independent CPU checker reports zero intersections in all IPC E2E
      checkpoints.
- [ ] P4-EC-09: Minimum contact separation remains above the recorded numerical safety
      epsilon.
- [ ] P4-EC-10: The independent checker reports zero intersections at every
      simulated frame of the canonical high-speed and self-contact scenes.
- [ ] P4-EC-11: In the calibrated penalty-versus-IPC comparison, the penalty version
      penetrates by 0.5%-1% of mean edge length while the IPC version remains
      above its documented positive safety epsilon.
- [ ] P4-EC-12: No IPC test relies on maximum world-space update clamping as a
      substitute for CCD feasibility.
- [ ] P4-EC-13: Start, closest-contact, and final screenshots are saved with distance,
      CCD, line-search, residual, and determinant diagnostics.
- [ ] P4-EC-14: The canonical 10,000-20,000-tetrahedron IPC scene and sampled
      frames meet
      p95 <= 33.3 ms, or an approved target revision is recorded.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| IPC derivative report | TBD | [ ] |
| CCD adversarial-corpus report | TBD | [ ] |
| Line-search energy history | TBD | [ ] |
| Independent intersection report | TBD | [ ] |
| Penalty-versus-IPC comparison | TBD | [ ] |
| Scene screenshots and diagnostics | TBD | [ ] |
| Performance report | TBD | [ ] |
| Reviewer approval | TBD | [ ] |

---

## Phase 5: IPC-compatible Coulomb friction

### Objective

Replace proximity-based ground drag with pairwise friction that supports
static sticking, kinetic sliding, arbitrary contact normals, moving bodies,
and momentum-consistent body-body interaction.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [x] Not started [ ] In progress [ ] Blocked [ ] Complete |
| Owner | TBD |
| Reviewer | TBD |
| Start date | TBD |
| Completion date | TBD |
| Branch or PR | TBD |
| Design records | TBD |
| Primary test command | TBD |
| Performance result | TBD |
| Canonical static coefficient mu-s | TBD before implementation |
| Canonical kinetic coefficient mu-k | TBD before implementation |
| Sliding acceleration sample window | TBD before implementation |
| Known limitations | TBD |
| Exit sign-off | TBD |

### Required implementation

- [ ] Select and document an IPC-compatible regularized or lagged Coulomb
      friction formulation.
- [ ] Implement the same friction potential, gradient, and tangent in the CPU
      Float64 oracle.
- [ ] Track persistent contact pairs and relative tangential displacement.
- [ ] Build stable tangent frames for arbitrary contact normals.
- [ ] Tie the friction bound to normal contact force and coefficient mu.
- [ ] Apply equal-and-opposite tangential contributions to both participants.
- [ ] Support different material-pair friction coefficients.
- [ ] Preserve a mu = 0 path equivalent to normal-only IPC.
- [ ] Define tangent-frame update and continuity behavior for persistent pairs.
- [ ] Remove grounded x/z damping from every parity scene.

### Required scenes

- [ ] A block on an incline below the critical stick angle.
- [ ] A block on an incline above the critical stick angle.
- [ ] Two moving deformable bodies in frictional contact.
- [ ] A small card house that stands before and collapses after a scripted
      impact.

### Mandatory exit criteria

- [ ] P5-EC-01: CPU friction gradients and tangents match finite differences
      with relative error <= 1e-4; GPU values match CPU within 1e-3.
- [ ] P5-EC-02: Across the checked-in rotating-normal corpus, a 1e-4-radian
      normal change produces no tangent-frame sign flip and changes the
      world-space friction force by <= 1e-3 relative error.
- [ ] P5-EC-03: With incline angle equal to 0.8 times atan(mu-s), drift over
      10 simulated seconds is
      < 0.5% of block length.
- [ ] P5-EC-04: With incline angle equal to 1.2 times atan(mu-s), downslope
      acceleration measured over the recorded post-transient sample window is
      within 10% of gravity times
      (sin(theta) - mu-k times cos(theta)).
- [ ] P5-EC-05: Tangential impulse magnitude stays within the recorded friction-cone
      tolerance relative to mu times normal impulse.
- [ ] P5-EC-06: Frictional work is non-positive beyond the recorded floating-point
      tolerance.
- [ ] P5-EC-07: The mu = 0 result matches normal-only IPC within 1e-3 scene scale.
- [ ] P5-EC-08: Pairwise friction changes total system linear momentum by < 0.5% when
      there are no external forces.
- [ ] P5-EC-09: The same force-free pairwise test changes angular momentum
      about the system center of mass by < 0.5%.
- [ ] P5-EC-10: With mu-high at least twice mu-low under otherwise identical conditions,
      the high-friction stopping distance is <= 75% of the low-friction
      stopping distance.
- [ ] P5-EC-11: The unstruck card house remains standing for at least 10 simulated
      seconds.
- [ ] P5-EC-12: After the scripted strike, the card-house height decreases by at least
      25% and at least half the cards rotate by more than 30 degrees.
- [ ] P5-EC-13: No friction scene dispatches grounded tangential damping or horizontal
      COM restoration.
- [ ] P5-EC-14: Start, stick-or-slide, impact, and final screenshots are saved with
      normal impulse, tangential impulse, work, momentum, and distance checks.
- [ ] P5-EC-15: The canonical Phase 5 friction performance scene and sampled
      frames meet p95 <= 33.3 ms.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| Friction formulation record | TBD | [ ] |
| Incline analytic-comparison report | TBD | [ ] |
| Friction-cone and work report | TBD | [ ] |
| Pairwise momentum report | TBD | [ ] |
| Card-house stability report | TBD | [ ] |
| Scene screenshots and diagnostics | TBD | [ ] |
| Performance report | TBD | [ ] |
| Reviewer approval | TBD | [ ] |

---

## Phase 6: Cloth and thin-shell energy elements

### Objective

Add the paper's co-dimensional capability using StVK triangle membrane energy,
quadratic bending, and the completed IPC and friction pipeline.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [x] Not started [ ] In progress [ ] Blocked [ ] Complete |
| Owner | TBD |
| Reviewer | TBD |
| Start date | TBD |
| Completion date | TBD |
| Branch or PR | TBD |
| Design records | TBD |
| Primary test command | TBD |
| Performance result | TBD |
| Known limitations | TBD |
| Exit sign-off | TBD |

### Required implementation

- [ ] Add triangle membrane elements with StVK in-plane energy.
- [ ] Add hinge elements with quadratic bending energy.
- [ ] Add triangle-area masses, thickness, and material parameters.
- [ ] Extend equilibrium-basis and Cubature preprocessing to cloth element
      families.
- [ ] Use separate solid and cloth element pipelines behind shared JGS2,
      contact, convergence, and diagnostics interfaces.
- [ ] Add cloth-solid contact and cloth self-collision exclusions.
- [ ] Support IPC and friction for co-dimensional contacts.
- [ ] Add double-sided cloth rendering and contact-debug overlays.

### Required scenes

- [ ] A single-triangle membrane patch.
- [ ] A two-triangle bending hinge.
- [ ] A cloth patch draping over a rigid sphere.
- [ ] A cloth patch draping over a deformable solid.
- [ ] A self-folding and self-colliding cloth patch.

### Mandatory exit criteria

- [ ] P6-EC-01: StVK membrane gradients and Hessians meet relative error <= 1e-4 on CPU
      and <= 1e-3 on GPU.
- [ ] P6-EC-02: Quadratic bending gradients and Hessians meet the same tolerances.
- [ ] P6-EC-03: Rigid transformation changes membrane and bending energy by <= 1e-8 on
      CPU and <= 1e-4 on GPU.
- [ ] P6-EC-04: Membrane patch and hinge force curves match the CPU reference within
      relative error <= 1e-3 on GPU.
- [ ] P6-EC-05: Pinned cloth vertices remain within 1e-5 scene scale of their targets.
- [ ] P6-EC-06: Cloth energy, strain diagnostics, and bending diagnostics remain finite.
- [ ] P6-EC-07: The independent intersection checker reports no cloth-solid or cloth
      self-intersections at E2E checkpoints.
- [ ] P6-EC-08: The self-folding scene produces active non-adjacent self-contact; during
      a scripted separating release it remains intersection-free and its
      minimum non-adjacent gap increases by at least one contact activation
      distance.
- [ ] P6-EC-09: Start, first-contact, peak-wrinkle, and final screenshots are saved with
      strain, bending, distance, contact-count, and residual checks.
- [ ] P6-EC-10: The canonical static-obstacle cloth scene with about 32,000
      triangles and its sampled frames meet p95 <= 33.3 ms,
      or an approved target revision is recorded.
- [ ] P6-EC-11: The canonical self-colliding cloth scene with about 8,000
      triangles and its sampled frames meet p95 <= 33.3 ms, or
      an approved target revision is recorded.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| Membrane derivative report | TBD | [ ] |
| Bending derivative report | TBD | [ ] |
| Rigid-invariance report | TBD | [ ] |
| Cloth collision report | TBD | [ ] |
| Scene screenshots and diagnostics | TBD | [ ] |
| Performance report | TBD | [ ] |
| Reviewer approval | TBD | [ ] |

---

## Phase 7: Gauss-Seidel, sparse preprocessing, and final parity

### Objective

Complete the remaining solver variants, replace demo-scale preprocessing for
moderate assets, optimize the measured bottlenecks, and validate all
capabilities together.

### Tracking fields

| Field | Value |
| --- | --- |
| Status | [x] Not started [ ] In progress [ ] Blocked [ ] Complete |
| Owner | TBD |
| Reviewer | TBD |
| Start date | TBD |
| Completion date | TBD |
| Branch or PR | TBD |
| Design records | TBD |
| Primary test command | TBD |
| Performance result | TBD |
| Known limitations | TBD |
| Exit sign-off | TBD |

### Required implementation

- [ ] Build and validate a vertex or generalized-subproblem conflict coloring.
- [ ] Generate Gauss-Seidel group equilibrium bases and Cubature data for that
      schedule.
- [ ] Dispatch color groups sequentially and members of a color in parallel.
- [ ] Define dynamic-contact conflict handling through dynamic coloring or an
      explicit Jacobi fallback.
- [ ] Retain Jacobi as a selectable and fully tested solver.
- [ ] Implement a versioned offline sparse preprocessor using one reusable
      sparse factorization, batched basis solves, sparse low modes, and scalable
      nonnegative Cubature training.
- [ ] Load precomputed binary artifacts in production and retain the dense
      TypeScript implementation as the tiny-system oracle.
- [ ] Record an explicit decision on whether in-browser custom meshes justify
      a Web Worker and WASM sparse-preprocessing path.
- [ ] Check in a convergence-protocol manifest containing the beam mesh,
      timestep, material values, initial state, reference solution, residual
      normalization, sampled iteration indices, fit window, and baseline
      implementation version.
- [ ] Check in a solver/capability matrix. Jacobi must cover every composite
      capability; Gauss-Seidel must cover at least non-contact Neo-Hookean,
      penalty contact, and IPC contact with the corresponding generalized
      contact-aware bases.
- [ ] Profile and optimize GPU collision refit, traversal, candidate
      compaction, material evaluation, contact assembly, solve, and rendering.
- [ ] Add a composite parity suite covering every capability below.

### Required composite scenes

- [ ] Stable Neo-Hookean soft and stiff falling solids.
- [ ] A high-speed multi-body IPC impact.
- [ ] A self-colliding deformable solid.
- [ ] A frictional card house or equivalent stack.
- [ ] An interactively manipulated stable Neo-Hookean body.
- [ ] Cloth covering an obstacle with cloth-solid and self-contact.

### Mandatory exit criteria

- [ ] P7-EC-01: No two subproblems within a Gauss-Seidel color share a documented
      conflict.
- [ ] P7-EC-02: Gauss-Seidel group bases satisfy their generalized equilibrium oracle
      within CPU relative error <= 1e-8.
- [ ] P7-EC-03: Jacobi and Gauss-Seidel converge to configurations within 1e-3 scene
      scale and final relative energies within 1e-4.
- [ ] P7-EC-04: On every solver-comparison scene in the checked-in matrix,
      Jacobi and Gauss-Seidel iteration counts differ by no more than 20%.
- [ ] P7-EC-05: Using the checked-in convergence protocol, the median local
      convergence order over the declared fit window is >= 1.6 while
      normalized error remains at least 100 times the measured noise floor.
- [ ] P7-EC-06: At every equal iteration index declared by that protocol, JGS2
      residual is at least 10 times lower than the frozen plain local-Jacobi or
      VBD-style baseline.
- [ ] P7-EC-07: Sparse precomputation artifacts are deterministic, versioned, and reject
      incompatible mesh, material, timestep, or schedule fingerprints.
- [ ] P7-EC-08: Production preprocessing uses one reusable sparse factorization
      for the full-coordinate basis solves and production loading requires no
      dense DOF-by-DOF matrix.
- [ ] P7-EC-09: Every required cell in the checked-in solver/capability matrix
      passes its physical, convergence, collision, and screenshot assertions
      with no not-applicable result.
- [ ] P7-EC-10: At least one Gauss-Seidel penalty-contact scene and one
      Gauss-Seidel IPC scene pass with generalized group contact-aware bases;
      neither may fall back entirely to Jacobi.
- [ ] P7-EC-11: Non-contact 20,000-30,000-tetrahedron solids meet p95 <= 16.7 ms.
- [ ] P7-EC-12: Contact-heavy parity scenes meet p95 <= 33.3 ms at their recorded
      representative sizes.
- [ ] P7-EC-13: The runtime keeps at most two GPU submissions in flight and shows no
      unbounded queue growth over 600 measured frames.
- [ ] P7-EC-14: Peak GPU memory for the composite suite is < 512 MB unless an approved
      target revision documents a hardware reason.
- [ ] P7-EC-15: There are zero WebGPU validation errors, device losses, candidate-buffer
      overflows, NaNs, infinities, or independent-checker intersections.
- [ ] P7-EC-16: The CPU/GPU/WASM architecture decision and final capability limitations
      are documented.
- [ ] P7-EC-17: Every item in the final capability checklist is complete.

### Required evidence

| Evidence | Location or result | Complete |
| --- | --- | --- |
| Coloring and generalized-basis report | TBD | [ ] |
| Jacobi-versus-GS convergence report | TBD | [ ] |
| Baseline convergence comparison | TBD | [ ] |
| Sparse artifact reproducibility report | TBD | [ ] |
| Composite E2E artifacts | TBD | [ ] |
| GPU timing and queue report | TBD | [ ] |
| Memory report | TBD | [ ] |
| Architecture decision | TBD | [ ] |
| Final reviewer approval | TBD | [ ] |

---

## Final capability checklist

The project may claim paper capability parity only when all items below are
complete.

- [ ] Stable Neo-Hookean tetrahedral solids.
- [ ] Distinct soft and stiff material parameters across bodies.
- [ ] Arbitrary external forces and kinematic manipulation.
- [ ] Current nonlinear JGS2 gradients and Hessians.
- [ ] Co-rotated equilibrium bases and validated Cubature.
- [ ] Full-coordinate basis precomputation from equations 19-23.
- [ ] Convergence-driven Jacobi.
- [ ] Graph-colored Gauss-Seidel.
- [ ] General penalty contact.
- [ ] IPC barriers and collision-free line search.
- [ ] Continuous collision detection.
- [ ] Deformable-deformable collision.
- [ ] Deformable self-collision.
- [ ] Rigid or kinematic obstacle collision.
- [ ] IPC-compatible Coulomb friction.
- [ ] StVK cloth membrane energy.
- [ ] Quadratic cloth bending.
- [ ] Cloth-solid and cloth self-collision.
- [ ] Moderate parity scenes meeting the recorded real-time targets.
- [ ] Deterministic unit, oracle, integration, and screenshot evidence.
- [ ] Documented CPU/GPU/WASM and preprocessing architecture.

## Criterion evidence log

Add a row for every gate-relevant run. Do not replace failed results; append the
later passing run so the history remains visible.

| Criterion ID | Date | Phase | Commit | Command or test | Hardware | Result | Artifact or log | Reviewer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0-EC-01 | 2026-07-16 | 0 | 1453c48 | npm.cmd run test:unit -- src/simulation/cpu/oracle.test.ts | i7-13700K | Pass: relative gradient error 4.406e-11 | src/simulation/cpu/oracle.test.ts | Automated oracle; human TBD |
| P0-EC-02 | 2026-07-16 | 0 | 1453c48 | npm.cmd run test:unit -- src/simulation/cpu/oracle.test.ts | i7-13700K | Pass: relative Hessian error 4.207e-12 | src/simulation/cpu/oracle.test.ts | Automated oracle; human TBD |
| P0-EC-05 | 2026-07-16 | 0 | 1453c48 | npm.cmd run test:unit -- src/simulation/cpu/oracle.test.ts | i7-13700K | Pass: maximum direct/full basis error 3.169e-16 | src/simulation/cpu/oracle.test.ts | Automated oracle; human TBD |
| P0-EC-06 | 2026-07-16 | 0 | 1453c48 | npm.cmd run test:e2e -- --grep "parity mode" | RTX 5090 / Chrome 150 | Pass: deterministic frame, exact even iteration count, and GPU wait | tests/e2e/jgs2.spec.ts | Automated hardware E2E; human TBD |
| P0-EC-09 | 2026-07-16 | 0 | 1453c48 | npm.cmd run test:e2e -- --grep "parity mode" | RTX 5090 / Chrome 150 | Pass: damping 1, contact damping 0, COM dispatch disabled | tests/e2e/jgs2.spec.ts | Automated hardware E2E; human TBD |
| P0-EC-12 | 2026-07-16 | 0 | 1453c48 | unit, build, and complete Playwright suite | i7-13700K / RTX 5090 | Pass: 47 unit and 8 E2E tests; zero skips/errors | docs/reproducibility.md | Automated suites; human TBD |
| P0-EC-01 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:unit -- src/simulation/cpu/phase0-canonical.test.ts | i7-13700K | Pass: all 64 poses; worst relative gradient error 1.221e-10 | src/simulation/cpu/phase0-canonical.test.ts | Automated oracle; independent gate audit |
| P0-EC-02 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:unit -- src/simulation/cpu/phase0-canonical.test.ts | i7-13700K | Pass: all 64 poses; worst relative Hessian error 4.389e-12 | src/simulation/cpu/phase0-canonical.test.ts | Automated oracle; independent gate audit |
| P0-EC-03 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e -- tests/e2e/canonical-gpu-oracle.spec.ts tests/e2e/gpu-oracle.spec.ts | RTX 5090 / Chrome 150 | Pass: worst canonical energy 1.659e-6, gradient 1.847e-6, Hessian 3.173e-8; floor-active errors also below 1e-3 | tests/e2e/canonical-gpu-oracle.spec.ts; tests/e2e/gpu-oracle.spec.ts | Automated CPU/GPU oracle; independent gate audit |
| P0-EC-04 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:unit -- src/simulation/cpu/phase0-canonical.test.ts; npm.cmd run test:e2e -- tests/e2e/equilibrium-oracle.spec.ts | i7-13700K / RTX 5090 | Pass: CPU worst 1.251e-17; GPU worst 5.469e-9 over 64 poses and three active blocks; no guarded pivot | src/simulation/cpu/phase0-canonical.test.ts; tests/e2e/equilibrium-oracle.spec.ts | Automated equilibrium oracle; independent gate audit |
| P0-EC-05 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:unit -- src/simulation/cpu/phase0-canonical.test.ts | i7-13700K | Pass: worst direct/full basis error 4.437e-16 | src/simulation/cpu/phase0-canonical.test.ts | Automated basis oracle; independent gate audit |
| P0-EC-06 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e -- --grep "parity mode" | RTX 5090 / Chrome 150 | Pass: pause, exact frame, exact nonlinear iteration, and GPU completion controls | tests/e2e/jgs2.spec.ts | Automated hardware E2E; independent gate audit |
| P0-EC-07 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:unit -- src/simulation/gpu/jgs2-diagnostics.test.ts src/simulation/diagnostics.test.ts; npm.cmd run test:e2e | i7-13700K / RTX 5090 | Pass: valid diagnostics finite; empty floor/contact and unavailable candidate-overflow diagnostics use finite zero with valid=false | src/simulation/gpu/jgs2-diagnostics.test.ts; tests/e2e/jgs2.spec.ts | Automated diagnostic gates; independent gate audit |
| P0-EC-08 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e | RTX 5090 / Chrome 150 | Pass: readbacks 0 -> 2 -> 2 -> 4 only at explicit checkpoints; at most two GPU submissions outstanding | tests/e2e/performance-baseline.spec.ts; tests/e2e/jgs2.spec.ts | Automated hardware instrumentation; independent gate audit |
| P0-EC-09 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e -- --grep "parity mode" | RTX 5090 / Chrome 150 | Pass: velocity damping 1; grounded tangential damping 0; horizontal COM restoration disabled | tests/e2e/jgs2.spec.ts | Automated hardware E2E; independent gate audit |
| P0-EC-10 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e -- --grep "force-free|frozen force-free" | RTX 5090 / Chrome 150 | Pass: base relative linear error 2.236119e-5; 32-state corpus worst 8.839392e-5 over 1,200 frames | tests/e2e/jgs2.spec.ts | Automated conservation corpus; independent gate audit |
| P0-EC-11 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e -- --grep "force-free|frozen force-free" | RTX 5090 / Chrome 150 | Pass: base relative angular error 6.200642e-4; 32-state corpus worst 3.312243e-3 over 1,200 frames | tests/e2e/jgs2.spec.ts | Automated conservation corpus; independent gate audit |
| P0-EC-12 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e | i7-13700K / RTX 5090 | Pass: 74 unit, build, frozen 26/74 unit and 7/14 E2E selectors, 14/14 hardware E2E; zero skips/errors | docs/reproducibility.md | Complete exact-current-code regression; independent gate audit |
| P0-EC-13 | 2026-07-16 | 0 | Phase 0 completion milestone | npm.cmd run test:e2e -- tests/e2e/performance-baseline.spec.ts | i7-13700K / RTX 5090 | Pass: mean 3.832 ms, p95 5.000 ms wall; GPU simulation timestamp 1.376256 ms; finite frame 721 | docs/evidence/phase0-performance-baseline.md | Automated performance evidence; independent gate audit |
| P0-EC-13 | 2026-07-16 | 0 | E2E performance metrics milestone | npm.cmd run test:e2e -- tests/e2e/performance-baseline.spec.ts | i7-13700K / RTX 5090 / Chrome 150 | Pass: serialized 269.4 FPS average, 173.9 FPS 1% low; wall mean 3.712 ms/p95 4.500 ms; CPU 0.132 ms/frame and 0.083 ms/step; GPU 1.450 ms/frame, 1.769 ms p95, and 1.432 ms/step; one timestamp map; byte-identical replay; wall mean and GPU p95 <= 8.333 ms necessary compute budget | docs/evidence/phase0-performance-baseline.md | Automated standard metrics; wall p95 diagnostic; production scheduler/time rate explicitly out of scope; independent audit findings resolved |
| P1-EC-01 | 2026-07-16 | 1 | Phase 1 CPU material milestone | npm.cmd run test:unit -- src/simulation/cpu/stable-neo-hookean.test.ts | i7-13700K | Partial pass: CPU rest-normalized energy and force are below 1e-10; GPU pending | src/simulation/cpu/stable-neo-hookean.test.ts | Automated CPU oracle; reviewer pending |
| P1-EC-02 | 2026-07-16 | 1 | Phase 1 CPU material milestone | npm.cmd run test:unit -- src/simulation/cpu/stable-neo-hookean.test.ts | i7-13700K | Partial pass: CPU energy, stress, tangent action, and tetrahedral response are objective; GPU pending | src/simulation/cpu/stable-neo-hookean.test.ts | Automated CPU oracle; reviewer pending |
| P1-EC-03 | 2026-07-16 | 1 | Phase 1 CPU material milestone | .\\node_modules\\.bin\\vitest.cmd run src/simulation/cpu/stable-neo-hookean.test.ts --disableConsoleIntercept --reporter=verbose | i7-13700K | Pass: 60 deformed poses; worst gradient 8.975e-10 and Hessian 1.957e-11 | src/simulation/cpu/stable-neo-hookean.test.ts | Automated derivative oracle; reviewer pending |
| P1-EC-05 | 2026-07-16 | 1 | Phase 1 CPU material milestone | npm.cmd run test:unit -- src/simulation/cpu/stable-neo-hookean.test.ts | i7-13700K | Partial pass: J=0.5, 0.1, and 0.01 finite; raw J=0/-0.1/-1 finite; separate accepted-step guard rejects J<=floor; GPU pending | src/simulation/cpu/stable-neo-hookean.test.ts; docs/phase1-stable-neo-hookean.md | Automated material/feasibility tests; reviewer pending |
| P1-EC-01 | 2026-07-16 | 1 | Phase 1 GPU material/frame milestone | npm.cmd run test:e2e -- tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | RTX 5090 / Chrome 150 | Pass: CPU rest energy/force below 1e-10; GPU zero-reference energy/force errors 0/0 | tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | Automated CPU/GPU oracle; independent gate audit |
| P1-EC-02 | 2026-07-16 | 1 | Phase 1 GPU material/frame milestone | npm.cmd run test:e2e -- tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | RTX 5090 / Chrome 150 | Pass: CPU objectivity below 1e-8; f32-exact GPU quarter-turn energy/force errors 0/0 using the roadmap default zero-reference metric | tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | Automated CPU/GPU oracle; independent gate audit findings resolved |
| P1-EC-04 | 2026-07-16 | 1 | Phase 1 GPU material/frame milestone | npm.cmd run test:e2e -- tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | RTX 5090 / Chrome 150 | Pass: all 64 frozen poses; material-only worst relative energy 2.234e-6, gradient 1.161e-6, local Hessian 1.089e-6; separate total implicit parity <=1e-3 | tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | Automated CPU/GPU oracle; independent gate audit findings resolved |
| P1-EC-05 | 2026-07-16 | 1 | Phase 1 GPU material/frame milestone | npm.cmd run test:e2e -- tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | RTX 5090 / Chrome 150 | Partial pass: GPU material and derivatives remain finite at J=0.5, 0.1, and 0.01; accepted assembled-step enforcement remains pending | tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | Automated positive-J corpus; independent gate audit; criterion intentionally unchecked |
| P1-FRAME-01 | 2026-07-16 | 1 | Phase 1 GPU material/frame milestone | npm.cmd run test:e2e -- tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | RTX 5090 / Chrome 150 | Pass: 0.01-scale non-affine polar(volume-weighted average F) CPU/GPU error 8.702e-8; stable runtime kernels with rest-equivalent diagnostic basis preprocessing remain finite at min J 0.9394; two explicit test-only readbacks | tests/e2e/stable-neo-hookean-gpu-oracle.spec.ts | Automated scale/frame/production smoke oracle; independent code-audit finding resolved |
| P1-GATE-02 | 2026-07-16 | 1 | Phase 1 GPU material/frame milestone | npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e | i7-13700K / RTX 5090 / Chrome 150 | Pass: 106 unit, build, frozen 26/106 unit and 7/16 E2E selectors, 16/16 hardware E2E; zero skips/errors; stress p95 5.000 ms | docs/reproducibility.md | Complete exact-current-code regression; independent code and gate audits |
| P1-EC-11 | 2026-07-16 | 1 | Phase 1 nonlinear Cubature milestone | npm.cmd run test:unit -- src/simulation/cpu/nonlinear-cubature-training.test.ts src/reproducibility/phase1-cubature-manifest.test.ts | i7-13700K | Pass: every active source has 12 Float64/f32-ranked candidates, retains K=6, and has conservative packed-vs-f64-target residual <=0.548989%; all 16 normalized targets valid | docs/evidence/phase1-nonlinear-cubature.md; manifests/phase1-cubature.v1.json | Automated independent dense/manifest gates; three-agent audit findings resolved |
| P1-EC-12 | 2026-07-16 | 1 | Phase 1 nonlinear Cubature milestone | npm.cmd run test:unit -- src/simulation/cpu/nonlinear-cubature-training.test.ts; npm.cmd run test:e2e -- tests/e2e/nonlinear-cubature-gpu.spec.ts | i7-13700K / RTX 5090 / Chrome 150 | Pass: packed selected-vs-independent-dense update RMS training 3.713075e-5, held-out 3.984377e-5, combined 3.796498e-5; two-iteration production f32 predictor/GPU update parity 2.113e-9 | docs/evidence/phase1-nonlinear-cubature.md | Automated CPU/GPU oracle; three-agent audit findings resolved |
| P1-GATE-03 | 2026-07-16 | 1 | Phase 1 nonlinear Cubature milestone | npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e | i7-13700K / RTX 5090 / Chrome 150 | Pass: 134 unit, build, frozen 26/134 unit and 7/17 E2E selectors, 17/17 hardware E2E; zero skips/errors; stress p95 4.9 ms | docs/reproducibility.md; docs/evidence/phase1-nonlinear-cubature.md | Complete exact-current-code regression; three-agent math/code/gate audits |
| P0-GATE-02 | 2026-07-16 | 0-1 | Standard performance observability milestone | npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e | i7-13700K / RTX 5090 / Chrome 150 | Pass: 149 unit, build, frozen 26/149 unit and 7/17 E2E selectors, 17/17 hardware E2E with zero skips; 414632.785 ms reported suite duration and 17 finite attempt durations; seven complete 600-frame scene reports; isolated stress 270.3 FPS average, 205.5 FPS 1% low, wall mean/p95 3.700/4.500 ms, CPU 0.146 ms/frame, GPU 1.462 ms/frame; necessary compute assessment passed | docs/reproducibility.md; docs/evidence/phase0-performance-baseline.md | Complete exact-current-code regression; three independent performance/schema/gate audits |
| P0-GATE-03 | 2026-07-16 | 0-1 | Production live-performance HUD milestone | npm.cmd run test:unit; npm.cmd run build; npm.cmd run test:baseline-manifest; npm.cmd run test:e2e | i7-13700K / RTX 5090 / Chrome 150 | Pass: 159 unit, build, frozen 26/159 unit and 7/18 E2E selectors, 18/18 hardware E2E in 252466.729 ms with zero retries/skips/errors; desktop/mobile live HUD, deterministic isolation, tab-resume epoch reset, strict timestamp samples; isolated stress 269.8 FPS average, 198.0 FPS 1% low, wall mean/p95 3.706/4.500 ms, CPU 0.132 ms/frame, GPU frame span 1.434 ms; necessary compute assessment passed | docs/reproducibility.md; docs/evidence/phase0-performance-baseline.md; tests/e2e/jgs2.spec.ts | Complete exact-current-code regression; three independent UI/GPU/gate audits; unused E2E HMR disabled after one reproduced local resource-pressure flake |

## Decision log

Use this table for threshold changes, algorithm choices, scope clarifications,
and accepted limitations.

| ID | Date | Phase | Decision | Reason and evidence | Downstream effect | Approver |
| --- | --- | --- | --- | --- | --- | --- |
| P0-D01 | 2026-07-16 | 0-1 | Split hardware E2E into an under-one-minute fast tier and a full qualification tier; use one-step breadth plus one-second sentinels for the quick 32-state corpus, in-place combined 0/12 smoke telemetry for screenshot tests, a combined 30/120 pass for the quick isolated baseline, and retain canonical 32-by-1,200 plus fresh 120/600 profiles in the full tier | The prior exhaustive corpus and repeated fresh scene profiles dominated feedback time. The selected quick workloads preserve GPU-path breadth, sentinel depth, per-scene observability, and settled-state timing while the full command preserves exhaustive and statistically meaningful evidence | Quick combined CPU timings include timestamp instrumentation when supported and short-sample tails are smoke-only; release, roadmap-exit, exhaustive-corpus, and formal performance evidence must use `test:e2e:full`; canonical manifests and historical evidence remain unchanged | Codex implementation agent; user-directed runtime target; independent review completed |
| P1-D01 | 2026-07-16 | 1 | Cap solve-only normalized local diagonal shift at 1e-3; target f32 eigenvalue floor is 256 times unit roundoff | Keeps the exact material oracle unmodified and turns excessive regularization into a visible failure; nonlinear corpus must leave fourfold headroom or trigger projection/Cubature diagnosis | GPU local solve must expose raw minimum eigenvalue, shift, normalized ratio, and descent; threshold changes require a new decision | Codex implementation agent; reviewer pending |
| P1-D02 | 2026-07-16 | 1 | Preserve raw stable Neo-Hookean behavior for all finite J, but require accepted runtime poses to satisfy J>1e-4 and revert an infeasible assembled Jacobi iteration | Matches the source material's collapse/inversion robustness while enforcing the roadmap's positive dynamic checkpoint policy | Line search and feasibility reductions must reject before acceptance; world-space clamping cannot substitute | Codex implementation agent; reviewer pending |
| P1-D03 | 2026-07-16 | 1 | Use a 4,800-tetrahedron non-contact solid at p95<=16.7 ms as the Phase 1 development gate; retain the final 20,000-30,000-tet Phase 7 target | Current dense browser preprocessing is intentionally a tiny-system oracle; the smaller gate validates the nonlinear hot loop without weakening final capability parity | Phase 1 manifest must freeze the exact interval; Phase 7 still requires scalable offline artifacts and the larger target | Codex implementation agent; reviewer pending |
| P1-D04 | 2026-07-16 | 1 | Keep Phase 1 Float64 oracle/preprocessing on CPU JavaScript and per-frame work on WebGPU; do not add WASM | WASM does not improve the GPU-resident loop and large in-browser sparse preprocessing is outside this phase | Reconsider only for arbitrary large user meshes; fixed large scenes should load offline artifacts | Codex implementation agent; reviewer pending |
| P1-D05 | 2026-07-16 | 1 | Define the current P1-EC-11 parity set as the versioned nonlinear Cubature fixture, require T>K and both Float64/f32 candidate rank >K, and fail stable preprocessing closed above 1% | The material-only single-tet oracle is not a sparse-Cubature fixture, while the 12-tet edge-constrained fixture has rank 12 and nonzero K=6 approximation error; future stable parity scenes cannot silently inherit this result | Every future stable parity scene must be added to the Cubature manifest or carry its own equivalent gate; unsuitable automatic preprocessing throws instead of shipping low-quality weights | Codex implementation agent; independent architecture and gate audits |

## Blocker log

| Opened | Phase | Blocker | Owner | Required resolution | Status | Closed |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Final sign-off

| Role | Name | Date | Approved |
| --- | --- | --- | --- |
| Roadmap owner | TBD | TBD | [ ] |
| Numerical-method reviewer | TBD | TBD | [ ] |
| Collision and friction reviewer | TBD | TBD | [ ] |
| GPU performance reviewer | TBD | TBD | [ ] |
| Final project approver | TBD | TBD | [ ] |
