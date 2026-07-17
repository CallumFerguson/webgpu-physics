# JGS2 implementation roadmap

This roadmap replaces the previous production-style plan. The goal is a small,
working implementation of the capabilities demonstrated in the JGS2 paper on
the author's current computer. It is not a general physics engine and does not
need the paper's largest scenes.

## Delivery rules

- Target: finish in one focused work session, ideally 8-10 hours and no more
  than an overnight run of roughly 12 hours.
- Treat the paper as the specification. Implement its equations directly and
  investigate alternatives only when a detail required by a demo is missing.
- Extend the existing solver in place. Do not rewrite working foundation code
  or introduce a second architecture.
- Timebox every item. If an item runs long, reduce scene size, use a simpler
  data structure, or narrow robustness to the included demos. Do not build new
  infrastructure to protect hypothetical future uses.
- Support only this machine, its current Chrome/WebGPU implementation, and the
  checked-in demo assets. Cross-browser, cross-GPU, mobile, and library-quality
  APIs are out of scope.
- Put per-frame solver work on the GPU. Keep one-time small-scene mesh setup,
  factorization, mode construction, Cubature fitting, and topology filtering
  on the CPU when that is quicker to implement.
- Do not add WebAssembly. The hot loop is already GPU-resident, and the small
  one-time CPU work does not justify another build and memory boundary.
- Prefer direct, readable implementations over abstractions. Duplication is
  acceptable when it is smaller than a reusable framework.
- Do not reproduce million-element benchmarks, build an asset pipeline, add an
  editor, or create a formal evidence archive.
- A paper capability is complete when it works in at least one deliberately
  small demo and its defining behavior is observable or numerically checked.
  It does not require exhaustive inputs or production robustness.

### Working-session checkpoints

| Elapsed time | Expected result |
| --- | --- |
| 0-1.5 hours | Nonlinear solid is visible and controllable in a public scene. |
| 1.5-4.5 hours | Small-scene IPC contact and friction work. |
| 4.5-6.5 hours | One cloth scene uses the contact path. |
| 6.5-8 hours | Colored Gauss-Seidel is selectable beside Jacobi. |
| 8-10 hours | Four concise demos, focused checks, build, and handoff are done. |

If progress is more than one hour behind a checkpoint, cut scene resolution,
UI polish, generality, and test breadth immediately. Keep the paper capability.

## Minimal testing policy

For each roadmap item:

1. Add at most one focused numeric/unit test when new math can be checked
   without a browser.
2. Add at most one focused browser test when GPU execution or visible behavior
   is the feature. Capture only the start and useful end state.
3. Run only the new or directly related test file while implementing. Re-run
   only the failing test after a fix.
4. Run `npm run build` once when the item is ready to commit.
5. Stop testing when the acceptance criteria below are demonstrated.

Do not repeatedly run all unit tests, `test:e2e`, or `test:e2e:full` during
feature work. The full E2E tier is not part of this roadmap. At the end, run
one build and the small set of new capability tests together; run the existing
fast E2E suite once only if shared solver changes make that useful.

## Existing foundation

The repository already contains the expensive foundation work:

- implicit tetrahedral elastodynamics and WebGPU rendering;
- the paper's full-coordinate JGS2 local perturbation basis;
- co-rotated local subspaces and nonnegative Cubature fitting;
- co-rotated linear and stable Neo-Hookean material calculations;
- CPU reference calculations and GPU nonlinear local solves;
- feasibility/Armijo globalization, convergence measurements, external forces,
  and soft target objectives; and
- four existing scenes, deterministic frame stepping, screenshots, and a live
  performance HUD.

The remaining work is to expose that foundation in complete demos and add the
paper capabilities that are genuinely absent: mesh contact/IPC, friction,
cloth, and Gauss-Seidel scheduling.

## Item 1 - Finish the nonlinear-solid demo path

**Timebox:** 1-1.5 hours

**Implementation**

- Switch at least the minimal and impact scenes to stable Neo-Hookean JGS2.
- Show the active material and iteration/convergence state in the HUD.
- Stop issuing remaining nonlinear iterations after the existing convergence
  flag is reached when practical; a CPU decision between frames is sufficient.
- Add one scripted moving soft target so the force/target objective is visible.

**Must pass before item 2**

- [x] Minimal scene visibly deforms under gravity without inversion or NaNs.
- [x] A scripted target pulls and releases a body without teleporting it or
      erasing its velocity.
- [x] HUD identifies stable Neo-Hookean and reports finite convergence values.
- [x] One focused browser test reaches the useful end state and saves the two
      screenshots.

**Work record**

- Status: Completed
- Started: 2026-07-17
- Completed: 2026-07-17
- Commit: `feat: complete nonlinear solid demo`
- Tests/commands run: focused Vitest (67 tests); focused hardware Playwright
  checks (4 tests); `npm run build`; `npm run test:e2e` (28 tests).
- Actual time: About 2.5 hours including regression hardening.
- Notes/blockers: Minimal and drop now use stable Neo-Hookean material. The
  scripted soft target has an inactive gravity phase, deterministic pull/hold,
  and state-preserving release. Production stable solves latch convergence on
  the GPU; exact APIs retain their full requested iteration history. The
  objective-free drop also uses the existing mass-COM correction, a common x/z
  translation that preserves deformation gradients and prevents finite JGS
  momentum drift; active force/target solves disable that correction.

## Item 2 - Small-scene IPC contact and friction

**Timebox:** 2.5-3 hours

**Implementation**

- Add vertex-triangle and edge-edge mesh contact for the included small scenes.
- Enumerate all topology-filtered vertex-triangle and edge-edge pairs once on
  the CPU for the deliberately small demos. Test those pairs on the GPU every
  step; do not add synchronous position readback, a dynamic broad phase, or a
  GPU BVH.
- Evaluate the IPC barrier gradient/Hessian contributions in the solver and
  cap the step with a collision-safe line search.
- Add the paper's lagged/dissipative friction model in its simplest useful form.
- Keep the existing plane penalty only as a debug comparison, not as the IPC
  capability demonstration.

**Must pass before item 3**

- [x] Two deformable bodies collide and settle without visible interpenetration.
- [x] A self-contact example does not pass through itself in its scripted case.
- [x] With friction enabled, a body on the level floor loses tangential speed;
      with friction disabled, the same body travels visibly farther.
- [x] One focused numeric test checks barrier value/gradient and one browser
      test covers the short collision/friction scene.

**Work record**

- Status: Completed
- Started: 2026-07-17
- Completed: 2026-07-17
- Commit: `feat: add small-scene IPC contact and friction`
- Tests/commands run: focused Vitest (102 tests); focused hardware Playwright
  contact demo (1 test); existing hardware GPU oracle (1 test); `npm run build`.
- Actual time: About 3 hours including hardware review and ABI hardening.
- Notes/blockers: The public contact scene disables the analytic plane penalty
  and uses CPU-enumerated topology-filtered VT/EE candidates, GPU barrier and
  lagged-friction terms, conservative safe-step caps, and post-scale f32
  feasibility validation. The small same-body pair exercises self-contact.
  Contact records append to the existing dynamic buffer, preserving the frozen
  Phase 1 seven-storage-buffer and 176-byte uniform ABI.

## Item 3 - Cloth through the same JGS2 solver

**Timebox:** 1.5-2 hours

**Implementation**

- Add a small triangle-cloth element using StVK membrane energy and a simple
  quadratic dihedral or edge bending energy, matching the paper's capability.
- Build its local JGS2 data with the existing basis/Cubature pipeline where
  possible; a cloth-specific compact path is acceptable if it is faster.
- Reuse item 2 contact, IPC, and friction.
- Add one small drape/fold scene with pinned vertices.

**Must pass before item 4**

- [ ] Pinned cloth falls, stretches, and bends while pinned vertices remain fixed.
- [ ] The included drape/fold does not pass through the collider or itself.
- [ ] No NaNs or inverted/degenerate triangles occur in the scripted run.
- [ ] One focused cloth energy/gradient test and one short browser test pass.

**Work record**

- Status: Not started
- Started:
- Completed:
- Commit:
- Tests/commands run:
- Actual time:
- Notes/blockers:

## Item 4 - Parallel Gauss-Seidel option

**Timebox:** 1-1.5 hours

**Implementation**

- Greedily color the small demo meshes on the CPU.
- Dispatch one color at a time while keeping vertices within a color parallel.
- Keep Jacobi selectable so the paper's JGS2 and GGS2 variants can be compared.
- Do not implement a general dynamic scheduler or optimize coloring.

**Must pass before item 5**

- [ ] Jacobi and colored Gauss-Seidel both run the same nonlinear-solid scene.
- [ ] Both remain finite and reach comparable final energy/configuration.
- [ ] The HUD reports the selected schedule and iteration count.
- [ ] One focused browser comparison test is sufficient.

**Work record**

- Status: Not started
- Started:
- Completed:
- Commit:
- Tests/commands run:
- Actual time:
- Notes/blockers:

## Item 5 - Final capability demos and handoff

**Timebox:** 1-1.5 hours

**Implementation**

Keep a small set of scenes that collectively demonstrate every capability:

1. minimal stable-Neo-Hookean cantilever with a scripted target;
2. deformable impact/contact with friction;
3. cloth drape/fold with contact; and
4. a moderate mixed stress scene with a Jacobi/Gauss-Seidel selector.

Remove or hide redundant scenes rather than maintaining several variations.
Keep the existing performance HUD. Real-time means the demos respond
interactively on this machine; no formal universal FPS promise is required.

**Definition of done**

- [ ] Each paper capability listed below is exercised by at least one scene.
- [ ] Every scene starts automatically and has an unambiguous visual result.
- [ ] The HUD reports FPS, 1% low, CPU submit/step time, GPU step/render time
      when timestamp queries are available, solver type, and iterations.
- [ ] One integrated browser smoke test visits the four scenes, takes start/end
      screenshots, and checks only finiteness plus the scene's main landmark.
- [ ] `npm run build` and the focused capability tests pass together.
- [ ] README run instructions and capability limitations are accurate.

**Work record**

- Status: Not started
- Started:
- Completed:
- Commit:
- Tests/commands run:
- Actual time:
- Notes/blockers:

## Paper capability checklist

The project is complete when these capabilities, rather than the paper's scene
sizes, are present:

- [x] variational implicit elastodynamics;
- [x] JGS2 local perturbation subspaces and optimal full-coordinate bases;
- [x] co-rotated subspaces;
- [x] nonnegative Cubature sampling and full-coordinate precomputation;
- [x] stable Neo-Hookean tetrahedral solids;
- [x] external forces and target objectives;
- [x] collision candidates, IPC barrier terms, and collision-safe line search;
- [x] frictional contact;
- [ ] triangle cloth with stretching and bending;
- [ ] parallel colored Gauss-Seidel as well as Jacobi; and
- [ ] small integrated scenes demonstrating all of the above at interactive
      speed on the target computer.

Large-scale preprocessing, millions of elements, every demo in the paper, and
publication-quality benchmark parity are explicitly not completion criteria.
