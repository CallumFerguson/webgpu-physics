# Reproducibility manifests and precomputation artifacts

Phase 0 uses two checked-in JSON manifests:

- [`manifests/baseline-tests.v1.json`](../manifests/baseline-tests.v1.json)
  freezes the pre-roadmap suite as 26 unit tests and 7 hardware-WebGPU E2E
  tests. Each test has a stable ID, repository-relative source, and human test
  selector. `allowedSkips` is intentionally empty.
- [`manifests/canonical-scenes.v1.json`](../manifests/canonical-scenes.v1.json)
  records the generator and version, seed, material values, timestep, solver
  settings, exact sampled frames, camera, and expected metrics for each
  baseline scene. It also defines the Phase 0 force-free conservation fixture,
  the exact single-tetrahedron oracle, and their deterministic corpora.

The baseline describes the suite that existed before roadmap work started.
New tests are required to pass as part of the complete current suite, but they
are not added retroactively to the frozen baseline. This distinction lets a
later gate require both the unchanged regression baseline and all newer tests.

Phase 1 adds a separate
[`manifests/phase1-scenes.v1.json`](../manifests/phase1-scenes.v1.json). It does
not broaden or rewrite either frozen Phase 0 schema. The manifest records the
stable Neo-Hookean single-tetrahedron fixture, corrected material parameters,
timestep, deterministic seed and generator version, 64-pose count, CPU/GPU
tolerances, and exact first/last corpus anchors. The executable corpus contains
rest, three rigid rotations, determinants `0.5`, `0.1`, and `0.01`, shear,
stretch, and 55 seeded positive-determinant states.

Phase 1 nonlinear Cubature has its own additive
[`manifests/phase1-cubature.v1.json`](../manifests/phase1-cubature.v1.json).
It freezes the private 12-tetrahedron full-rank fixture, stable material and timestep,
rest-Hessian mode settings, nonlinear training and held-out corpus protocol,
determinant floor, packed-f32 residual/update tolerances, exact pose anchors,
and the expected selected tetrahedra and packed weights for every active
source. This small fixture establishes algorithm and CPU/GPU data-path parity;
it is not a large-scene performance artifact.

## Validation

`src/reproducibility/manifests.test.ts` parses both JSON files through the
runtime-independent validators in `src/reproducibility/manifests.ts`. The tests
reject duplicate or malformed IDs, count drift, unknown skip/corpus references,
missing generator metadata, non-finite physical values, incompatible Cubature
settings, unsorted checkpoints, and sampled times that do not equal
`frame * timestep`. They also compare every baseline fixture's material,
timestep, solver, and camera values directly with the current scene builders.

`src/reproducibility/phase1-manifest.test.ts` independently validates the
Phase 1 schema and binds its executable mesh, material, timestep, gravity,
floor, packed Cubature width, and corpus anchors to `src/scenes/phase1.ts`.
Diagnostic-only settings such as exact-all evaluation, zero floor stiffness,
and parity mode are asserted directly. Material ABI tests also require the
stable material tag and converted Lamé constants to survive CPU-to-GPU packing.

`src/reproducibility/phase1-cubature-manifest.test.ts` validates the nonlinear
Cubature schema and binds it to the executable fixture, corpus constants and
anchors, thresholds, and packed preprocessing output. It rejects malformed
thresholds, pose anchors, and non-f32 expected weights. The numerical tests
then independently enforce the all-element dense projection identity,
nonnegative selection, packed training residual, held-out unregularized update
RMS, and production WebGPU update parity.

`scripts/verify-baseline-tests.mjs` independently asks Vitest and Playwright to
list the tests they actually collect, then matches every frozen source and
selector. This prevents a structurally valid manifest from concealing a
renamed or deleted baseline test.

Run the manifest and artifact checks alone with:

```powershell
npm.cmd run test:unit -- src/reproducibility
npm.cmd run test:baseline-manifest
```

The Phase 0 regression gate still runs every command recorded by the baseline:

```powershell
npm.cmd run test:unit
npm.cmd run test:baseline-manifest
npm.cmd run build
npm.cmd run test:e2e
```

The list reporter logs the finalized duration of every non-skipped E2E
attempt. The built-in JSON reporter preserves per-attempt status, start time,
duration, retry, errors, and whole-run duration in
`test-results/playwright-results.json`, including failures and skips. Duration
values are milliseconds. Discovery-only `--list` runs use only the list
reporter, so they do not overwrite the last executed test report.

Each of the seven screenshot-producing scene tests additionally attaches
`scene-performance.json`. It profiles two fresh deterministic replays: 120
warm-up and 600 measured serialized frames for wall/CPU submission timing,
then the same frames with GPU simulation/render timestamps. The GPU replay
resolves 2,400 timestamps with one map and must end with byte-identical f32
positions and velocities. Average FPS is reciprocal mean wall time; 1% low is
the reciprocal mean of the slowest `ceil(1%)` wall samples. CPU values are
explicitly submission time, not wall time minus GPU time. `timestamp-query` is
optional: unavailable devices still record wall/CPU metrics and explicitly
report GPU metrics as unavailable rather than failing screenshot correctness.
The scene tests disable Playwright tracing so trace collection cannot enter the
timing distribution. Their per-scene budget assessments are logged and stored
but are informational: shared desktop load must not turn screenshot or physics
correctness into a performance failure. The isolated
`performance-baseline.spec.ts` workload owns the enforced hardware compute
budget.

FPS in these artifacts is serialized benchmark throughput: one simulation
step, one render, and a queue drain per measured sample. The necessary compute
gate requires serialized wall mean and, when timestamp-query is available, GPU
frame p95 to fit `min(16.667 ms, one timestep)`. Serialized wall p95 and 1% low
remain reported diagnostics rather than hard gates because each sample includes
an artificial queue drain and browser/event-loop synchronization jitter. This
does not measure the production `requestAnimationFrame` scheduler or prove that
simulation time advances at wall-clock speed. The current production loop
submits at most one simulation step per animation callback, so a `1/120`-second
scene on a 60 Hz display needs a future catch-up/time-rate gate before it can be
called real-time in the application.

The reported baseline selectors must all be present. Any skipped baseline test
must have its stable ID in `allowedSkips`, and the skip condition must match.
Because the v1 manifest contains no allowed skips, any baseline skip is an
unexpected failure. Hardware-browser output and the exact commands belong in
the roadmap criterion evidence log; the manifests do not turn a prior run into
evidence for a later commit.

The canonical Phase 0 corpora are executable rather than descriptive only.
`src/simulation/cpu/phase0-canonical.test.ts` evaluates all 64 frozen
single-tetrahedron poses. The P0-EC-03 hardware oracle evaluates ten
representative canonical poses plus a floor-active state. The P0-EC-04 GPU
equilibrium oracle evaluates all 64 poses and three active vertex blocks. The
conservation suite runs the frozen 2-by-2-by-2 force-free fixture and all 32
seeded rigid-velocity states for 1,200 frames each. GPU oracle tests reuse the
same fixture/corpus IDs and generators.

The checked-in Phase 0 timing summary is
[`docs/evidence/phase0-performance-baseline.md`](evidence/phase0-performance-baseline.md).
Its Playwright test generates a versioned JSON attachment containing all raw
samples on every run; the repository retains a compact machine-readable
summary beside the report.
The refreshed stress run recorded 269.4 average FPS and 173.9 FPS 1% low:
mean 3.712 ms and p95 4.500 ms end-to-end wall frames, mean CPU submission
0.132 ms/frame and 0.083 ms/simulation step, and mean GPU timestamp durations
1.450 ms/frame and 1.432 ms/simulation step. Its 3.712 ms serialized wall mean
and 1.769 ms GPU-frame p95 fit the scene's 8.333 ms serialized
`1/120`-second step-compute budget; wall p95 remains a diagnostic at 4.500 ms.
This is a necessary throughput result, not a production real-time claim. These
numbers are Phase 0 instrumentation evidence, not the final large
nonlinear-material/contact performance gate.

The Phase 1 GPU material oracle evaluates all 64 stable Neo-Hookean poses
against the Float64 nonlinear CPU oracle. Material-only energy, gradient, and
local tangent blocks are compared without inertia masking; total implicit
objective parity is checked separately. Exact quarter turns use the roadmap's
default zero-reference metric. A second hardware test compares non-affine
CPU/GPU vertex polar frames on a `0.01`-scale mesh, checks a production stable
step for finite positive-determinant output, and requires explicit test-only
readbacks. These additive tests do not alter the frozen Phase 0 counts.
The completed GPU material/frame milestone gate passed 106 unit tests, the
production build, frozen selector verification at 26/106 unit and 7/16 E2E,
and all 16 hardware E2E tests with zero skips.

The nonlinear Cubature capability evidence is recorded in
[`docs/evidence/phase1-nonlinear-cubature.md`](evidence/phase1-nonlinear-cubature.md).
Its focused production WebGPU run covers six training/held-out shapes and uses
two explicit test-only readbacks per shape (predicted and solved positions);
normal simulation retains the no-readback GPU-resident contract.
The completion gate passed 134 unit tests, the production build, frozen
selector verification at 26/134 unit and 7/17 E2E, and all 17 hardware E2E
tests with zero skips. The corrected two-iteration Cubature oracle reported
maximum update error `2.113e-9` and predictor error `1.683e-9`; the unchanged
stress timing test reported `p95 = 4.9 ms`.

The later standard-performance-observability gate passed all 149 unit tests,
the production build, frozen selector verification at 26/149 unit and 7/17
E2E, and all 17 hardware E2E tests with zero skips. The JSON reporter recorded
17 finite per-attempt durations and a 414,632.785 ms whole-run duration. All
seven scene reports contained complete 600-frame CPU/GPU replay samples. The
final isolated stress baseline recorded 270.3 serialized average FPS, 205.5
FPS 1% low, 3.700 ms wall mean, 4.500 ms diagnostic wall p95, 0.146 ms CPU
submission per frame, and 1.462 ms GPU mean per frame; its sustained-wall and
GPU-p95 compute assessment passed.

## Precomputation artifacts

`src/reproducibility/precomputation-artifact.ts` defines schema version 1 for
offline precomputation metadata. Every artifact records separate SHA-256
fingerprints for:

- mesh topology;
- materials;
- timestep;
- basis construction settings;
- Cubature settings;
- solver schedule.

It also records the producer version and binary-payload format, length, and
fingerprint. `fingerprintCanonicalValue` hashes a canonical representation:
plain-object keys are sorted, all numbers must be finite, and typed-array kinds
are included. This makes fingerprints independent of object construction order
while distinguishing topology arrays from floating-point payloads.

Artifact loading should call `assertPrecomputationArtifactCompatible` before
decoding the payload. A schema/format mismatch is rejected immediately; input
mismatches name every stale component so the artifact can be regenerated once.
The fingerprints detect accidental or stale inputs but are not digital
signatures, so artifacts must still come from a trusted source.

The artifact module establishes the compatibility contract only. Wiring a
production offline preprocessor and binary loader is a later roadmap task.
