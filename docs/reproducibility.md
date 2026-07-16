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
npm.cmd run test:screenshot
```

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
The completion run recorded mean 3.832 ms and p95 5.000 ms end-to-end wall
frames plus a 1.376256 ms explicit GPU simulation timestamp. These numbers are
Phase 0 instrumentation evidence, not a performance target for later material
and contact capabilities. The completion gate passed 74 unit and 14 hardware
E2E tests with zero skips.

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
