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

## Validation

`src/reproducibility/manifests.test.ts` parses both JSON files through the
runtime-independent validators in `src/reproducibility/manifests.ts`. The tests
reject duplicate or malformed IDs, count drift, unknown skip/corpus references,
missing generator metadata, non-finite physical values, incompatible Cubature
settings, unsorted checkpoints, and sampled times that do not equal
`frame * timestep`. They also compare every baseline fixture's material,
timestep, solver, and camera values directly with the current scene builders.

Run the manifest and artifact checks alone with:

```powershell
npm.cmd run test:unit -- src/reproducibility
```

The Phase 0 regression gate still runs every command recorded by the baseline:

```powershell
npm.cmd run test:unit
npm.cmd run build
npm.cmd run test:screenshot
```

The reported baseline selectors must all be present. Any skipped baseline test
must have its stable ID in `allowedSkips`, and the skip condition must match.
Because the v1 manifest contains no allowed skips, any baseline skip is an
unexpected failure. Hardware-browser output and the exact commands belong in
the roadmap criterion evidence log; the manifests do not turn a prior run into
evidence for a later commit.

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
