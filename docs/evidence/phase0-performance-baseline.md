# Phase 0 performance baseline

This is the checked-in evidence summary for roadmap criterion `P0-EC-13`.
The reproducible Playwright test also attaches a versioned JSON report with all
600 raw samples to its test result.

## Recorded run

| Field | Value |
| --- | --- |
| Recorded UTC | 2026-07-16T21:08:33.536Z |
| Command | `npm.cmd run test:screenshot -- tests/e2e/performance-baseline.spec.ts` |
| Playwright project | `chrome-hardware` |
| Browser | Google Chrome 150.0.7871.124 |
| GPU | NVIDIA GeForce RTX 5090, Blackwell, driver 596.21 |
| CPU | Intel Core i7-13700K, 24 logical processors |
| OS | Windows 11 Home x64 |
| Software fallback | Rejected; hardware adapter confirmed |

## Workload and method

- Scene: `stress`.
- 72 vertices, 72 tetrahedra, 120 surface triangles, 180 surface
  edges, 6 bodies, and 2 materials.
- Timestep: `1/120` seconds.
- 17 JGS2 iterations and 6 Cubature samples per vertex.
- 120 warm-up frames followed by 600 measured frames.
- Each wall-clock sample covers one simulation frame, one render, and GPU queue
  completion through `await window.__jgs2Test.stepFrames(1)`.
- Preprocessing, explicit diagnostics, the GPU timestamp probe, screenshots,
  and Playwright tracing are outside the measured interval.
- Percentiles use nearest rank.

## Results

| Metric | Result |
| --- | ---: |
| Minimum wall frame | 2.100 ms |
| Mean wall frame | 3.832 ms |
| p50 wall frame | 3.700 ms |
| p95 wall frame | 5.000 ms |
| p99 wall frame | 6.500 ms |
| Maximum wall frame | 14.800 ms |
| Explicit GPU simulation timestamp | 1.376256 ms |

The adapter advertised and enabled `timestamp-query`. The GPU value brackets
the simulation command stream with empty timestamped compute passes; it omits
rendering and readback, so it is reported separately from end-to-end wall time.

The run ended at frame 721 with finite state, no browser errors, and no fallback
adapter. Explicit diagnostic-readback counts were `0 -> 2 -> 2 -> 4`: the count
did not change during the timestamp probe, warm-up, measurement, simulation, or
rendering, and changed only for the two intentional diagnostic checkpoints.

## Interpretation

This is a Phase 0 instrumentation baseline, not the final paper-capability
performance gate. The workload is intentionally small and has no general
body-body collision pipeline. Later phase reports replace it with their
canonical material, collision, IPC, friction, and cloth workloads and apply the
roadmap's recorded p95 targets.
