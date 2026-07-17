# Phase 0 performance baseline

This is the checked-in evidence summary for roadmap criterion `P0-EC-13`.
The reproducible Playwright test attaches the complete versioned JSON report,
including all wall, CPU, and GPU samples, to its test result.
The compact machine-readable summary is
[`phase0-performance-baseline.v2.json`](phase0-performance-baseline.v2.json);
the original v1 single-probe evidence remains unchanged for history.

## Recorded run

| Field | Value |
| --- | --- |
| Recorded UTC | 2026-07-17T02:02:19.631Z |
| Command | `npm.cmd run test:e2e -- tests/e2e/performance-baseline.spec.ts` |
| Playwright project | `chrome-hardware` |
| Browser | Google Chrome 150.0.7871.124 |
| GPU | NVIDIA GeForce RTX 5090, Blackwell, driver 596.21 |
| CPU | Intel Core i7-13700K, 24 logical processors |
| OS | Windows 11 Home x64 |
| Software fallback | Rejected; hardware adapter confirmed |

## Workload and method

- Scene: `stress`: 72 vertices, 72 tetrahedra, 120 surface triangles, 180
  surface edges, 6 bodies, and 2 materials.
- Timestep: `1/120` seconds, with 17 JGS2 iterations and 6 Cubature samples
  per vertex.
- CPU/wall replay: a fresh deterministic scene, 120 warm-up frames, then 600
  serialized production-shaped simulation-and-render frames.
- GPU replay: another fresh deterministic scene over the same 720 frames,
  with distinct simulation and render timestamp intervals for all 600 measured
  frames. All 2,400 timestamps are resolved and mapped once after measurement.
- The replays end with byte-identical f32 position and velocity buffers.
- Diagnostics, timestamp resolve/map, screenshots, preprocessing, and
  Playwright tracing are outside timed intervals.
- Percentiles use nearest rank. Average FPS is `1000 / mean wall ms`; 1% low
  is `1000 / mean(slowest ceil(1%) wall samples)`.
- “CPU” means main-thread encode/submit time and excludes GPU execution or
  waiting. The values therefore do not add arithmetically to wall time.

## Results

| Metric | Mean | p95 |
| --- | ---: | ---: |
| End-to-end wall frame | 3.712 ms | 4.500 ms |
| CPU frame submission | 0.132 ms | 0.200 ms |
| CPU simulation-step submission | 0.083 ms | 0.200 ms |
| GPU frame | 1.450 ms | 1.769 ms |
| GPU simulation step | 1.432 ms | 1.769 ms |
| GPU render pass | 0.004 ms | 0.066 ms |

| Throughput metric | Result |
| --- | ---: |
| Serialized average FPS | 269.4 |
| Serialized 1% low FPS | 173.9 |
| Serialized step-compute budget | 8.333 ms/frame |
| Serialized wall-mean budget result | Pass (3.712 ms) |
| GPU-frame p95 budget result | Pass (1.769 ms) |

The wall distribution ranged from 2.600 ms to 8.500 ms, with p50 3.600 ms
and p99 4.600 ms. Both replays ended at frame 720 with finite state, no
browser errors, no diagnostic readback during measurement, and exactly one
timestamp-buffer map after the GPU interval.

The 2026-07-17 live-HUD compatibility gate reran this exact 120-warm-up/600-
measured-frame workload after timestamp writes were moved onto the real predict
and finalize passes. It recorded 269.8 serialized average FPS, 198.0 FPS 1%
low, 3.706 ms wall mean, 4.500 ms wall p95, 0.132 ms CPU submission per frame,
1.434 ms GPU frame span, 1.414 ms GPU simulation step, and 0.006 ms GPU render.
The one-map, byte-equivalent replay and both 8.333 ms necessary compute checks
passed as part of the retry-free 18/18 hardware E2E run.

## Interpretation

This remains a small Phase 0 instrumentation baseline rather than the final
paper-capability performance gate. Later phase reports replace it with their
canonical nonlinear material, collision, IPC, friction, and cloth workloads.
The pass is a necessary serialized compute-throughput condition only: sustained
wall mean and GPU-frame p95 fit the step budget. Serialized wall p95 and 1% low
remain diagnostics because each wall sample includes a queue drain and
browser/event-loop synchronization jitter. The result does not exercise
production `requestAnimationFrame` scheduling or establish that the
`1/120`-second stress scene advances simulation time at wall-clock speed on a
60 Hz display.
