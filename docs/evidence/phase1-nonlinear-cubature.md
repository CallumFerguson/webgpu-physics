# Phase 1 nonlinear Cubature evidence

This report records the capability gate for the stable Neo-Hookean
current-pose Cubature path from JGS2 Eqs. 14-18. It is evidence for the
nonlinear algorithm and packed production data flow, not for large-mesh
preprocessing scale.

## Frozen fixture and corpus

`manifests/phase1-cubature.v1.json` freezes a two-cell unit solid with 12
tetrahedra, ten active source vertices, one fixed edge, stable Neo-Hookean
material (`E = 5,000`, `nu = 0.3`), and `dt = 1/60`. Every source has 12
nonzero, numerically independent candidate columns in both Float64 and the
packed-f32 representation. The runtime retains exactly six, so this gate
measures a genuine approximation rather than duplicate-column elimination.

The training split contains both signs of eight low-frequency modes at
deformation-gradient-normalized amplitude `0.12`. The held-out split contains
eight deterministic mixtures of 12 modes, including four modes absent from
training: four poses use amplitude `0.06`, four use `0.18`, and inertial targets
are blended by `0.25`. The minimum measured determinant over training and
held-out poses is `0.8651572868`, above the frozen `0.5` floor. Unconstrained
corpus tests remove six rigid modes per connected component and add four
exactly rotated validation poses for a wholly free mesh.

## Mathematical checks

For every source and every training/held-out pose, the CPU evaluates current
stable Neo-Hookean element gradients and Hessians, distributed inertia, and
the current basis `B_vi = R_v Ubar_vi R_i^T`. Incident source blocks are
subtracted from complementary candidates. Source plus every candidate at unit
weight matches an independently assembled dense `B^T g` and `B^T H B` to a
relative tolerance of `1e-10`; the worst observed diagnostic was below
`1e-15`.

Eq. 18 normalizes each nontrivial complementary-gradient target and fits
nonnegative weights. All 16 training targets must be valid; no row may be
silently omitted. The conservative packed metric is frozen as

```text
||A_packed w_packed - b_f64|| / ||b_f64||.
```

It therefore includes target movement caused by basis packing instead of
recomputing a correlated packed target. Exact and selected 3-by-3 CPU systems
must be Cholesky-SPD, and the selected-update comparison uses an independent
dense Float64 reference with no regularization.

## Results

| Source | f64/f32 rank | Packed residual |
| ---: | ---: | ---: |
| 1 | 12 / 12 | 3.008680e-3 |
| 2 | 12 / 12 | 4.943350e-5 |
| 3 | 12 / 12 | 1.693528e-3 |
| 4 | 12 / 12 | 8.497507e-4 |
| 5 | 12 / 12 | 2.284955e-3 |
| 7 | 12 / 12 | 5.265101e-4 |
| 8 | 12 / 12 | 3.183878e-4 |
| 9 | 12 / 12 | 7.802684e-5 |
| 10 | 12 / 12 | 1.643806e-3 |
| 11 | 12 / 12 | 5.489887e-3 |

The maximum is `0.548989%`, below the `1%` P1-EC-11 threshold for every
active source after runtime packing.

| Data | Training RMS | Validation RMS | Combined RMS |
| --- | ---: | ---: | ---: |
| Float64 selection | 3.713077e-5 | 3.984374e-5 | 3.796498e-5 |
| Packed-f32 selection and bases | 3.713075e-5 | 3.984377e-5 | 3.796498e-5 |

All three aggregate values are below the `2%` P1-EC-12 threshold. The worst
packed individual system is `1.606601%` at held-out pose 07, source 11, and is
included rather than skipped.

The hardware test initializes velocities so the production f32 implicit-Euler
predictor reproduces each requested target. Three training-shaped cases target
their deformed pose; three held-out cases use the frozen blended corpus target.
It then mirrors two exact nonlinear Jacobi iterations on CPU and GPU, making
the second pass exercise a nonzero inertial gradient. The test reads predicted
and solved positions explicitly, keeps every solved determinant positive, and
reports maximum GPU-versus-packed-CPU update error `2.113e-9` against the
`1e-3` production parity tolerance.

## Commands

```powershell
npm.cmd run test:unit -- src/simulation/cpu/cubature.test.ts src/simulation/cpu/jgs2-local.test.ts src/simulation/cpu/nonlinear-cubature.test.ts src/simulation/cpu/nonlinear-cubature-training.test.ts src/reproducibility/phase1-cubature-manifest.test.ts
npm.cmd run test:e2e -- tests/e2e/nonlinear-cubature-gpu.spec.ts
```

Stable preprocessing fails closed if any packed source residual exceeds 1%.
A regression fixture verifies that an unsuitable 12-tet/K=6 configuration is
rejected rather than silently shipped. The milestone is accepted only after
the complete unit, build, frozen-baseline, and hardware-browser suites also
pass on the exact commit.

The final exact-tree gate passed `134/134` unit tests, the production build,
frozen selector verification at `26/134` unit and `7/17` E2E, and `17/17`
hardware E2E tests with zero skips.

## Architecture and remaining scale work

Float64 pose generation, dense tiny-system oracles, NNLS selection, and packing
run once on the CPU. Current deformation gradients, polar frames, stable
material derivatives, Cubature projection, and local updates run per frame on
WebGPU without production readback. WebAssembly is not used: it would not help
the GPU-resident loop. It should be reconsidered only if arbitrary large meshes
must be sparsely precomputed inside the browser; fixed large scenes should load
offline artifacts instead.
