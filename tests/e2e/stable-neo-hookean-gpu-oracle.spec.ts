import { expect, test } from "@playwright/test";

import { PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES } from "../../src/scenes/phase1";

const PHASE1_GPU_CASE_INDICES = Object.freeze(
  Array.from({ length: 64 }, (_unused, index) => index),
);
const GPU_PARITY_TOLERANCE =
  PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES.cpuParityRelativeError;
const GPU_REST_TOLERANCE =
  PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES.restEnergyAndForce;
const GPU_RIGID_TOLERANCE =
  PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES.rigidEnergyAndForce;

interface Phase1GpuOracleCase {
  readonly id: string;
  readonly index: number;
  readonly kind: string;
  readonly expectedDeterminant: number;
  readonly minimumDeterminant: number;
  readonly finite: boolean;
  readonly energyError: number;
  readonly gradientError: number;
  readonly localHessianError: number;
  readonly implicitEnergyError: number;
  readonly implicitGradientError: number;
  readonly implicitLocalHessianError: number;
  readonly zeroReferenceEnergyError: number;
  readonly zeroReferenceForceError: number;
  readonly error: string;
}

interface Phase1GpuOracleResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly fatalError: string;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly elapsedMilliseconds: number;
  readonly cases: readonly Phase1GpuOracleCase[];
}

interface Phase1GpuFrameResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly meshScale: number;
  readonly maximumRelativeError: number;
  readonly solvedPositionsFinite: boolean;
  readonly minimumSolvedDeterminant: number;
  readonly explicitReadbacks: number;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
}

test("P1-EC-01/02/04/05: exact stable Neo-Hookean GPU oracle matches Float64 CPU", async ({
  page,
}, testInfo) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(`Page error: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`Console error: ${message.text()}`);
    }
  });

  await page.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });
  const result = await page.evaluate<Phase1GpuOracleResult, readonly number[]>(
    async (caseIndices) => {
      const [cpuApi, scenes, phase1, gpuApi, diagnosticsApi] = await Promise.all([
        import("/src/simulation/cpu/index.ts"),
        import("/src/scenes/index.ts"),
        import("/src/scenes/phase1.ts"),
        import("/src/simulation/gpu/index.ts"),
        import("/src/simulation/gpu/jgs2-diagnostics.ts"),
      ]);
      const adapter = await navigator.gpu?.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        throw new Error("Phase 1 GPU parity requires a hardware WebGPU adapter.");
      }
      const adapterDescription = [
        adapter.info.description,
        adapter.info.vendor,
        adapter.info.architecture,
        adapter.info.device,
      ]
        .filter(Boolean)
        .join(" / ");
      const device = await adapter.requestDevice();
      const uncapturedErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        uncapturedErrors.push(event.error.message);
      });
      device.pushErrorScope("validation");

      const definition = phase1.buildPhase1StableNeoHookeanOracleDefinition();
      const corpus = phase1.generatePhase1StableNeoHookeanPoseCorpus();
      const precomputed = cpuApi.buildPrecomputedScene(definition);
      const baseInput = scenes.toJGS2GpuInput(precomputed);
      const offsets = gpuApi.computeJGS2DynamicOffsets(
        baseInput.vertexCount,
        baseInput.tetCount,
        1,
      );

      const mirrorRestPositions = new Float64Array(definition.mesh.positions.length);
      const mirrorMasses = new Float64Array(baseInput.vertexCount);
      for (let vertex = 0; vertex < baseInput.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          mirrorRestPositions[vertex * 3 + coordinate] =
            baseInput.vertexRest[vertex * 4 + coordinate]!;
        }
        mirrorMasses[vertex] = baseInput.vertexRest[vertex * 4 + 3]!;
      }
      const mirrorInverseRest = new Float64Array(baseInput.tetCount * 9);
      const mirrorVolumes = new Float64Array(baseInput.tetCount);
      for (let tetrahedron = 0; tetrahedron < baseInput.tetCount; tetrahedron += 1) {
        mirrorVolumes[tetrahedron] = baseInput.tetMeta[tetrahedron * 4]!;
        for (let row = 0; row < 3; row += 1) {
          for (let column = 0; column < 3; column += 1) {
            mirrorInverseRest[tetrahedron * 9 + row * 3 + column] =
              baseInput.tetInverseDm[tetrahedron * 12 + column * 4 + row]!;
          }
        }
      }
      const mirrorMesh = {
        ...definition.mesh,
        positions: mirrorRestPositions,
      };
      const mirrorRestData = {
        volumes: mirrorVolumes,
        inverseRestMatrices: mirrorInverseRest,
        stiffnessMatrices: Float64Array.from(baseInput.tetRestStiffness),
      };

      const buffers: GPUBuffer[] = [];
      let evaluator:
        | InstanceType<typeof diagnosticsApi.JGS2GpuOracleEvaluator>
        | undefined;
      const cases: Phase1GpuOracleCase[] = [];
      let fatalError = "";
      const started = performance.now();

      const createInitializedBuffer = (
        label: string,
        usage: GPUBufferUsageFlags,
        data: ArrayBufferView,
      ): GPUBuffer => {
        const buffer = device.createBuffer({
          label,
          size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
          usage,
          mappedAtCreation: true,
        });
        new Uint8Array(buffer.getMappedRange()).set(
          new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
        );
        buffer.unmap();
        buffers.push(buffer);
        return buffer;
      };

      try {
        const dynamic = createInitializedBuffer(
          "phase1-stable-oracle-dynamic",
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          gpuApi.packJGS2InitialDynamic(baseInput, offsets),
        );
        const vertices = createInitializedBuffer(
          "phase1-stable-oracle-vertices",
          GPUBufferUsage.STORAGE,
          gpuApi.packJGS2VertexStatic(baseInput),
        );
        const tets = createInitializedBuffer(
          "phase1-stable-oracle-tets",
          GPUBufferUsage.STORAGE,
          gpuApi.packJGS2TetStatic(baseInput),
        );
        const stiffness = createInitializedBuffer(
          "phase1-stable-oracle-stiffness",
          GPUBufferUsage.STORAGE,
          baseInput.tetRestStiffness,
        );
        const adjacency = createInitializedBuffer(
          "phase1-stable-oracle-adjacency",
          GPUBufferUsage.STORAGE,
          baseInput.adjacency,
        );
        evaluator = await diagnosticsApi.JGS2GpuOracleEvaluator.create(
          device,
          baseInput.vertexCount,
          baseInput.tetCount,
          { dynamic, vertices, tets, stiffness, adjacency },
        );

        for (const index of caseIndices) {
          const pose = corpus[index];
          if (!pose) {
            throw new Error(`Missing Phase 1 corpus case ${index}.`);
          }
          try {
            const poseInputPositions = new Float32Array(baseInput.positions.length);
            const snappedPositions = new Float64Array(pose.positions.length);
            for (let vertex = 0; vertex < baseInput.vertexCount; vertex += 1) {
              for (let coordinate = 0; coordinate < 3; coordinate += 1) {
                const value = Math.fround(pose.positions[vertex * 3 + coordinate]!);
                poseInputPositions[vertex * 4 + coordinate] = value;
                snappedPositions[vertex * 3 + coordinate] = value;
              }
              poseInputPositions[vertex * 4 + 3] = 1;
            }
            const packedDynamic = gpuApi.packJGS2InitialDynamic(
              { ...baseInput, positions: poseInputPositions },
              offsets,
            );
            device.queue.writeBuffer(dynamic, 0, packedDynamic);

            const objective = await evaluator.evaluate({
              currentPositionOffset: offsets.posA,
              predictedPositionOffset: offsets.predicted,
              finalUpdateOffset: offsets.finalUpdate,
              timestep: definition.settings.timestep,
              floorHeight: definition.settings.floorY,
              floorStiffness: 0,
              rotationEpsilon: 1e-7,
            });
            const materialReference = cpuApi.evaluateStableNeoHookeanMesh(
              mirrorMesh,
              mirrorRestData,
              definition.materials,
              snappedPositions,
            );
            const gpuMaterialGradient: number[] = [];
            const cpuMaterialGradient: number[] = [];
            const gpuMaterialLocalHessians: number[] = [];
            const cpuMaterialLocalHessians: number[] = [];
            const fullDimension = baseInput.vertexCount * 3;
            for (const vertex of precomputed.restSystem.activeVertices) {
              const record = objective.vertices[vertex]!;
              gpuMaterialGradient.push(...record.gradient);
              for (let coordinate = 0; coordinate < 3; coordinate += 1) {
                cpuMaterialGradient.push(
                  materialReference.gradient[vertex * 3 + coordinate]!,
                );
              }
              for (let row = 0; row < 3; row += 1) {
                for (let column = 0; column < 3; column += 1) {
                  gpuMaterialLocalHessians.push(
                    record.localHessian[row * 3 + column]! -
                      (row === column ? record.inertiaWeight : 0),
                  );
                  cpuMaterialLocalHessians.push(
                    materialReference.hessian[
                      (vertex * 3 + row) * fullDimension +
                        vertex * 3 +
                        column
                    ]!,
                  );
                }
              }
            }

            const predictedPositions = snappedPositions.slice();
            const targetOffset = [0.012, -0.009, 0.007] as const;
            for (const vertex of precomputed.restSystem.activeVertices) {
              for (let coordinate = 0; coordinate < 3; coordinate += 1) {
                const predicted = Math.fround(
                  snappedPositions[vertex * 3 + coordinate]! -
                    targetOffset[coordinate]!,
                );
                predictedPositions[vertex * 3 + coordinate] = predicted;
                packedDynamic[(offsets.predicted + vertex) * 4 + coordinate] =
                  predicted;
              }
            }
            device.queue.writeBuffer(dynamic, 0, packedDynamic);

            const cpuOracle = cpuApi.createStableNeoHookeanImplicitEulerOracle({
              mesh: mirrorMesh,
              restData: mirrorRestData,
              materials: definition.materials,
              lumpedMasses: mirrorMasses,
              restSystem: precomputed.restSystem,
              timestep: definition.settings.timestep,
              predictedPositions,
            });
            const coordinates = cpuApi.activeCoordinatesFromFullPositions(
              snappedPositions,
              precomputed.restSystem,
            );
            const reference = cpuOracle.evaluate(coordinates);
            const diagnostics = await evaluator.evaluate({
              currentPositionOffset: offsets.posA,
              predictedPositionOffset: offsets.predicted,
              finalUpdateOffset: offsets.finalUpdate,
              timestep: definition.settings.timestep,
              floorHeight: definition.settings.floorY,
              floorStiffness: 0,
              rotationEpsilon: 1e-7,
            });

            const gpuImplicitGradient: number[] = [];
            const gpuImplicitLocalHessians: number[] = [];
            const cpuImplicitLocalHessians: number[] = [];
            for (const vertex of precomputed.restSystem.activeVertices) {
              const activeBase = precomputed.restSystem.vertexToActiveDof[vertex]!;
              gpuImplicitGradient.push(...diagnostics.vertices[vertex]!.gradient);
              gpuImplicitLocalHessians.push(
                ...diagnostics.vertices[vertex]!.localHessian,
              );
              for (let row = 0; row < 3; row += 1) {
                for (let column = 0; column < 3; column += 1) {
                  cpuImplicitLocalHessians.push(
                    reference.hessian[
                      (activeBase + row) * reference.gradient.length +
                        activeBase +
                        column
                    ]!,
                  );
                }
              }
            }
            const objectiveForceNorm = Math.sqrt(
              gpuMaterialGradient.reduce(
                (sum, value) => sum + value * value,
                0,
              ),
            );
            cases.push({
              id: pose.id,
              index: pose.index,
              kind: pose.kind,
              expectedDeterminant: pose.determinant,
              minimumDeterminant: diagnostics.minimumDeformationDeterminant,
              finite: diagnostics.finite,
              energyError: gpuApi.jgs2DiagnosticRelativeError(
                [objective.components.elasticity],
                [materialReference.energy],
              ),
              gradientError: gpuApi.jgs2DiagnosticRelativeError(
                gpuMaterialGradient,
                cpuMaterialGradient,
              ),
              localHessianError: gpuApi.jgs2DiagnosticRelativeError(
                gpuMaterialLocalHessians,
                cpuMaterialLocalHessians,
              ),
              implicitEnergyError: gpuApi.jgs2DiagnosticRelativeError(
                [diagnostics.energy],
                [reference.energy],
              ),
              implicitGradientError: gpuApi.jgs2DiagnosticRelativeError(
                gpuImplicitGradient,
                reference.gradient,
              ),
              implicitLocalHessianError: gpuApi.jgs2DiagnosticRelativeError(
                gpuImplicitLocalHessians,
                cpuImplicitLocalHessians,
              ),
              zeroReferenceEnergyError: Math.abs(
                objective.components.elasticity,
              ),
              zeroReferenceForceError: objectiveForceNorm,
              error: "",
            });
          } catch (error) {
            cases.push({
              id: pose.id,
              index: pose.index,
              kind: pose.kind,
              expectedDeterminant: pose.determinant,
              minimumDeterminant: 0,
              finite: false,
              energyError: Number.POSITIVE_INFINITY,
              gradientError: Number.POSITIVE_INFINITY,
              localHessianError: Number.POSITIVE_INFINITY,
              implicitEnergyError: Number.POSITIVE_INFINITY,
              implicitGradientError: Number.POSITIVE_INFINITY,
              implicitLocalHessianError: Number.POSITIVE_INFINITY,
              zeroReferenceEnergyError: Number.POSITIVE_INFINITY,
              zeroReferenceForceError: Number.POSITIVE_INFINITY,
              error: error instanceof Error ? error.message : String(error),
            });
          }
        }
      } catch (error) {
        fatalError = error instanceof Error ? error.message : String(error);
      }

      await device.queue.onSubmittedWorkDone();
      const validationError = await device.popErrorScope();
      evaluator?.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
      device.destroy();
      return {
        adapterDescription,
        isFallbackAdapter: adapter.info.isFallbackAdapter,
        fatalError,
        validationError: validationError?.message ?? "",
        uncapturedErrors,
        elapsedMilliseconds: performance.now() - started,
        cases,
      };
    },
    PHASE1_GPU_CASE_INDICES,
  );

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.fatalError).toBe("");
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(browserErrors).toEqual([]);
  expect(result.cases.map((entry) => entry.index)).toEqual([
    ...PHASE1_GPU_CASE_INDICES,
  ]);

  let worstEnergy = { error: 0, id: "" };
  let worstGradient = { error: 0, id: "" };
  let worstHessian = { error: 0, id: "" };
  let worstRestEnergy = 0;
  let worstRestForce = 0;
  let worstRigidEnergy = 0;
  let worstRigidForce = 0;
  for (const entry of result.cases) {
    expect(entry.error, `${entry.id} GPU error`).toBe("");
    expect(entry.finite, `${entry.id} finite`).toBe(true);
    expect(entry.minimumDeterminant, `${entry.id} positive J`).toBeGreaterThan(0);
    expect(
      Math.abs(entry.minimumDeterminant - entry.expectedDeterminant),
      `${entry.id} determinant`,
    ).toBeLessThanOrEqual(1e-5);
    expect(entry.energyError, `${entry.id} energy`).toBeLessThanOrEqual(
      GPU_PARITY_TOLERANCE,
    );
    expect(entry.gradientError, `${entry.id} gradient`).toBeLessThanOrEqual(
      GPU_PARITY_TOLERANCE,
    );
    expect(entry.localHessianError, `${entry.id} local Hessian`).toBeLessThanOrEqual(
      GPU_PARITY_TOLERANCE,
    );
    expect(
      entry.implicitEnergyError,
      `${entry.id} implicit energy`,
    ).toBeLessThanOrEqual(GPU_PARITY_TOLERANCE);
    expect(
      entry.implicitGradientError,
      `${entry.id} implicit gradient`,
    ).toBeLessThanOrEqual(GPU_PARITY_TOLERANCE);
    expect(
      entry.implicitLocalHessianError,
      `${entry.id} implicit local Hessian`,
    ).toBeLessThanOrEqual(GPU_PARITY_TOLERANCE);
    if (entry.kind === "rest") {
      worstRestEnergy = Math.max(
        worstRestEnergy,
        entry.zeroReferenceEnergyError,
      );
      worstRestForce = Math.max(
        worstRestForce,
        entry.zeroReferenceForceError,
      );
      expect(entry.zeroReferenceEnergyError).toBeLessThanOrEqual(
        GPU_REST_TOLERANCE,
      );
      expect(entry.zeroReferenceForceError).toBeLessThanOrEqual(
        GPU_REST_TOLERANCE,
      );
    }
    if (entry.kind === "rigid") {
      worstRigidEnergy = Math.max(
        worstRigidEnergy,
        entry.zeroReferenceEnergyError,
      );
      worstRigidForce = Math.max(
        worstRigidForce,
        entry.zeroReferenceForceError,
      );
      expect(entry.zeroReferenceEnergyError).toBeLessThanOrEqual(
        GPU_RIGID_TOLERANCE,
      );
      expect(entry.zeroReferenceForceError).toBeLessThanOrEqual(
        GPU_RIGID_TOLERANCE,
      );
    }
    if (entry.energyError > worstEnergy.error) {
      worstEnergy = { error: entry.energyError, id: entry.id };
    }
    if (entry.gradientError > worstGradient.error) {
      worstGradient = { error: entry.gradientError, id: entry.id };
    }
    if (entry.localHessianError > worstHessian.error) {
      worstHessian = { error: entry.localHessianError, id: entry.id };
    }
  }

  const evidence = {
    adapter: result.adapterDescription,
    selectedCaseIndices: PHASE1_GPU_CASE_INDICES,
    parityTolerance: GPU_PARITY_TOLERANCE,
    objectiveTolerances: {
      rest: GPU_REST_TOLERANCE,
      rigid: GPU_RIGID_TOLERANCE,
    },
    objectiveWorst: {
      restEnergy: worstRestEnergy,
      restForce: worstRestForce,
      rigidEnergy: worstRigidEnergy,
      rigidForce: worstRigidForce,
    },
    elapsedMilliseconds: result.elapsedMilliseconds,
    worst: {
      energy: worstEnergy,
      gradient: worstGradient,
      localHessian: worstHessian,
    },
    cases: result.cases,
  };
  await testInfo.attach("phase1-stable-neo-hookean-gpu-oracle.json", {
    body: Buffer.from(JSON.stringify(evidence, null, 2)),
    contentType: "application/json",
  });
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: result.adapterDescription,
  });
  console.log(
    `P1 stable material GPU oracle (${result.cases.length} poses, ` +
      `${result.elapsedMilliseconds.toFixed(1)} ms): energy ` +
      `${worstEnergy.error.toExponential(3)} (${worstEnergy.id}), gradient ` +
      `${worstGradient.error.toExponential(3)} (${worstGradient.id}), ` +
      `local Hessian ${worstHessian.error.toExponential(3)} (${worstHessian.id}); ` +
      `rest energy/force ${worstRestEnergy.toExponential(3)}/` +
      `${worstRestForce.toExponential(3)}, rigid energy/force ` +
      `${worstRigidEnergy.toExponential(3)}/${worstRigidForce.toExponential(3)}`,
  );
});

test("P1-FRAME-01: GPU vertex frames are scale-invariant polar averages of deformation gradients", async ({
  page,
}, testInfo) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });
  await page.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });

  const result = await page.evaluate<Phase1GpuFrameResult>(async () => {
    const [cpuApi, scenes, phase1, gpuApi] = await Promise.all([
      import("/src/simulation/cpu/index.ts"),
      import("/src/scenes/index.ts"),
      import("/src/scenes/phase1.ts"),
      import("/src/simulation/gpu/index.ts"),
    ]);
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("Phase 1 frame parity requires hardware WebGPU.");
    }
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    const device = await adapter.requestDevice();
    const uncapturedErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      uncapturedErrors.push(event.error.message);
    });
    device.pushErrorScope("validation");

    const meshScale = 1e-2;
    const mesh = cpuApi.generateRegularCuboidMesh({
      cells: [2, 2, 2],
      origin: [-0.8 * meshScale, -0.5 * meshScale, -0.65 * meshScale],
      size: [1.6 * meshScale, 1.1 * meshScale, 1.3 * meshScale],
    });
    const definition = {
      id: "phase1.vertex-frame-oracle",
      title: "Phase 1 vertex frame oracle",
      description: "Private non-affine CPU/GPU vertex-frame parity fixture.",
      mesh,
      materials: [phase1.PHASE1_STABLE_NEO_HOOKEAN_MATERIAL],
      settings: {
        timestep: 1 / 60,
        gravity: [0, 0, 0] as const,
        floorY: -100,
        solverIterations: 1,
        cubatureSamples: 4 as const,
      },
      camera: {
        eye: [4, 3, 5] as const,
        target: [0, 0, 0] as const,
        up: [0, 1, 0] as const,
        fovYRadians: Math.PI / 4,
      },
      landmark: {
        vertex: 0,
        label: "frame fixture",
        expectedMotion: [0, 0, 0] as const,
      },
    };
    // This test isolates the vertex-frame construction on a deliberately
    // dense 48-tet fixture. Precompute its rest-equivalent linear bases, then
    // switch only the packed material parameters/tag to stable Neo-Hookean.
    // Nonlinear Cubature itself has a separate quality-gated production E2E.
    const preprocessingDefinition = {
      ...definition,
      materials: [
        {
          ...phase1.PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
          model: "corotated-linear" as const,
        },
      ],
    };
    const precomputed = cpuApi.buildPrecomputedScene(preprocessingDefinition);
    const baseInput = scenes.toJGS2GpuInput(precomputed);
    const stableParameters = cpuApi.computeStableNeoHookeanParameters(
      phase1.PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
    );
    for (let tetrahedron = 0; tetrahedron < baseInput.tetCount; tetrahedron += 1) {
      baseInput.tetMeta[tetrahedron * 4 + 1] = stableParameters.lambda;
      baseInput.tetMeta[tetrahedron * 4 + 2] = stableParameters.mu;
      baseInput.tetMeta[tetrahedron * 4 + 3] =
        gpuApi.JGS2_MATERIAL_STABLE_NEO_HOOKEAN;
    }
    const pose = new Float64Array(mesh.positions.length);
    const inputPositions = new Float32Array(baseInput.positions.length);
    for (let vertex = 0; vertex < mesh.positions.length / 3; vertex += 1) {
      const x = mesh.positions[vertex * 3]!;
      const y = mesh.positions[vertex * 3 + 1]!;
      const z = mesh.positions[vertex * 3 + 2]!;
      const normalizedX = x / meshScale;
      const normalizedY = y / meshScale;
      const normalizedZ = z / meshScale;
      const deformed = [
        meshScale * (normalizedX + 0.25 * normalizedY * normalizedZ),
        meshScale * (normalizedY - 0.15 * normalizedX * normalizedX),
        meshScale * (normalizedZ + 0.18 * normalizedX * normalizedY),
      ];
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const value = Math.fround(deformed[coordinate]!);
        pose[vertex * 3 + coordinate] = value;
        inputPositions[vertex * 4 + coordinate] = value;
      }
      inputPositions[vertex * 4 + 3] = 1;
    }
    const stencil = cpuApi.buildVertexDeformationGradientStencil(
      mesh,
      precomputed.restTetraData,
    );
    const cpuFrames = cpuApi.computeVertexPolarFrames(stencil, pose);
    const solver = await gpuApi.JGS2GpuSolver.create(
      device,
      { ...baseInput, positions: inputPositions },
      {
        timestep: definition.settings.timestep,
        gravity: [0, 0, 0],
        iterations: 1,
        floorHeight: definition.settings.floorY,
        floorStiffness: 0,
        maxStep: 0.01 * meshScale,
        parityMode: true,
      },
    );
    solver.stepExactIterations(1);
    await solver.awaitIdle();
    const packedGpuFrames = await solver.readVertexRotations();
    const packedSolvedPositions = await solver.readPositions();
    let maximumRelativeError = 0;
    for (let vertex = 0; vertex < baseInput.vertexCount; vertex += 1) {
      const gpuFrame = new Float64Array(9);
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 3; column += 1) {
          gpuFrame[row * 3 + column] =
            packedGpuFrames[vertex * 12 + column * 4 + row]!;
        }
      }
      maximumRelativeError = Math.max(
        maximumRelativeError,
        cpuApi.relativeError(
          gpuFrame,
          cpuFrames.subarray(vertex * 9, (vertex + 1) * 9),
        ),
      );
    }
    const solvedPositions = new Float64Array(mesh.positions.length);
    let solvedPositionsFinite = true;
    for (let vertex = 0; vertex < baseInput.vertexCount; vertex += 1) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const value = packedSolvedPositions[vertex * 4 + coordinate]!;
        solvedPositions[vertex * 3 + coordinate] = value;
        solvedPositionsFinite = solvedPositionsFinite && Number.isFinite(value);
      }
    }
    const solvedMaterial = cpuApi.evaluateStableNeoHookeanMesh(
      mesh,
      precomputed.restTetraData,
      definition.materials,
      solvedPositions,
    );
    const minimumSolvedDeterminant = Math.min(
      ...solvedMaterial.deformationDeterminants,
    );
    const explicitReadbacks = solver.explicitDiagnosticReadbackCount;
    solver.destroy();
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    device.destroy();
    return {
      adapterDescription,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      meshScale,
      maximumRelativeError,
      solvedPositionsFinite,
      minimumSolvedDeterminant,
      explicitReadbacks,
      validationError: validationError?.message ?? "",
      uncapturedErrors,
    };
  });

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(browserErrors).toEqual([]);
  expect(result.meshScale).toBe(1e-2);
  expect(result.solvedPositionsFinite).toBe(true);
  expect(result.minimumSolvedDeterminant).toBeGreaterThan(0);
  expect(result.explicitReadbacks).toBe(2);
  expect(result.maximumRelativeError).toBeLessThanOrEqual(1e-4);
  await testInfo.attach("phase1-vertex-frame-parity.json", {
    body: Buffer.from(JSON.stringify(result, null, 2)),
    contentType: "application/json",
  });
  console.log(
    `P1 GPU vertex-frame scale parity (${result.meshScale}x): maximum ` +
      `relative error ${result.maximumRelativeError.toExponential(3)}, ` +
      `solved min J ${result.minimumSolvedDeterminant.toExponential(3)}`,
  );
});
