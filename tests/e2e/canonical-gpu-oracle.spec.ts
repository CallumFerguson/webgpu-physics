import { expect, test } from "@playwright/test";

const CANONICAL_GPU_CASE_INDICES = [0, 1, 2, 3, 4, 7, 17, 31, 47, 63] as const;
const GPU_ORACLE_RELATIVE_TOLERANCE = 1e-3;
const GPU_OBJECTIVE_ENERGY_TOLERANCE = 1e-3;
const GPU_OBJECTIVE_FORCE_TOLERANCE = 5e-2;

interface CanonicalGpuOracleCaseResult {
  readonly id: string;
  readonly index: number;
  readonly kind: "rest" | "rigid" | "deformed";
  readonly determinant: number;
  readonly finite: boolean;
  readonly energy: number;
  readonly elasticityEnergy: number;
  readonly inertiaEnergy: number;
  readonly floorContactEnergy: number;
  readonly gradientNorm: number;
  readonly objectiveEnergy: number;
  readonly objectiveGradientNorm: number;
  readonly energyError: number;
  readonly gradientError: number;
  readonly localHessianError: number;
  readonly cpuEnergy: number;
  readonly cpuGradientNorm: number;
  readonly canonicalCpuEnergy: number;
  readonly canonicalCpuGradientNorm: number;
  readonly minimumDeformationDeterminant: number;
  readonly error: string;
}

interface CanonicalGpuOracleResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly elapsedMilliseconds: number;
  readonly fatalError: string;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly cases: readonly CanonicalGpuOracleCaseResult[];
}

test("P0-EC-03: canonical exact GPU energy, gradient, and local Hessian match CPU", async ({
  page,
}, testInfo) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => {
    browserErrors.push(`Page error: ${error.message}`);
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`Console error: ${message.text()}`);
    }
  });

  await page.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });
  const result = await page.evaluate<CanonicalGpuOracleResult, readonly number[]>(
    async (caseIndices) => {
      const [cpuApi, scenes, gpuApi, diagnosticsApi] = await Promise.all([
        import("/src/simulation/cpu/index.ts"),
        import("/src/scenes/index.ts"),
        import("/src/simulation/gpu/index.ts"),
        import("/src/simulation/gpu/jgs2-diagnostics.ts"),
      ]);
      const adapter = await navigator.gpu?.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        throw new Error("P0-EC-03 requires a hardware WebGPU adapter.");
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

      const fixture = cpuApi.buildPhase0OracleFixture();
      const corpus = cpuApi.generatePhase0OraclePoseCorpus();
      const precomputed = cpuApi.buildPrecomputedScene(fixture.definition);
      const baseInput = scenes.toJGS2GpuInput(precomputed);
      const offsets = gpuApi.computeJGS2DynamicOffsets(
        baseInput.vertexCount,
        baseInput.tetCount,
        1,
      );
      // Mirror the actual f32 GPU coefficients in Float64. This keeps the CPU
      // implementation independent while ensuring both sides evaluate the
      // same uploaded problem rather than slightly different quantizations.
      const gpuMirrorRestPositions = new Float64Array(
        fixture.mesh.positions.length,
      );
      const gpuMirrorMasses = new Float64Array(baseInput.vertexCount);
      for (let vertex = 0; vertex < baseInput.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          gpuMirrorRestPositions[vertex * 3 + coordinate] =
            baseInput.vertexRest[vertex * 4 + coordinate]!;
        }
        gpuMirrorMasses[vertex] = baseInput.vertexRest[vertex * 4 + 3]!;
      }
      const gpuMirrorInverseRest = new Float64Array(baseInput.tetCount * 9);
      const gpuMirrorVolumes = new Float64Array(baseInput.tetCount);
      for (let tetrahedron = 0; tetrahedron < baseInput.tetCount; tetrahedron += 1) {
        gpuMirrorVolumes[tetrahedron] = baseInput.tetMeta[tetrahedron * 4]!;
        for (let row = 0; row < 3; row += 1) {
          for (let column = 0; column < 3; column += 1) {
            gpuMirrorInverseRest[tetrahedron * 9 + row * 3 + column] =
              baseInput.tetInverseDm[tetrahedron * 12 + column * 4 + row]!;
          }
        }
      }
      const gpuMirrorMesh = {
        ...fixture.mesh,
        positions: gpuMirrorRestPositions,
      };
      const gpuMirrorRestData = {
        volumes: gpuMirrorVolumes,
        inverseRestMatrices: gpuMirrorInverseRest,
        stiffnessMatrices: Float64Array.from(baseInput.tetRestStiffness),
      };

      const buffers: GPUBuffer[] = [];
      let evaluator: InstanceType<typeof diagnosticsApi.JGS2GpuOracleEvaluator> | undefined;
      const cases: CanonicalGpuOracleCaseResult[] = [];
      let fatalError = "";
      const started = performance.now();

      const createInitializedBuffer = (
        label: string,
        usage: GPUBufferUsageFlags,
        data: ArrayBufferView,
      ): GPUBuffer => {
        const size = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
        const buffer = device.createBuffer({
          label,
          size,
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
          "canonical-oracle-dynamic",
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          gpuApi.packJGS2InitialDynamic(baseInput, offsets),
        );
        const vertices = createInitializedBuffer(
          "canonical-oracle-vertices",
          GPUBufferUsage.STORAGE,
          gpuApi.packJGS2VertexStatic(baseInput),
        );
        const tets = createInitializedBuffer(
          "canonical-oracle-tets",
          GPUBufferUsage.STORAGE,
          gpuApi.packJGS2TetStatic(baseInput),
        );
        const stiffness = createInitializedBuffer(
          "canonical-oracle-stiffness",
          GPUBufferUsage.STORAGE,
          baseInput.tetRestStiffness,
        );
        const adjacency = createInitializedBuffer(
          "canonical-oracle-adjacency",
          GPUBufferUsage.STORAGE,
          baseInput.adjacency,
        );
        evaluator = await diagnosticsApi.JGS2GpuOracleEvaluator.create(
          device,
          baseInput.vertexCount,
          baseInput.tetCount,
          { dynamic, vertices, tets, stiffness, adjacency },
        );

        // Each evaluate() is awaited before the next pose is uploaded. This
        // keeps one diagnostic submission and one mapped readback outstanding.
        for (const index of caseIndices) {
          const pose = corpus[index];
          if (!pose) {
            cases.push({
              id: `missing/${index}`,
              index,
              kind: "deformed",
              determinant: 0,
              finite: false,
              energy: 0,
              elasticityEnergy: 0,
              inertiaEnergy: 0,
              floorContactEnergy: 0,
              gradientNorm: 0,
              objectiveEnergy: 0,
              objectiveGradientNorm: 0,
              energyError: Number.POSITIVE_INFINITY,
              gradientError: Number.POSITIVE_INFINITY,
              localHessianError: Number.POSITIVE_INFINITY,
              cpuEnergy: 0,
              cpuGradientNorm: 0,
              canonicalCpuEnergy: 0,
              canonicalCpuGradientNorm: 0,
              minimumDeformationDeterminant: 0,
              error: `Missing canonical corpus case ${index}.`,
            });
            continue;
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
            const poseInput = { ...baseInput, positions: poseInputPositions };
            const packedDynamic = gpuApi.packJGS2InitialDynamic(poseInput, offsets);
            device.queue.writeBuffer(dynamic, 0, packedDynamic);

            // The material-only checkpoint is the direct objectivity test.
            // Keeping current == predicted isolates co-rotated elastic force.
            const objectiveDiagnostics = await evaluator.evaluate({
              currentPositionOffset: offsets.posA,
              predictedPositionOffset: offsets.predicted,
              finalUpdateOffset: offsets.finalUpdate,
              timestep: fixture.definition.settings.timestep,
              floorHeight: fixture.definition.settings.floorY,
              floorStiffness: 0,
              rotationEpsilon: 1e-7,
            });

            // A deterministic nonzero inertial target makes the roadmap's
            // max(1, ||a||, ||b||) relative error well-conditioned at rigid
            // poses, whose exact material gradient is zero. The separate
            // checkpoint above still gates the actual f32 objectivity error.
            const predictedPositions = snappedPositions.slice();
            const targetOffset = [0.01, -0.008, 0.006] as const;
            for (const vertex of fixture.restSystem.activeVertices) {
              for (let coordinate = 0; coordinate < 3; coordinate += 1) {
                const predicted = Math.fround(
                  snappedPositions[vertex * 3 + coordinate]! -
                    targetOffset[coordinate]!,
                );
                predictedPositions[vertex * 3 + coordinate] = predicted;
                packedDynamic[
                  (offsets.predicted + vertex) * 4 + coordinate
                ] = predicted;
              }
            }
            device.queue.writeBuffer(dynamic, 0, packedDynamic);

            const cpuOracle = cpuApi.createCorotatedLinearImplicitEulerOracle({
              mesh: gpuMirrorMesh,
              restData: gpuMirrorRestData,
              lumpedMasses: gpuMirrorMasses,
              restSystem: fixture.restSystem,
              timestep: fixture.definition.settings.timestep,
              predictedPositions,
              rotationPositions: snappedPositions,
            });
            const coordinates = cpuApi.activeCoordinatesFromFullPositions(
              snappedPositions,
              fixture.restSystem,
            );
            const reference = cpuOracle.evaluate(coordinates);
            const canonicalReference = cpuApi.evaluatePhase0OraclePose(
              fixture,
              pose,
            ).evaluation;
            const diagnostics = await evaluator.evaluate({
              currentPositionOffset: offsets.posA,
              predictedPositionOffset: offsets.predicted,
              finalUpdateOffset: offsets.finalUpdate,
              timestep: fixture.definition.settings.timestep,
              floorHeight: fixture.definition.settings.floorY,
              floorStiffness: 0,
              rotationEpsilon: 1e-7,
            });

            const gpuGradient: number[] = [];
            const gpuLocalHessians: number[] = [];
            const cpuLocalHessians: number[] = [];
            for (const vertex of fixture.restSystem.activeVertices) {
              const activeBase = fixture.restSystem.vertexToActiveDof[vertex]!;
              gpuGradient.push(...diagnostics.vertices[vertex]!.gradient);
              gpuLocalHessians.push(...diagnostics.vertices[vertex]!.localHessian);
              for (let row = 0; row < 3; row += 1) {
                for (let column = 0; column < 3; column += 1) {
                  cpuLocalHessians.push(
                    reference.hessian[
                      (activeBase + row) * reference.gradient.length +
                        activeBase +
                        column
                    ]!,
                  );
                }
              }
            }

            cases.push({
              id: pose.id,
              index: pose.index,
              kind: pose.kind,
              determinant: pose.determinant,
              finite: diagnostics.finite,
              energy: diagnostics.energy,
              elasticityEnergy: diagnostics.components.elasticity,
              inertiaEnergy: diagnostics.components.inertia,
              floorContactEnergy: diagnostics.components.floorContact,
              gradientNorm: diagnostics.gradientNorm,
              objectiveEnergy: objectiveDiagnostics.energy,
              objectiveGradientNorm: objectiveDiagnostics.gradientNorm,
              energyError: gpuApi.jgs2DiagnosticRelativeError(
                [diagnostics.energy],
                [reference.energy],
              ),
              gradientError: gpuApi.jgs2DiagnosticRelativeError(
                gpuGradient,
                reference.gradient,
              ),
              localHessianError: gpuApi.jgs2DiagnosticRelativeError(
                gpuLocalHessians,
                cpuLocalHessians,
              ),
              cpuEnergy: reference.energy,
              cpuGradientNorm: Math.hypot(...reference.gradient),
              canonicalCpuEnergy: canonicalReference.energy,
              canonicalCpuGradientNorm: Math.hypot(...canonicalReference.gradient),
              minimumDeformationDeterminant:
                diagnostics.minimumDeformationDeterminant,
              error: "",
            });
          } catch (error) {
            cases.push({
              id: pose.id,
              index: pose.index,
              kind: pose.kind,
              determinant: pose.determinant,
              finite: false,
              energy: 0,
              elasticityEnergy: 0,
              inertiaEnergy: 0,
              floorContactEnergy: 0,
              gradientNorm: 0,
              objectiveEnergy: 0,
              objectiveGradientNorm: 0,
              energyError: Number.POSITIVE_INFINITY,
              gradientError: Number.POSITIVE_INFINITY,
              localHessianError: Number.POSITIVE_INFINITY,
              cpuEnergy: 0,
              cpuGradientNorm: 0,
              canonicalCpuEnergy: 0,
              canonicalCpuGradientNorm: 0,
              minimumDeformationDeterminant: 0,
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
        elapsedMilliseconds: performance.now() - started,
        fatalError,
        validationError: validationError?.message ?? "",
        uncapturedErrors,
        cases,
      };
    },
    CANONICAL_GPU_CASE_INDICES,
  );

  expect(result.isFallbackAdapter).toBe(false);
  expect(result.adapterDescription).not.toMatch(/swiftshader/i);
  expect(result.fatalError).toBe("");
  expect(result.validationError).toBe("");
  expect(result.uncapturedErrors).toEqual([]);
  expect(browserErrors).toEqual([]);
  expect(result.cases.map((entry) => entry.index)).toEqual(
    [...CANONICAL_GPU_CASE_INDICES],
  );
  expect(result.cases.filter((entry) => entry.kind === "rest")).toHaveLength(1);
  expect(result.cases.filter((entry) => entry.kind === "rigid")).toHaveLength(3);
  expect(result.cases.filter((entry) => entry.kind === "deformed").length).toBeGreaterThanOrEqual(4);
  expect(result.cases.at(-1)?.index).toBe(63);

  let worstEnergy = { error: 0, id: "" };
  let worstGradient = { error: 0, id: "" };
  let worstLocalHessian = { error: 0, id: "" };
  for (const entry of result.cases) {
    expect(entry.error, `${entry.id} GPU oracle error`).toBe("");
    expect(entry.finite, `${entry.id} finite`).toBe(true);
    expect(entry.minimumDeformationDeterminant, `${entry.id} det(F)`).toBeGreaterThan(0);
    expect(entry.inertiaEnergy, `${entry.id} comparison inertial energy`).toBeGreaterThan(0);
    expect(entry.floorContactEnergy, `${entry.id} disabled floor`).toBe(0);
    expect(entry.energyError, `${entry.id} energy`).toBeLessThanOrEqual(
      GPU_ORACLE_RELATIVE_TOLERANCE,
    );
    expect(entry.gradientError, `${entry.id} gradient`).toBeLessThanOrEqual(
      GPU_ORACLE_RELATIVE_TOLERANCE,
    );
    expect(entry.localHessianError, `${entry.id} local Hessian`).toBeLessThanOrEqual(
      GPU_ORACLE_RELATIVE_TOLERANCE,
    );
    if (entry.kind !== "deformed") {
      expect(Math.abs(entry.canonicalCpuEnergy), `${entry.id} CPU objectivity energy`).toBeLessThan(
        1e-8,
      );
      expect(entry.canonicalCpuGradientNorm, `${entry.id} CPU objectivity force`).toBeLessThan(
        1e-7,
      );
      expect(Math.abs(entry.objectiveEnergy), `${entry.id} GPU objectivity energy`).toBeLessThanOrEqual(
        GPU_OBJECTIVE_ENERGY_TOLERANCE,
      );
      expect(entry.objectiveGradientNorm, `${entry.id} GPU objectivity force`).toBeLessThanOrEqual(
        GPU_OBJECTIVE_FORCE_TOLERANCE,
      );
    }
    if (entry.energyError > worstEnergy.error) {
      worstEnergy = { error: entry.energyError, id: entry.id };
    }
    if (entry.gradientError > worstGradient.error) {
      worstGradient = { error: entry.gradientError, id: entry.id };
    }
    if (entry.localHessianError > worstLocalHessian.error) {
      worstLocalHessian = { error: entry.localHessianError, id: entry.id };
    }
  }

  const evidence = {
    adapter: result.adapterDescription,
    selectedCaseIndices: CANONICAL_GPU_CASE_INDICES,
    relativeTolerance: GPU_ORACLE_RELATIVE_TOLERANCE,
    objectivityTolerances: {
      energy: GPU_OBJECTIVE_ENERGY_TOLERANCE,
      forceNorm: GPU_OBJECTIVE_FORCE_TOLERANCE,
    },
    elapsedMilliseconds: result.elapsedMilliseconds,
    worst: {
      energy: worstEnergy,
      gradient: worstGradient,
      localHessian: worstLocalHessian,
    },
    validationError: result.validationError,
    uncapturedErrors: result.uncapturedErrors,
    cases: result.cases,
  };
  await testInfo.attach("phase0-canonical-gpu-cpu-oracle.json", {
    body: Buffer.from(JSON.stringify(evidence, null, 2)),
    contentType: "application/json",
  });
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: result.adapterDescription,
  });
  console.log(
    `P0-EC-03 canonical GPU oracle (${result.cases.length} poses, ` +
      `${result.elapsedMilliseconds.toFixed(1)} ms): energy ` +
      `${worstEnergy.error.toExponential(3)} (${worstEnergy.id}), gradient ` +
      `${worstGradient.error.toExponential(3)} (${worstGradient.id}), local Hessian ` +
      `${worstLocalHessian.error.toExponential(3)} (${worstLocalHessian.id})`,
  );
});
