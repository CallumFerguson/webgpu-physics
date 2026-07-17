import { expect, test } from "@playwright/test";

interface OracleComparison {
  readonly adapterDescription: string;
  readonly energy: number;
  readonly inertiaEnergy: number;
  readonly elasticityEnergy: number;
  readonly energyError: number;
  readonly gradientError: number;
  readonly localHessianError: number;
  readonly finite: boolean;
  readonly floorContactEnergy: number;
  readonly gradientNorm: number;
  readonly residualNumerator: number;
  readonly residualDenominator: number;
  readonly gradientValid: boolean;
  readonly relativeResidual: number;
  readonly relativeResidualValid: boolean;
  readonly maximumUpdate: number;
  readonly maximumUpdateValid: boolean;
  readonly initialMaximumUpdateValid: boolean;
  readonly explicitUpdateMaximum: number;
  readonly explicitUpdatesValid: boolean;
  readonly minimumDeformationDeterminant: number;
  readonly minimumDeformationDeterminantValid: boolean;
  readonly activeVertexCount: number;
  readonly minimumContactDistance: number;
  readonly minimumContactDistanceValid: boolean;
  readonly cpuMinimumContactDistance: number;
  readonly activeContactCount: number;
  readonly activeContactCountValid: boolean;
  readonly cpuActiveContactCount: number;
  readonly candidateBufferOverflow: boolean;
  readonly candidateBufferOverflowValid: boolean;
  readonly validationError: string;
}

test("P0-EC-03 exact GPU oracle matches the CPU oracle with active floor contact", async ({
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
  await page.goto("/?test=1");

  const comparison = await page.evaluate<OracleComparison>(async () => {
    const scenes = await import("/src/scenes/index.ts");
    const gpuApi = await import("/src/simulation/gpu/index.ts");
    const cpuApi = await import("/src/simulation/cpu/index.ts");
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error(
        "P0-EC-03 requires the configured hardware WebGPU adapter.",
      );
    }
    const adapterDescription = [
      adapter.info.description,
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
    ]
      .filter(Boolean)
      .join(" / ");
    if (
      adapter.info.isFallbackAdapter ||
      /swiftshader/i.test(adapterDescription)
    ) {
      throw new Error(
        `P0-EC-03 requires hardware WebGPU; got ${adapterDescription || "a fallback adapter"}.`,
      );
    }
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");

    const scene = scenes.buildScene("drop");
    const input = scenes.toJGS2GpuInput(scene);
    // P0-EC-03 is the retained co-rotated CPU/GPU oracle contract. The public
    // drop demo now uses stable Neo-Hookean material, so keep this regression
    // fixture on its original material path independently of demo evolution.
    for (let tetrahedron = 0; tetrahedron < input.tetCount; tetrahedron += 1) {
      input.tetMeta[tetrahedron * 4 + 3] =
        gpuApi.JGS2_MATERIAL_COROTATED_LINEAR;
    }
    const timestep = scene.settings.timestep;
    const solver = await gpuApi.JGS2GpuSolver.create(device, input, {
      timestep,
      gravity: scene.settings.gravity,
      iterations: scene.settings.solverIterations,
      floorHeight: scene.settings.floorY,
      floorStiffness: 250_000,
      parityMode: true,
      regularization: 1e-6,
      rotationEpsilon: 1e-7,
      maxStep: 0.075,
    });

    try {
      const initialDiagnostics = await solver.readOracleDiagnostics();
      // This checkpoint is after impact, so the same comparison covers the
      // all-element elastic, inertial, and quadratic floor-contact terms.
      solver.stepFramesExactIterations(80, scene.settings.solverIterations);
      await solver.awaitIdle();
      const positions4 = await solver.readPositions();
      const predicted4 = await solver.readPredictedPositions();
      const finalUpdates = await solver.readFinalIterationUpdates();
      const diagnostics = await solver.readOracleDiagnostics();

      const positions = new Float64Array(input.vertexCount * 3);
      const predictedPositions = new Float64Array(input.vertexCount * 3);
      for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          positions[vertex * 3 + coordinate] =
            positions4[vertex * 4 + coordinate]!;
          predictedPositions[vertex * 3 + coordinate] =
            predicted4[vertex * 4 + coordinate]!;
        }
      }

      const oracle = cpuApi.createCorotatedLinearImplicitEulerOracle({
        mesh: scene.mesh,
        restData: scene.restTetraData,
        lumpedMasses: scene.lumpedMasses,
        restSystem: scene.restSystem,
        timestep,
        predictedPositions,
        rotationPositions: positions,
        floorContact: {
          height: scene.settings.floorY,
          stiffness: 250_000,
        },
      });
      const activeCoordinates = cpuApi.activeCoordinatesFromFullPositions(
        positions,
        scene.restSystem,
      );
      const reference = oracle.evaluate(activeCoordinates);

      const gpuGradient: number[] = [];
      const gpuLocalHessians: number[] = [];
      const cpuLocalHessians: number[] = [];
      let explicitUpdateMaximum = 0;
      let explicitUpdatesValid = true;
      let cpuMinimumContactDistance = Number.POSITIVE_INFINITY;
      let cpuActiveContactCount = 0;
      for (const vertex of scene.restSystem.activeVertices) {
        const activeBase = scene.restSystem.vertexToActiveDof[vertex]!;
        const updateMagnitude = finalUpdates[vertex * 4]!;
        explicitUpdateMaximum = Math.max(
          explicitUpdateMaximum,
          updateMagnitude,
        );
        explicitUpdatesValid &&=
          Number.isFinite(updateMagnitude) &&
          updateMagnitude >= 0 &&
          finalUpdates[vertex * 4 + 3]! >= 0.5;
        const signedFloorDistance =
          positions4[vertex * 4 + 1]! - scene.settings.floorY;
        cpuMinimumContactDistance = Math.min(
          cpuMinimumContactDistance,
          signedFloorDistance,
        );
        if (signedFloorDistance < 0) {
          cpuActiveContactCount += 1;
        }
        gpuGradient.push(...diagnostics.vertices[vertex]!.gradient);
        gpuLocalHessians.push(
          ...diagnostics.vertices[vertex]!.localHessian,
        );
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

      const validationError = await device.popErrorScope();
      return {
        adapterDescription,
        energy: diagnostics.energy,
        inertiaEnergy: diagnostics.components.inertia,
        elasticityEnergy: diagnostics.components.elasticity,
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
        finite: diagnostics.finite,
        floorContactEnergy: diagnostics.components.floorContact,
        gradientNorm: diagnostics.gradientNorm,
        residualNumerator: diagnostics.residualNumerator,
        residualDenominator: diagnostics.residualDenominator,
        gradientValid: diagnostics.gradientValid,
        relativeResidual: diagnostics.relativeResidual,
        relativeResidualValid: diagnostics.relativeResidualValid,
        maximumUpdate: diagnostics.maximumUpdate,
        maximumUpdateValid: diagnostics.maximumUpdateValid,
        initialMaximumUpdateValid:
          initialDiagnostics.maximumUpdateValid,
        explicitUpdateMaximum,
        explicitUpdatesValid,
        minimumDeformationDeterminant:
          diagnostics.minimumDeformationDeterminant,
        minimumDeformationDeterminantValid:
          diagnostics.minimumDeformationDeterminantValid,
        activeVertexCount: diagnostics.activeVertexCount,
        minimumContactDistance: diagnostics.minimumContactDistance,
        minimumContactDistanceValid:
          diagnostics.minimumContactDistanceValid,
        cpuMinimumContactDistance,
        activeContactCount: diagnostics.activeContactCount,
        activeContactCountValid: diagnostics.activeContactCountValid,
        cpuActiveContactCount,
        candidateBufferOverflow: diagnostics.candidateBufferOverflow,
        candidateBufferOverflowValid:
          diagnostics.candidateBufferOverflowValid,
        validationError: validationError?.message ?? "",
      };
    } finally {
      solver.destroy();
      device.destroy();
    }
  });

  await testInfo.attach("p0-ec-03-gpu-cpu-oracle.json", {
    body: JSON.stringify(comparison, null, 2),
    contentType: "application/json",
  });
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: comparison.adapterDescription,
  });
  console.log(
    `P0-EC-03 relative errors: energy ${comparison.energyError.toExponential(3)}, ` +
      `gradient ${comparison.gradientError.toExponential(3)}, ` +
      `local Hessian ${comparison.localHessianError.toExponential(3)}`,
  );

  expect(comparison.validationError).toBe("");
  expect(browserErrors).toEqual([]);
  expect(comparison.finite).toBe(true);
  expect(comparison.adapterDescription).not.toMatch(/swiftshader/i);
  expect(comparison.energy).toBeGreaterThan(0);
  expect(comparison.inertiaEnergy).toBeGreaterThanOrEqual(0);
  expect(comparison.elasticityEnergy).toBeGreaterThanOrEqual(0);
  expect(comparison.floorContactEnergy).toBeGreaterThan(0);
  expect(comparison.gradientNorm).toBeGreaterThan(0);
  expect(comparison.residualNumerator).toBe(comparison.gradientNorm);
  expect(comparison.residualDenominator).toBeGreaterThanOrEqual(1);
  expect(comparison.relativeResidual).toBeCloseTo(
    comparison.residualNumerator / comparison.residualDenominator,
    6,
  );
  expect(comparison.relativeResidualValid).toBe(true);
  expect(comparison.gradientValid).toBe(true);
  expect(comparison.initialMaximumUpdateValid).toBe(false);
  expect(comparison.maximumUpdate).toBeGreaterThan(0);
  expect(comparison.maximumUpdateValid).toBe(true);
  expect(comparison.explicitUpdatesValid).toBe(true);
  expect(comparison.maximumUpdate).toBeCloseTo(
    comparison.explicitUpdateMaximum,
    7,
  );
  expect(comparison.minimumDeformationDeterminant).toBeGreaterThan(0);
  expect(comparison.minimumDeformationDeterminantValid).toBe(true);
  expect(comparison.activeVertexCount).toBeGreaterThan(0);
  expect(comparison.energyError).toBeLessThanOrEqual(1e-3);
  expect(comparison.gradientError).toBeLessThanOrEqual(1e-3);
  expect(comparison.localHessianError).toBeLessThanOrEqual(1e-3);

  expect(comparison.minimumContactDistance).toBeLessThan(0);
  expect(comparison.minimumContactDistanceValid).toBe(true);
  expect(comparison.minimumContactDistance).toBeCloseTo(
    comparison.cpuMinimumContactDistance,
    6,
  );
  expect(comparison.activeContactCount).toBeGreaterThan(0);
  expect(comparison.activeContactCountValid).toBe(true);
  expect(comparison.activeContactCount).toBe(
    comparison.cpuActiveContactCount,
  );

  // General contact-candidate generation is not implemented yet, so overflow
  // remains explicitly invalid even though analytic-floor metrics are valid.
  expect(comparison.candidateBufferOverflow).toBe(false);
  expect(comparison.candidateBufferOverflowValid).toBe(false);
});
