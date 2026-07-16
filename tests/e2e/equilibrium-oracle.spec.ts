import { expect, test } from "@playwright/test";

import {
  buildPhase0OracleFixture,
  computeDirectComplementaryEquilibriumBasis,
  evaluatePhase0OraclePose,
  generatePhase0OraclePoseCorpus,
  relativeError,
  solveEquilibriumBasisNewtonStep,
  solveFullNewtonStep,
} from "../../src/simulation/cpu";

interface CanonicalEquilibriumCase {
  readonly id: string;
  readonly index: number;
  readonly kind: "rest" | "rigid" | "deformed";
  readonly determinant: number;
  readonly dimension: number;
  readonly activeVertices: readonly number[];
  readonly vertexToActiveDof: readonly number[];
  readonly gradient: readonly number[];
  readonly hessian: readonly number[];
  readonly bases: readonly number[];
  readonly fullNewtonStep: readonly number[];
  readonly maximumCpuBlockError: number;
}

interface GpuCaseResult {
  readonly id: string;
  readonly steps: readonly number[];
  readonly error: string;
  /** A returned result means every unmodified f32 Cholesky pivot was valid. */
  readonly exactNoRegularizationValidityAccepted: boolean;
}

interface GpuCorpusResult {
  readonly adapterDescription: string;
  readonly isFallbackAdapter: boolean;
  readonly elapsedMilliseconds: number;
  readonly validationError: string;
  readonly uncapturedErrors: readonly string[];
  readonly cases: readonly GpuCaseResult[];
}

function buildCanonicalEquilibriumCorpus(): readonly CanonicalEquilibriumCase[] {
  const fixture = buildPhase0OracleFixture();
  return generatePhase0OraclePoseCorpus().map((pose) => {
    const { evaluation } = evaluatePhase0OraclePose(fixture, pose);
    const fullNewtonStep = solveFullNewtonStep(evaluation);
    const basisCount = fixture.restSystem.activeVertices.length;
    const bases = new Float32Array(
      basisCount * fixture.restSystem.dimension * 3,
    );
    let maximumCpuBlockError = 0;

    for (let basisIndex = 0; basisIndex < basisCount; basisIndex += 1) {
      const vertex = fixture.restSystem.activeVertices[basisIndex]!;
      const localBase = fixture.restSystem.vertexToActiveDof[vertex]!;
      const direct = computeDirectComplementaryEquilibriumBasis(
        evaluation.hessian,
        evaluation.gradient.length,
        localBase,
      );
      bases.set(
        Float32Array.from(direct.basis),
        basisIndex * fixture.restSystem.dimension * 3,
      );
      maximumCpuBlockError = Math.max(
        maximumCpuBlockError,
        relativeError(
          solveEquilibriumBasisNewtonStep(evaluation, direct.basis),
          fullNewtonStep.subarray(localBase, localBase + 3),
        ),
      );
    }

    return {
      id: pose.id,
      index: pose.index,
      kind: pose.kind,
      determinant: pose.determinant,
      dimension: fixture.restSystem.dimension,
      activeVertices: [...fixture.restSystem.activeVertices],
      vertexToActiveDof: Array.from(fixture.restSystem.vertexToActiveDof),
      gradient: Array.from(Float32Array.from(evaluation.gradient)),
      hessian: Array.from(Float32Array.from(evaluation.hessian)),
      bases: Array.from(bases),
      fullNewtonStep: Array.from(fullNewtonStep),
      maximumCpuBlockError,
    };
  });
}

test("P0-EC-04: all canonical GPU equilibrium-basis blocks match Float64 full Newton", async ({
  page,
}, testInfo) => {
  test.setTimeout(180_000);
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => {
    browserErrors.push(`Page error: ${error.message}`);
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`Console error: ${message.text()}`);
    }
  });

  const corpus = buildCanonicalEquilibriumCorpus();
  expect(corpus).toHaveLength(64);
  expect(corpus.map((entry) => entry.index)).toEqual(
    Array.from({ length: 64 }, (_unused, index) => index),
  );
  for (const entry of corpus) {
    expect(entry.maximumCpuBlockError, `${entry.id} CPU block`).toBeLessThanOrEqual(
      1e-8,
    );
  }

  await page.goto("/?scene=minimal&test=1&parity=1", {
    waitUntil: "domcontentloaded",
  });
  const gpuResult = await page.evaluate<GpuCorpusResult, readonly CanonicalEquilibriumCase[]>(
    async (inputCases) => {
      const { solveDenseEquilibriumBasisOnGpu } = await import(
        "/src/simulation/gpu/equilibrium-oracle.ts"
      );
      const adapter = await navigator.gpu?.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        throw new Error("P0-EC-04 requires a hardware WebGPU adapter.");
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
      const started = performance.now();
      const cases: GpuCaseResult[] = [];

      try {
        // Sequential awaits keep exactly one submitted oracle solve outstanding.
        for (const input of inputCases) {
          try {
            const steps = await solveDenseEquilibriumBasisOnGpu(device, {
              gradient: new Float32Array(input.gradient),
              hessian: new Float32Array(input.hessian),
              bases: new Float32Array(input.bases),
            });
            cases.push({
              id: input.id,
              steps: Array.from(steps),
              error: "",
              exactNoRegularizationValidityAccepted: true,
            });
          } catch (error) {
            cases.push({
              id: input.id,
              steps: [],
              error: error instanceof Error ? error.message : String(error),
              exactNoRegularizationValidityAccepted: false,
            });
          }
        }
        await device.queue.onSubmittedWorkDone();
        const validationError = await device.popErrorScope();
        return {
          adapterDescription,
          isFallbackAdapter: adapter.info.isFallbackAdapter,
          elapsedMilliseconds: performance.now() - started,
          validationError: validationError?.message ?? "",
          uncapturedErrors,
          cases,
        };
      } finally {
        device.destroy();
      }
    },
    corpus,
  );

  expect(gpuResult.isFallbackAdapter).toBe(false);
  expect(gpuResult.adapterDescription).not.toMatch(/swiftshader/i);
  expect(gpuResult.validationError).toBe("");
  expect(gpuResult.uncapturedErrors).toEqual([]);
  expect(browserErrors).toEqual([]);
  expect(gpuResult.cases).toHaveLength(corpus.length);

  let maximumGpuBlockError = 0;
  let worstCaseId = "";
  let worstVertex = -1;
  const caseEvidence: Array<{
    id: string;
    kind: CanonicalEquilibriumCase["kind"];
    determinant: number;
    maximumCpuBlockError: number;
    maximumGpuBlockError: number;
    exactNoRegularizationValidityAccepted: boolean;
    blockErrors: number[];
  }> = [];

  for (let caseIndex = 0; caseIndex < corpus.length; caseIndex += 1) {
    const input = corpus[caseIndex]!;
    const result = gpuResult.cases[caseIndex]!;
    expect(result.id).toBe(input.id);
    expect(result.error, `${input.id} GPU oracle error`).toBe("");
    expect(
      result.exactNoRegularizationValidityAccepted,
      `${input.id} must use unmodified positive f32 pivots`,
    ).toBe(true);
    expect(result.steps).toHaveLength(input.activeVertices.length * 3);

    let caseMaximum = 0;
    const blockErrors: number[] = [];
    for (let basisIndex = 0; basisIndex < input.activeVertices.length; basisIndex += 1) {
      const vertex = input.activeVertices[basisIndex]!;
      const localBase = input.vertexToActiveDof[vertex]!;
      const gpuBlock = result.steps.slice(basisIndex * 3, basisIndex * 3 + 3);
      expect(
        gpuBlock.every(Number.isFinite),
        `${input.id} vertex ${vertex} GPU block finite`,
      ).toBe(true);
      const blockError = relativeError(
        gpuBlock,
        input.fullNewtonStep.slice(localBase, localBase + 3),
      );
      blockErrors.push(blockError);
      caseMaximum = Math.max(caseMaximum, blockError);
      if (blockError > maximumGpuBlockError) {
        maximumGpuBlockError = blockError;
        worstCaseId = input.id;
        worstVertex = vertex;
      }
      expect(
        blockError,
        `${input.id} vertex ${vertex} GPU/Float64 Newton block`,
      ).toBeLessThanOrEqual(1e-3);
    }
    caseEvidence.push({
      id: input.id,
      kind: input.kind,
      determinant: input.determinant,
      maximumCpuBlockError: input.maximumCpuBlockError,
      maximumGpuBlockError: caseMaximum,
      exactNoRegularizationValidityAccepted:
        result.exactNoRegularizationValidityAccepted,
      blockErrors,
    });
  }

  const evidence = {
    adapter: gpuResult.adapterDescription,
    corpusCaseCount: corpus.length,
    activeBlocksPerCase: corpus[0]!.activeVertices.length,
    elapsedMilliseconds: gpuResult.elapsedMilliseconds,
    maximumGpuBlockError,
    worstCaseId,
    worstVertex,
    validationError: gpuResult.validationError,
    uncapturedErrors: gpuResult.uncapturedErrors,
    cases: caseEvidence,
  };
  await testInfo.attach("phase0-canonical-gpu-equilibrium-oracle.json", {
    body: Buffer.from(JSON.stringify(evidence, null, 2)),
    contentType: "application/json",
  });
  testInfo.annotations.push({
    type: "webgpu-adapter",
    description: gpuResult.adapterDescription,
  });
  console.log(
    `P0-EC-04 canonical GPU corpus (${corpus.length} poses, ` +
      `${gpuResult.elapsedMilliseconds.toFixed(1)} ms): maximum block error ` +
      `${maximumGpuBlockError.toExponential(3)} at ${worstCaseId} vertex ${worstVertex}`,
  );
});
