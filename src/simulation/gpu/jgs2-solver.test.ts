import { afterEach, describe, expect, it, vi } from "vitest";

import {
  computeJGS2InitialDynamicSceneScale,
  DEFAULT_JGS2_STEP_SETTINGS,
  JGS2_DEGENERATE_DYNAMIC_SCENE_SCALE,
  JGS2_MAX_BATCH_FRAMES,
  JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION,
  JGS2GpuSolver,
  resolveJGS2StepSettings,
} from "./jgs2-solver";
import {
  computeJGS2DynamicOffsets,
  JGS2_MATERIAL_COROTATED_LINEAR,
  type JGS2GpuInput,
} from "./layout";

function createBatchTestSolver(globalizationEnabled = false): {
  readonly solver: JGS2GpuSolver;
  readonly counters: {
    submissions: number;
    uniformWrites: number;
    computePasses: number;
    finishes: number;
    timestampWriteIndices: number[];
    timestampPassLabels: string[];
    passLabels: string[];
  };
} {
  const counters = {
    submissions: 0,
    uniformWrites: 0,
    computePasses: 0,
    finishes: 0,
    timestampWriteIndices: [] as number[],
    timestampPassLabels: [] as string[],
    passLabels: [] as string[],
  };
  const computePass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: () => undefined,
    end: () => undefined,
  } as unknown as GPUComputePassEncoder;
  const commandBuffer = {} as GPUCommandBuffer;
  const encoder = {
    beginComputePass: (descriptor?: GPUComputePassDescriptor) => {
      counters.computePasses += 1;
      counters.passLabels.push(descriptor?.label ?? "");
      if (descriptor?.timestampWrites?.beginningOfPassWriteIndex !== undefined) {
        counters.timestampPassLabels.push(descriptor.label ?? "");
        counters.timestampWriteIndices.push(
          descriptor.timestampWrites.beginningOfPassWriteIndex,
        );
      }
      if (descriptor?.timestampWrites?.endOfPassWriteIndex !== undefined) {
        counters.timestampPassLabels.push(descriptor.label ?? "");
        counters.timestampWriteIndices.push(
          descriptor.timestampWrites.endOfPassWriteIndex,
        );
      }
      return computePass;
    },
    finish: () => {
      counters.finishes += 1;
      return commandBuffer;
    },
  } as unknown as GPUCommandEncoder;
  const device = {
    createCommandEncoder: () => encoder,
    queue: {
      writeBuffer: () => {
        counters.uniformWrites += 1;
      },
      submit: (commands: readonly GPUCommandBuffer[]) => {
        expect(commands).toEqual([commandBuffer]);
        counters.submissions += 1;
      },
    },
  } as unknown as GPUDevice;
  const pipeline = {} as GPUComputePipeline;
  const buffer = {} as GPUBuffer;
  const bindGroup = {} as GPUBindGroup;
  const dynamicOffsets = computeJGS2DynamicOffsets(4, 1, 1);
  const solver = Object.create(JGS2GpuSolver.prototype) as JGS2GpuSolver;
  Object.assign(solver, {
    destroyed: false,
    device,
    inputShape: { vertexCount: 4, tetCount: 1, cubatureK: 0, bodyCount: 1 },
    vertexCount: 4,
    tetCount: 1,
    cubatureK: 0,
    bodyCount: 1,
    dynamicOffsets,
    pipelines: {
      predict: pipeline,
      tetPolarRotation: pipeline,
      vertexPolarRotation: pipeline,
      solve: pipeline,
      assembledVertexEnergy: pipeline,
      candidateTetrahedron: pipeline,
      reduceCandidate: pipeline,
      applyCandidate: pipeline,
      convergenceGradient: pipeline,
      reduceConvergence: pipeline,
      copyPosition: pipeline,
      bodyHorizontalCorrection: pipeline,
      applyBodyHorizontalCorrection: pipeline,
      finalize: pipeline,
    },
    uniforms: {
      base: buffer,
      fromBToA: buffer,
      fromAToB: buffer,
      baseBindGroup: bindGroup,
      fromBToABindGroup: bindGroup,
      fromAToBBindGroup: bindGroup,
    },
    defaultSettings: DEFAULT_JGS2_STEP_SETTINGS,
    submittedSettings: DEFAULT_JGS2_STEP_SETTINGS,
    preprocessingTimestep: DEFAULT_JGS2_STEP_SETTINGS.timestep,
    sceneScale: 1,
    globalizationEnabled,
    globalizationRecordsAvailable: false,
    submittedIterations: 0,
  });
  return { solver, counters };
}

describe("JGS2 runtime settings", () => {
  it("preserves the existing demo stabilizers by default", () => {
    const settings = resolveJGS2StepSettings(
      DEFAULT_JGS2_STEP_SETTINGS,
      {},
    );

    expect(settings.parityMode).toBe(false);
    expect(settings.velocityDamping).toBeLessThan(1);
    expect(settings.contactTangentialDamping).toBeGreaterThan(0);
    expect(settings.horizontalBodyCorrection).toBe(true);
  });

  it("enforces parity-safe settings even when conflicting values are supplied", () => {
    const settings = resolveJGS2StepSettings(
      DEFAULT_JGS2_STEP_SETTINGS,
      {
        parityMode: true,
        velocityDamping: 0.25,
        contactTangentialDamping: 99,
        horizontalBodyCorrection: true,
      },
    );

    expect(settings.parityMode).toBe(true);
    expect(settings.velocityDamping).toBe(1);
    expect(settings.contactTangentialDamping).toBe(0);
    expect(settings.horizontalBodyCorrection).toBe(false);
  });
});

describe("JGS2 convergence scene scale", () => {
  it("excludes a distant fixed anchor from the dynamic AABB diagonal", () => {
    const scale = computeJGS2InitialDynamicSceneScale({
      vertexCount: 3,
      positions: new Float32Array([
        0, 0, 0, 1,
        3, 4, 0, 1,
        10_000, -20_000, 30_000, 1,
      ]),
      vertexInfo: new Uint32Array([
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 1, 0,
      ]),
    });

    expect(scale).toBe(5);
  });

  it("uses a finite positive fallback for a degenerate dynamic AABB", () => {
    const scale = computeJGS2InitialDynamicSceneScale({
      vertexCount: 2,
      positions: new Float32Array([
        2, 3, 4, 1,
        50_000, 60_000, 70_000, 1,
      ]),
      vertexInfo: new Uint32Array([
        0, 0, 0, 0,
        0, 0, 1, 0,
      ]),
    });

    expect(scale).toBe(JGS2_DEGENERATE_DYNAMIC_SCENE_SCALE);
    expect(Number.isFinite(scale)).toBe(true);
    expect(scale).toBeGreaterThan(0);
  });
});

describe("JGS2 batched frame submission", () => {
  it("encodes normalized-iteration frames into exactly one queue submission", () => {
    const { solver, counters } = createBatchTestSolver();

    solver.stepFrames(3, {
      iterations: 2,
      horizontalBodyCorrection: false,
    });

    expect(counters.submissions).toBe(1);
    expect(counters.finishes).toBe(1);
    expect(counters.uniformWrites).toBe(3);
    // predict + three (tet rotation, vertex rotation, solve) iterations + finalize
    expect(counters.computePasses).toBe(3 * 11);
    expect(solver.lastSubmittedIterationCount).toBe(3);
  });

  it("retains even iteration counts for an exact batched submission", () => {
    const { solver, counters } = createBatchTestSolver();

    solver.stepFramesExactIterations(2, 2, {
      horizontalBodyCorrection: false,
    });

    expect(counters.submissions).toBe(1);
    expect(counters.finishes).toBe(1);
    expect(counters.uniformWrites).toBe(3);
    // predict + two solve triplets + even-result copy + finalize
    expect(counters.computePasses).toBe(2 * 9);
    expect(solver.lastSubmittedIterationCount).toBe(2);
  });

  it("encodes the stable globalization and assembled-convergence passes in order", () => {
    const { solver, counters } = createBatchTestSolver(true);

    solver.stepFramesExactIterations(1, 2, {
      horizontalBodyCorrection: true,
    });

    expect(counters.submissions).toBe(1);
    // predict + one exact-f32 source preflight pair + two (rotations, local
    // solve, assembled gate, convergence) octets + even-result copy +
    // finalize. Stable settings forcibly disable the post-feasibility body
    // translation.
    expect(counters.computePasses).toBe(21);
    expect(counters.passLabels).toEqual([
      "jgs2-predict-pass",
      "jgs2-source-feasibility-preflight-pass",
      "jgs2-source-feasibility-preflight-reduction-pass",
      "jgs2-tet-polar-pass-0",
      "jgs2-vertex-polar-pass-0",
      "jgs2-solve-pass-0",
      "jgs2-candidate-tetrahedron-pass-0",
      "jgs2-reduce-candidate-pass-0",
      "jgs2-apply-candidate-pass-0",
      "jgs2-convergence-gradient-pass-0",
      "jgs2-reduce-convergence-pass-0",
      "jgs2-tet-polar-pass-1",
      "jgs2-vertex-polar-pass-1",
      "jgs2-solve-pass-1",
      "jgs2-candidate-tetrahedron-pass-1",
      "jgs2-reduce-candidate-pass-1",
      "jgs2-apply-candidate-pass-1",
      "jgs2-convergence-gradient-pass-1",
      "jgs2-reduce-convergence-pass-1",
      "jgs2-copy-even-result-to-canonical-position-pass",
      "jgs2-finalize-pass",
    ]);
    expect(solver.lastSubmittedSettings.horizontalBodyCorrection).toBe(false);
  });

  it("guards stable diagnostics until a production iteration has been submitted", async () => {
    const { solver, counters } = createBatchTestSolver(true);

    await expect(solver.readGlobalizationDiagnostics()).rejects.toThrow(
      /unavailable until a stable nonlinear iteration or diagnostic test kernel/,
    );
    expect(counters.submissions).toBe(0);

    solver.stepExactIterations(1);

    expect(
      (
        solver as unknown as {
          readonly globalizationRecordsAvailable: boolean;
        }
      ).globalizationRecordsAvailable,
    ).toBe(true);
    expect(counters.submissions).toBe(1);
  });

  it("bounds stable per-frame history without restricting the legacy path", () => {
    const stable = createBatchTestSolver(true);
    expect(() => stable.solver.stepExactIterations(65)).toThrow(/at most 64/);
    expect(stable.counters.submissions).toBe(0);

    const legacy = createBatchTestSolver(false);
    expect(() => legacy.solver.stepExactIterations(65)).not.toThrow();
    expect(legacy.counters.submissions).toBe(1);
  });

  it("bounds total stable encoded work without restricting the legacy path", () => {
    const boundaryFrames =
      JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION / 64;
    expect(Number.isSafeInteger(boundaryFrames)).toBe(true);

    const boundary = createBatchTestSolver(true);
    expect(() =>
      boundary.solver.stepFramesExactIterations(boundaryFrames, 64),
    ).not.toThrow();
    expect(boundary.counters.submissions).toBe(1);

    const rejected = createBatchTestSolver(true);
    expect(() =>
      rejected.solver.stepFramesExactIterations(boundaryFrames + 1, 64),
    ).toThrow(
      new RegExp(String(JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION)),
    );
    expect(rejected.counters.uniformWrites).toBe(0);
    expect(rejected.counters.submissions).toBe(0);

    const legacy = createBatchTestSolver(false);
    expect(() =>
      legacy.solver.stepFramesExactIterations(boundaryFrames + 1, 64),
    ).not.toThrow();
    expect(legacy.counters.submissions).toBe(1);
  });

  it("requires bounded positive batch sizes", () => {
    const { solver, counters } = createBatchTestSolver();

    expect(() => solver.stepFrames(0)).toThrow(/positive safe integer/);
    expect(() => solver.stepFrames(JGS2_MAX_BATCH_FRAMES + 1)).toThrow(
      new RegExp(String(JGS2_MAX_BATCH_FRAMES)),
    );
    expect(counters.submissions).toBe(0);
  });

  it("timestamps the real first and last passes without adding compute passes", () => {
    const { solver, counters } = createBatchTestSolver();

    solver.stepExactIterationsWithGpuTimestampWrites(
      2,
      {
        querySet: {} as GPUQuerySet,
        startWriteIndex: 4,
        endWriteIndex: 5,
      },
      { horizontalBodyCorrection: false },
    );

    expect(counters.submissions).toBe(1);
    expect(counters.timestampWriteIndices).toEqual([4, 5]);
    expect(counters.timestampPassLabels).toEqual([
      "jgs2-predict-pass",
      "jgs2-finalize-pass",
    ]);
    expect(counters.computePasses).toBe(9);
    expect(solver.lastSubmittedIterationCount).toBe(2);
  });
});

function createMinimalGpuInput(): JGS2GpuInput {
  return {
    vertexCount: 4,
    tetCount: 1,
    cubatureK: 0,
    positions: new Float32Array([
      0, 0, 0, 1,
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
    ]),
    vertexRest: new Float32Array([
      0, 0, 0, 1,
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
    ]),
    vertexColors: new Float32Array(16),
    vertexInfo: new Uint32Array([
      0, 1, 0, 0,
      1, 1, 0, 0,
      2, 1, 0, 0,
      3, 1, 0, 0,
    ]),
    tetIndices: new Uint32Array([0, 1, 2, 3]),
    tetInverseDm: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
    ]),
    tetMeta: new Float32Array([
      1 / 6,
      1,
      1,
      JGS2_MATERIAL_COROTATED_LINEAR,
    ]),
    tetRestStiffness: new Float32Array(12 * 12),
    adjacency: new Uint32Array([0, 0, 0, 0]),
    cubatureTetIds: new Uint32Array(0),
    cubatureWeights: new Float32Array(0),
    cubatureBasis: new Float32Array(0),
  };
}

type CreateFailureStage =
  | "storage"
  | "storage-map"
  | "pipeline"
  | "uniform"
  | "bind-group";

function createFailingGpuDevice(stage: CreateFailureStage): {
  readonly device: GPUDevice;
  readonly failure: Error;
  readonly buffers: ReadonlyArray<{
    readonly label: string;
    readonly destroy: ReturnType<typeof vi.fn>;
  }>;
} {
  vi.stubGlobal("GPUBufferUsage", {
    COPY_SRC: 1,
    COPY_DST: 2,
    STORAGE: 4,
    UNIFORM: 8,
    VERTEX: 16,
  });
  vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });

  const failure = new Error(`forced ${stage} creation failure`);
  const buffers: Array<{
    readonly label: string;
    readonly destroy: ReturnType<typeof vi.fn>;
  }> = [];
  let storageCreationCount = 0;
  let uniformCreationCount = 0;
  let pipelineCreationCount = 0;
  const device = {
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const label = descriptor.label ?? "";
      const isUniform = label.startsWith("jgs2-uniform");
      if (isUniform) {
        uniformCreationCount += 1;
        if (stage === "uniform" && uniformCreationCount === 2) {
          throw failure;
        }
      } else {
        storageCreationCount += 1;
        if (stage === "storage" && storageCreationCount === 3) {
          throw failure;
        }
      }
      const destroy = vi.fn();
      const mapped = new ArrayBuffer(Number(descriptor.size));
      const failMappedInitialization =
        !isUniform &&
        stage === "storage-map" &&
        storageCreationCount === 3;
      buffers.push({ label, destroy });
      return {
        destroy,
        getMappedRange: () => {
          if (failMappedInitialization) {
            throw failure;
          }
          return mapped;
        },
        unmap: () => undefined,
      } as unknown as GPUBuffer;
    },
    createBindGroupLayout: () => ({} as GPUBindGroupLayout),
    createPipelineLayout: () => ({} as GPUPipelineLayout),
    createShaderModule: () =>
      ({
        getCompilationInfo: async () => ({ messages: [] }),
      }) as unknown as GPUShaderModule,
    createComputePipelineAsync: async () => {
      pipelineCreationCount += 1;
      if (stage === "pipeline" && pipelineCreationCount === 1) {
        throw failure;
      }
      return {} as GPUComputePipeline;
    },
    createBindGroup: () => {
      if (stage === "bind-group") {
        throw failure;
      }
      return {} as GPUBindGroup;
    },
  } as unknown as GPUDevice;
  return { device, failure, buffers };
}

describe("JGS2 create-time cleanup", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each([
    ["storage", 2],
    ["storage-map", 3],
    ["pipeline", 6],
    ["uniform", 7],
    ["bind-group", 9],
  ] as const)(
    "destroys every allocated buffer after a %s creation failure",
    async (stage, expectedBufferCount) => {
      const { device, failure, buffers } = createFailingGpuDevice(stage);

      await expect(
        JGS2GpuSolver.create(device, createMinimalGpuInput()),
      ).rejects.toBe(failure);

      expect(buffers).toHaveLength(expectedBufferCount);
      for (const buffer of buffers) {
        expect(buffer.destroy, buffer.label).toHaveBeenCalledTimes(1);
      }
    },
  );
});

type ReadbackFailureStage = "encoder" | "submit" | "map";

function createReadbackFailureSolver(stage: ReadbackFailureStage): {
  readonly solver: JGS2GpuSolver;
  readonly failure: Error;
  readonly destroyReadback: ReturnType<typeof vi.fn>;
} {
  vi.stubGlobal("GPUBufferUsage", { COPY_DST: 1, MAP_READ: 2 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });

  const failure = new Error(`forced ${stage} readback failure`);
  const destroyReadback = vi.fn();
  const commandBuffer = {} as GPUCommandBuffer;
  const readback = {
    destroy: destroyReadback,
    mapAsync: async () => {
      if (stage === "map") {
        throw failure;
      }
    },
    getMappedRange: () => new ArrayBuffer(16),
    unmap: () => undefined,
  } as unknown as GPUBuffer;
  const encoder = {
    copyBufferToBuffer: () => undefined,
    finish: () => commandBuffer,
  } as unknown as GPUCommandEncoder;
  const device = {
    createBuffer: () => readback,
    createCommandEncoder: () => {
      if (stage === "encoder") {
        throw failure;
      }
      return encoder;
    },
    queue: {
      submit: () => {
        if (stage === "submit") {
          throw failure;
        }
      },
    },
  } as unknown as GPUDevice;
  const solver = Object.create(JGS2GpuSolver.prototype) as JGS2GpuSolver;
  Object.assign(solver, {
    destroyed: false,
    device,
    vertexCount: 1,
    dynamicOffsets: { posA: 0 },
    buffers: { dynamic: {} as GPUBuffer },
    diagnosticReadbackCount: 0,
  });
  return { solver, failure, destroyReadback };
}

describe("JGS2 diagnostic readback cleanup", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each(["encoder", "submit", "map"] as const)(
    "destroys its staging buffer after a %s failure",
    async (stage) => {
      const { solver, failure, destroyReadback } =
        createReadbackFailureSolver(stage);

      await expect(solver.readPositions()).rejects.toBe(failure);

      expect(destroyReadback).toHaveBeenCalledTimes(1);
      expect(solver.explicitDiagnosticReadbackCount).toBe(1);
    },
  );
});
