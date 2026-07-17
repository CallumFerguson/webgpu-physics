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
  JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT,
  JGS2_OBJECTIVE_TARGET_ACTIVE_BIT,
  JGS2_UNIFORM_BYTES,
  JGS2_VERTEX_OBJECTIVE_BYTES,
  JGS2_VERTEX_OBJECTIVE_WORDS,
  type JGS2GpuInput,
} from "./layout";
import { JGS2GpuOracleEvaluator } from "./jgs2-diagnostics";

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
    objectiveWrites: Array<{
      readonly offset: number;
      readonly data: Float32Array;
    }>;
    uniformPayloads: Uint8Array[];
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
    objectiveWrites: [] as Array<{
      readonly offset: number;
      readonly data: Float32Array;
    }>,
    uniformPayloads: [] as Uint8Array[],
  };
  const computePass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: () => undefined,
    end: () => undefined,
  } as unknown as GPUComputePassEncoder;
  const commandBuffer = {} as GPUCommandBuffer;
  const objectiveBuffer = {} as GPUBuffer;
  const uniformBuffer = {} as GPUBuffer;
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
      writeBuffer: (
        destination: GPUBuffer,
        offset: number,
        data: GPUAllowSharedBufferSource,
      ) => {
        const view = data as ArrayBufferView;
        const bytes = new Uint8Array(
          view.buffer,
          view.byteOffset,
          view.byteLength,
        ).slice();
        if (destination === objectiveBuffer) {
          counters.objectiveWrites.push({
            offset,
            data: new Float32Array(bytes.buffer),
          });
        } else {
          counters.uniformWrites += 1;
          counters.uniformPayloads.push(bytes);
        }
      },
      submit: (commands: readonly GPUCommandBuffer[]) => {
        expect(commands).toEqual([commandBuffer]);
        counters.submissions += 1;
      },
    },
  } as unknown as GPUDevice;
  const pipeline = {} as GPUComputePipeline;
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
    buffers: {
      dynamic: {} as GPUBuffer,
      objectives: objectiveBuffer,
    },
    uniforms: {
      base: uniformBuffer,
      fromBToA: uniformBuffer,
      fromAToB: uniformBuffer,
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
    objectiveData: new Float32Array(
      (globalizationEnabled ? 4 : 1) * JGS2_VERTEX_OBJECTIVE_WORDS,
    ),
    pinnedVertices: new Uint8Array(4),
    objectiveFlagsValue: 0,
    activeExternalForceCount: 0,
    activeTargetCount: 0,
    objectiveRevisionValue: 0,
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

describe("JGS2 mutable objective API", () => {
  it("retains a full stable mirror but only one inert legacy binding record", () => {
    const { solver: stable } = createBatchTestSolver(true);
    const { solver: legacy } = createBatchTestSolver(false);
    const objectiveData = (solver: JGS2GpuSolver) =>
      (solver as unknown as { readonly objectiveData: Float32Array })
        .objectiveData;

    expect(objectiveData(stable)).toHaveLength(
      stable.vertexCount * JGS2_VERTEX_OBJECTIVE_WORDS,
    );
    expect(objectiveData(legacy)).toHaveLength(JGS2_VERTEX_OBJECTIVE_WORDS);
    expect(() => legacy.setExternalForce(0, [0, 0, 0])).toThrow(
      /stable Neo-Hookean production path/,
    );
  });

  it("updates sparse force and target records with O(1) activity bookkeeping", () => {
    const { solver, counters } = createBatchTestSolver(true);

    expect(solver.objectiveActivity).toEqual({
      flags: 0,
      externalForces: false,
      quadraticTargets: false,
      externalForceVertexCount: 0,
      quadraticTargetVertexCount: 0,
    });
    expect(solver.objectiveRevision).toBe(0);

    solver.setExternalForce(2, [1, -2, 3]);
    expect(counters.objectiveWrites.at(-1)).toEqual({
      offset: 2 * JGS2_VERTEX_OBJECTIVE_BYTES,
      data: new Float32Array([1, -2, 3, 0]),
    });
    expect(solver.objectiveActivity).toEqual({
      flags: JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT,
      externalForces: true,
      quadraticTargets: false,
      externalForceVertexCount: 1,
      quadraticTargetVertexCount: 0,
    });

    solver.setQuadraticTarget(1, [4, 5, 6], 7);
    expect(counters.objectiveWrites.at(-1)).toEqual({
      offset: JGS2_VERTEX_OBJECTIVE_BYTES + 16,
      data: new Float32Array([4, 5, 6, 7]),
    });
    expect(solver.objectiveActivity).toEqual({
      flags:
        JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT |
        JGS2_OBJECTIVE_TARGET_ACTIVE_BIT,
      externalForces: true,
      quadraticTargets: true,
      externalForceVertexCount: 1,
      quadraticTargetVertexCount: 1,
    });

    solver.setExternalForce(2, [0, 0, 0]);
    solver.releaseQuadraticTarget(1);

    expect(counters.objectiveWrites.at(-1)).toEqual({
      offset: JGS2_VERTEX_OBJECTIVE_BYTES + 16,
      data: new Float32Array([4, 5, 6, 0]),
    });
    expect(counters.uniformWrites).toBe(0);
    expect(counters.submissions).toBe(0);
    expect(solver.objectiveActivity).toEqual({
      flags: 0,
      externalForces: false,
      quadraticTargets: false,
      externalForceVertexCount: 0,
      quadraticTargetVertexCount: 0,
    });
    expect(solver.objectiveRevision).toBe(4);
  });

  it("replaces and clears complete force and target fields in single writes", () => {
    const { solver, counters } = createBatchTestSolver(true);
    const forces = new Float32Array([
      1, 0, 0,
      0, 0, 0,
      0, 2, 0,
      0, 0, 0,
    ]);
    const targets = new Float32Array([
      10, 11, 12,
      20, 21, 22,
      30, 31, 32,
      40, 41, 42,
    ]);
    const stiffnesses = new Float32Array([0, 3, 4, 0]);

    solver.replaceExternalForces(forces);
    solver.replaceQuadraticTargets(targets, stiffnesses);

    expect(counters.objectiveWrites).toHaveLength(2);
    expect(counters.objectiveWrites[0]).toMatchObject({ offset: 0 });
    expect(counters.objectiveWrites[0]!.data).toHaveLength(
      4 * JGS2_VERTEX_OBJECTIVE_WORDS,
    );
    expect(counters.objectiveWrites[1]).toMatchObject({ offset: 0 });
    expect(Array.from(counters.objectiveWrites[1]!.data.slice(8, 16))).toEqual([
      0, 0, 0, 0, 20, 21, 22, 3,
    ]);
    expect(solver.objectiveActivity).toMatchObject({
      externalForceVertexCount: 2,
      quadraticTargetVertexCount: 2,
      externalForces: true,
      quadraticTargets: true,
    });

    solver.clearExternalForces();
    solver.releaseAllQuadraticTargets();

    expect(counters.objectiveWrites).toHaveLength(4);
    expect(counters.objectiveWrites[2]).toMatchObject({ offset: 0 });
    expect(counters.objectiveWrites[3]).toMatchObject({ offset: 0 });
    for (let vertex = 0; vertex < 4; vertex += 1) {
      const base = vertex * JGS2_VERTEX_OBJECTIVE_WORDS;
      const record = counters.objectiveWrites[3]!.data;
      expect(Array.from(record.slice(base, base + 4))).toEqual([0, 0, 0, 0]);
      expect(record[base + 7]).toBe(0);
    }
    expect(solver.objectiveActivity).toMatchObject({
      flags: 0,
      externalForceVertexCount: 0,
      quadraticTargetVertexCount: 0,
    });
    expect(solver.objectiveRevision).toBe(4);
  });

  it("validates atomically and rejects active objectives on pinned vertices", () => {
    const { solver, counters } = createBatchTestSolver(true);
    (
      solver as unknown as { readonly pinnedVertices: Uint8Array }
    ).pinnedVertices[1] = 1;

    const invalidCalls = [
      () => solver.setExternalForce(1, [1, 0, 0]),
      () => solver.setQuadraticTarget(1, [0, 0, 0], 1),
      () => solver.setQuadraticTarget(0, [0, 0, 0], -1),
      () => solver.setExternalForce(4, [0, 0, 0]),
      () =>
        solver.replaceExternalForces(
          new Float32Array([
            0, 0, 0,
            1, 0, 0,
            0, 0, 0,
            0, 0, 0,
          ]),
        ),
      () => {
        const forces = new Float32Array(12);
        forces[11] = Number.POSITIVE_INFINITY;
        solver.replaceExternalForces(forces);
      },
      () =>
        solver.replaceQuadraticTargets(
          new Float32Array(12),
          new Float32Array([0, 0, -1, 0]),
        ),
    ];
    for (const invoke of invalidCalls) {
      expect(invoke).toThrow();
    }

    expect(counters.objectiveWrites).toHaveLength(0);
    expect(solver.objectiveRevision).toBe(0);
    expect(solver.objectiveActivity.flags).toBe(0);

    expect(() => solver.setExternalForce(1, [0, 0, 0])).not.toThrow();
    expect(() =>
      solver.setQuadraticTarget(1, [8, 9, 10], 0),
    ).not.toThrow();
    expect(counters.objectiveWrites).toHaveLength(2);
    expect(solver.objectiveActivity.flags).toBe(0);
  });

  it("uploads objective activity through reserved offsets2.w without resizing uniforms", () => {
    const { solver, counters } = createBatchTestSolver(true);
    solver.setExternalForce(0, [1, 0, 0]);
    solver.setQuadraticTarget(3, [1, 2, 3], 5);

    solver.stepExactIterations(1, { horizontalBodyCorrection: false });

    expect(counters.uniformPayloads).toHaveLength(3);
    for (const payload of counters.uniformPayloads) {
      expect(payload.byteLength).toBe(JGS2_UNIFORM_BYTES);
      expect(new Uint32Array(payload.buffer)[15]).toBe(
        JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT |
          JGS2_OBJECTIVE_TARGET_ACTIVE_BIT,
      );
    }
  });

  it("forwards the queue-coherent objective activity to exact diagnostics", async () => {
    const { solver } = createBatchTestSolver(true);
    solver.setExternalForce(0, [1, 0, 0]);
    solver.setQuadraticTarget(3, [1, 2, 3], 5);
    const diagnostics = { energy: 123 } as Awaited<
      ReturnType<JGS2GpuSolver["readOracleDiagnostics"]>
    >;
    let capturedObjectiveFlags: number | undefined;
    const evaluate = vi.fn(
      async (settings: { readonly objectiveFlags?: number }) => {
        capturedObjectiveFlags = settings.objectiveFlags;
        return diagnostics;
      },
    );
    const destroy = vi.fn();
    const create = vi
      .spyOn(JGS2GpuOracleEvaluator, "create")
      .mockResolvedValue({ evaluate, destroy } as unknown as JGS2GpuOracleEvaluator);
    Object.assign(solver, { diagnosticReadbackCount: 0 });

    try {
      await expect(solver.readOracleDiagnostics()).resolves.toBe(diagnostics);

      expect(create).toHaveBeenCalledTimes(1);
      const source = create.mock.calls[0]![3];
      expect(source.objectives).toBe(
        (
          solver as unknown as {
            readonly buffers: { readonly objectives: GPUBuffer };
          }
        ).buffers.objectives,
      );
      expect(evaluate).toHaveBeenCalledTimes(1);
      expect(capturedObjectiveFlags).toBe(
        JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT |
          JGS2_OBJECTIVE_TARGET_ACTIVE_BIT,
      );
      expect(solver.explicitDiagnosticReadbackCount).toBe(1);
    } finally {
      create.mockRestore();
    }
  });

  it("rejects every mutation on the legacy path and every access after destroy", () => {
    const legacy = createBatchTestSolver(false);
    const legacyCalls = [
      () => legacy.solver.setExternalForce(0, [0, 0, 0]),
      () => legacy.solver.replaceExternalForces(new Float32Array(12)),
      () => legacy.solver.clearExternalForces(),
      () => legacy.solver.setQuadraticTarget(0, [0, 0, 0], 0),
      () =>
        legacy.solver.replaceQuadraticTargets(
          new Float32Array(12),
          new Float32Array(4),
        ),
      () => legacy.solver.releaseQuadraticTarget(0),
      () => legacy.solver.releaseAllQuadraticTargets(),
    ];
    for (const invoke of legacyCalls) {
      expect(invoke).toThrow(/stable Neo-Hookean production path/);
    }
    expect(legacy.counters.objectiveWrites).toHaveLength(0);

    const destroyed = createBatchTestSolver(true);
    Object.assign(destroyed.solver, { destroyed: true });
    const destroyedCalls = [
      () => destroyed.solver.objectiveActivity,
      () => destroyed.solver.objectiveRevision,
      () => destroyed.solver.setExternalForce(0, [0, 0, 0]),
      () => destroyed.solver.replaceExternalForces(new Float32Array(12)),
      () => destroyed.solver.clearExternalForces(),
      () => destroyed.solver.setQuadraticTarget(0, [0, 0, 0], 0),
      () =>
        destroyed.solver.replaceQuadraticTargets(
          new Float32Array(12),
          new Float32Array(4),
        ),
      () => destroyed.solver.releaseQuadraticTarget(0),
      () => destroyed.solver.releaseAllQuadraticTargets(),
    ];
    for (const invoke of destroyedCalls) {
      expect(invoke).toThrow(/destroyed/);
    }
    expect(destroyed.counters.objectiveWrites).toHaveLength(0);
  });
});

describe("JGS2 batched frame submission", () => {
  const stopAfterConvergenceFlags = (
    payloads: readonly Uint8Array[],
  ): number[] =>
    payloads.map(
      (payload) =>
        new Float32Array(
          payload.buffer,
          payload.byteOffset,
          payload.byteLength / Float32Array.BYTES_PER_ELEMENT,
        )[43]!,
    );

  it("enables GPU convergence stopping only for stable production stepping", () => {
    const stable = createBatchTestSolver(true);
    stable.solver.step({ iterations: 2 });
    expect(stable.solver.lastSubmittedIterationCount).toBe(3);
    expect(stopAfterConvergenceFlags(stable.counters.uniformPayloads)).toEqual([
      1, 1, 1,
    ]);

    const legacy = createBatchTestSolver(false);
    legacy.solver.step({ iterations: 2 });
    expect(stopAfterConvergenceFlags(legacy.counters.uniformPayloads)).toEqual([
      0, 0, 0,
    ]);
  });

  it("keeps exact-iteration APIs eligible to execute and record full history", () => {
    const single = createBatchTestSolver(true);
    single.solver.stepExactIterations(3);
    expect(single.solver.lastSubmittedIterationCount).toBe(3);
    expect(stopAfterConvergenceFlags(single.counters.uniformPayloads)).toEqual([
      0, 0, 0,
    ]);

    const batched = createBatchTestSolver(true);
    batched.solver.stepFramesExactIterations(2, 2);
    expect(batched.solver.lastSubmittedIterationCount).toBe(2);
    expect(stopAfterConvergenceFlags(batched.counters.uniformPayloads)).toEqual([
      0, 0, 0,
    ]);

    const timestamped = createBatchTestSolver(true);
    timestamped.solver.stepExactIterationsWithGpuTimestampWrites(2, {
      querySet: {} as GPUQuerySet,
      startWriteIndex: 2,
      endWriteIndex: 3,
    });
    expect(stopAfterConvergenceFlags(timestamped.counters.uniformPayloads)).toEqual([
      0, 0, 0,
    ]);
  });

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
    // solve, assembled gate, convergence) octets + even-result copy + the two
    // objective-free free-body conservation passes + finalize.
    expect(counters.computePasses).toBe(23);
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
      "jgs2-body-horizontal-correction-pass",
      "jgs2-apply-body-horizontal-correction-pass",
      "jgs2-finalize-pass",
    ]);
    expect(solver.lastSubmittedSettings.horizontalBodyCorrection).toBe(true);
  });

  it("disables stable free-body correction while an objective is active", () => {
    const { solver, counters } = createBatchTestSolver(true);
    solver.setQuadraticTarget(0, [0.25, 0.5, -0.25], 1_000);

    solver.stepExactIterations(1, { horizontalBodyCorrection: true });

    expect(counters.passLabels).not.toContain(
      "jgs2-body-horizontal-correction-pass",
    );
    expect(counters.passLabels).not.toContain(
      "jgs2-apply-body-horizontal-correction-pass",
    );
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

describe("JGS2 objective creation requirements", () => {
  it("requires seven storage-buffer bindings", async () => {
    const device = {
      limits: { maxStorageBuffersPerShaderStage: 6 },
    } as unknown as GPUDevice;

    await expect(
      JGS2GpuSolver.create(device, createMinimalGpuInput()),
    ).rejects.toThrow(/requires seven storage buffers/);
  });

  it("rejects initially active objectives on the legacy material path", async () => {
    const input: JGS2GpuInput = {
      ...createMinimalGpuInput(),
      objectives: {
        externalForces: new Float32Array([
          1, 0, 0,
          0, 0, 0,
          0, 0, 0,
          0, 0, 0,
        ]),
      },
    };
    const device = {
      limits: { maxStorageBuffersPerShaderStage: 7 },
    } as unknown as GPUDevice;

    await expect(JGS2GpuSolver.create(device, input)).rejects.toThrow(
      /require the stable Neo-Hookean production path/,
    );
  });
});

type CreateFailureStage =
  | "storage"
  | "storage-map"
  | "pipeline"
  | "uniform"
  | "bind-group";

function createFailingGpuDevice(stage: CreateFailureStage): {
  readonly device: GPUDevice;
  readonly failure: Error;
  readonly bindGroupLayoutEntries: GPUBindGroupLayoutEntry[];
  readonly bindGroupEntries: GPUBindGroupEntry[];
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
  const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
  const bindGroupEntries: GPUBindGroupEntry[] = [];
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
        label,
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
    createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => {
      bindGroupLayoutEntries.push(...descriptor.entries);
      return {} as GPUBindGroupLayout;
    },
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
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => {
      bindGroupEntries.push(...descriptor.entries);
      if (stage === "bind-group") {
        throw failure;
      }
      return {} as GPUBindGroup;
    },
  } as unknown as GPUDevice;
  return {
    device,
    failure,
    bindGroupLayoutEntries,
    bindGroupEntries,
    buffers,
  };
}

describe("JGS2 create-time cleanup", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each([
    ["storage", 2],
    ["storage-map", 3],
    ["pipeline", 7],
    ["uniform", 8],
    ["bind-group", 10],
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

  it("binds the objective storage slot at 6 and moves uniforms to 7", async () => {
    const {
      device,
      failure,
      bindGroupLayoutEntries,
      bindGroupEntries,
      buffers,
    } = createFailingGpuDevice("bind-group");

    await expect(
      JGS2GpuSolver.create(device, createMinimalGpuInput()),
    ).rejects.toBe(failure);

    expect(bindGroupLayoutEntries.map((entry) => entry.binding)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7,
    ]);
    expect(bindGroupLayoutEntries[6]!.buffer?.type).toBe(
      "read-only-storage",
    );
    expect(bindGroupLayoutEntries[7]!.buffer?.type).toBe("uniform");
    expect(bindGroupLayoutEntries[7]!.buffer?.minBindingSize).toBe(
      JGS2_UNIFORM_BYTES,
    );

    const objectiveBinding = bindGroupEntries.find(
      (entry) => entry.binding === 6,
    );
    expect(
      (
        (objectiveBinding?.resource as GPUBufferBinding)
          .buffer as unknown as { readonly label: string }
      ).label,
    ).toBe("jgs2-objectives");
    expect(buffers.some((buffer) => buffer.label === "jgs2-objectives")).toBe(
      true,
    );
    expect(bindGroupEntries.some((entry) => entry.binding === 7)).toBe(true);
  });
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
