import { describe, expect, it } from "vitest";

import {
  DEFAULT_JGS2_STEP_SETTINGS,
  JGS2_MAX_BATCH_FRAMES,
  JGS2GpuSolver,
  resolveJGS2StepSettings,
} from "./jgs2-solver";
import { computeJGS2DynamicOffsets } from "./layout";

function createBatchTestSolver(): {
  readonly solver: JGS2GpuSolver;
  readonly counters: {
    submissions: number;
    uniformWrites: number;
    computePasses: number;
    finishes: number;
    timestampWriteIndices: number[];
    timestampPassLabels: string[];
  };
} {
  const counters = {
    submissions: 0,
    uniformWrites: 0,
    computePasses: 0,
    finishes: 0,
    timestampWriteIndices: [] as number[],
    timestampPassLabels: [] as string[],
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
