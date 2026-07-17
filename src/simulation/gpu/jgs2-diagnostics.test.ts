import { describe, expect, it, vi } from "vitest";

import {
  JGS2_DIAGNOSTICS_UNIFORM_BYTES,
  JGS2GpuOracleEvaluator,
  computeJGS2DiagnosticsLayout,
  decodeJGS2GpuOracleDiagnostics,
  jgs2DiagnosticsShader,
  jgs2DiagnosticRelativeError,
  packJGS2DiagnosticsUniforms,
} from "./jgs2-diagnostics";

function writeVec4(
  target: Float32Array,
  vec4Offset: number,
  values: readonly [number, number, number, number],
): void {
  target.set(values, vec4Offset * 4);
}

describe("JGS2 GPU oracle diagnostic layout", () => {
  it("allocates rotations, metrics, vertex records, and the summary contiguously", () => {
    expect(computeJGS2DiagnosticsLayout(2, 1)).toEqual({
      tetRotation: 0,
      tetMetrics: 3,
      vertexRecords: 4,
      summary: 22,
      vec4Count: 29,
      byteLength: 464,
    });
  });

  it("packs integer offsets and f32 physics values into the WGSL ABI", () => {
    const layout = computeJGS2DiagnosticsLayout(2, 1);
    const packed = packJGS2DiagnosticsUniforms(2, 1, layout, {
      currentPositionOffset: 10,
      predictedPositionOffset: 20,
      finalUpdateOffset: 30,
      timestep: 1 / 60,
      floorHeight: -0.25,
      floorStiffness: 1234,
      rotationEpsilon: 1e-7,
      objectiveFlags: 3,
    });
    expect(packed.byteLength).toBe(JGS2_DIAGNOSTICS_UNIFORM_BYTES);
    const integers = new Uint32Array(
      packed.buffer,
      packed.byteOffset,
      packed.byteLength / 4,
    );
    const floats = new Float32Array(
      packed.buffer,
      packed.byteOffset,
      packed.byteLength / 4,
    );
    expect(Array.from(integers.subarray(0, 12))).toEqual([
      2, 1, 3, 0,
      10, 20, 0, 3,
      4, 22, 30, 0,
    ]);
    expect(floats[12]).toBeCloseTo(3600, 3);
    expect(floats[13]).toBeCloseTo(-0.25, 7);
    expect(floats[14]).toBeCloseTo(1234, 7);
    expect(floats[15]).toBeCloseTo(1e-7, 12);
  });

  it("decodes row-major Hessians, exact components, and validity flags", () => {
    const layout = computeJGS2DiagnosticsLayout(2, 1);
    const packed = new Float32Array(layout.vec4Count * 4);
    writeVec4(packed, layout.tetMetrics, [5, 0.75, 1, 0]);

    const active = layout.vertexRecords;
    writeVec4(packed, active, [1, 2, 3, 1]);
    writeVec4(packed, active + 1, [10, 11, 12, 0.5]);
    writeVec4(packed, active + 2, [20, 21, 22, 0.25]);
    writeVec4(packed, active + 3, [30, 31, 32, 0.4]);
    writeVec4(packed, active + 4, [14, 1, 3600, 1]);
    writeVec4(packed, active + 5, [1, 4, 9, -0.1]);
    writeVec4(packed, active + 6, [1, 0, 0, 0]);
    writeVec4(packed, active + 7, [-4, 1, -2, -3]);
    writeVec4(packed, active + 8, [0.5, -1, 1.5, 2]);

    const pinned = layout.vertexRecords + 9;
    writeVec4(packed, pinned, [0, 0, 0, 0]);
    writeVec4(packed, pinned + 4, [0, 1, 0, 0]);

    writeVec4(packed, layout.summary, [0.5, 5, 0.25, -0.1]);
    writeVec4(packed, layout.summary + 1, [Math.sqrt(14), 0.4, 0.75, 1]);
    writeVec4(packed, layout.summary + 2, [14, 1, 1, 1]);
    writeVec4(packed, layout.summary + 3, [1, 1, 1, 1]);
    writeVec4(packed, layout.summary + 4, [Math.sqrt(14), 5, Math.sqrt(14) / 5, 1]);
    writeVec4(packed, layout.summary + 5, [
      -3,
      2,
      Math.sqrt(21),
      Math.sqrt(3.5),
    ]);
    writeVec4(packed, layout.summary + 6, [1, 2, 3, 0]);

    const decoded = decodeJGS2GpuOracleDiagnostics(packed, 2, 1, layout);
    expect(decoded.energy).toBeCloseTo(4.75, 7);
    expect(decoded.components).toEqual({
      inertia: 0.5,
      elasticity: 5,
      externalForce: -3,
      quadraticTarget: 2,
      floorContact: 0.25,
    });
    expect(decoded.componentGradientNorms).toEqual({
      inertia: 1,
      material: 2,
      externalForce: Math.fround(Math.sqrt(21)),
      target: Math.fround(Math.sqrt(3.5)),
      contact: 3,
    });
    expect(decoded.gradientNorm).toBeCloseTo(Math.sqrt(14), 6);
    expect(decoded.residualNumerator).toBe(decoded.gradientNorm);
    expect(decoded.residualDenominator).toBe(5);
    expect(decoded.relativeResidual).toBeCloseTo(Math.sqrt(14) / 5, 6);
    expect(decoded.relativeResidualValid).toBe(true);
    expect(decoded.maximumUpdate).toBeCloseTo(0.4, 6);
    expect(decoded.minimumDeformationDeterminant).toBeCloseTo(0.75, 7);
    expect(decoded.finite).toBe(true);
    expect(decoded.activeVertexCount).toBe(1);
    expect(decoded.gradientValid).toBe(true);
    expect(decoded.maximumUpdateValid).toBe(true);
    expect(decoded.minimumDeformationDeterminantValid).toBe(true);
    expect(decoded.minimumContactDistance).toBeCloseTo(-0.1, 7);
    expect(decoded.minimumContactDistanceValid).toBe(true);
    expect(decoded.activeContactCount).toBe(1);
    expect(decoded.activeContactCountValid).toBe(true);
    expect(decoded.candidateBufferOverflow).toBe(false);
    expect(decoded.candidateBufferOverflowValid).toBe(false);
    expect(decoded.vertices[0]).toMatchObject({
      vertex: 0,
      active: true,
      gradient: [1, 2, 3],
      inertiaEnergy: 0.5,
      externalForceEnergy: -3,
      targetEnergy: 2,
      externalForceGradient: [-4, 1, -2],
      targetGradient: [0.5, -1, 1.5],
      floorContactEnergy: 0.25,
      updateMagnitudeValid: true,
      finite: true,
    });
    expect(decoded.vertices[0]!.updateMagnitude).toBeCloseTo(0.4, 6);
    expect(Array.from(decoded.vertices[0]!.localHessian)).toEqual([
      10, 11, 12,
      20, 21, 22,
      30, 31, 32,
    ]);
    expect(decoded.vertices[1]!.active).toBe(false);
    expect(decoded.tetrahedra[0]).toEqual({
      tetrahedron: 0,
      elasticityEnergy: 5,
      deformationDeterminant: 0.75,
      finite: true,
    });
  });

  it("uses explicit finite sentinels and false validity for an empty tet set", () => {
    const layout = computeJGS2DiagnosticsLayout(1, 0);
    const packed = new Float32Array(layout.vec4Count * 4);
    writeVec4(packed, layout.vertexRecords + 4, [0, 1, 0, 0]);
    writeVec4(packed, layout.summary + 1, [0, 0, 0, 1]);
    writeVec4(packed, layout.summary + 2, [0, 0, 0, 0]);
    writeVec4(packed, layout.summary + 3, [0, 0, 0, 0]);
    writeVec4(packed, layout.summary + 4, [0, 1, 0, 0]);

    const decoded = decodeJGS2GpuOracleDiagnostics(packed, 1, 0, layout);
    expect(decoded.minimumDeformationDeterminant).toBe(0);
    expect(decoded.minimumDeformationDeterminantValid).toBe(false);
    expect(decoded.gradientNorm).toBe(0);
    expect(decoded.gradientValid).toBe(false);
    expect(decoded.maximumUpdate).toBe(0);
    expect(decoded.maximumUpdateValid).toBe(false);
    expect(decoded.residualDenominator).toBe(1);
    expect(decoded.relativeResidual).toBe(0);
    expect(decoded.relativeResidualValid).toBe(false);
  });

  it("rejects objective activity bits outside the force/target ABI", () => {
    const layout = computeJGS2DiagnosticsLayout(1, 0);
    expect(() =>
      packJGS2DiagnosticsUniforms(1, 0, layout, {
        currentPositionOffset: 0,
        predictedPositionOffset: 1,
        finalUpdateOffset: 2,
        timestep: 1 / 60,
        floorHeight: 0,
        floorStiffness: 0,
        rotationEpsilon: 1e-7,
        objectiveFlags: 4,
      }),
    ).toThrow(/objectiveFlags/);
  });

  it("binds the objective ABI and never consumes a released target position", () => {
    expect(jgs2DiagnosticsShader).toContain(
      "@group(0) @binding(5)\nvar<storage, read> vertexObjectives",
    );
    expect(jgs2DiagnosticsShader).toContain(
      "@group(0) @binding(6)\nvar<storage, read_write> diagnosticData",
    );
    expect(jgs2DiagnosticsShader).toContain(
      "@group(0) @binding(7)\nvar<uniform> params",
    );

    const forceHelperStart = jgs2DiagnosticsShader.indexOf(
      "fn vertexExternalForce",
    );
    const forceHelperEnd = jgs2DiagnosticsShader.indexOf(
      "fn vertexTargetPositionStiffness",
      forceHelperStart,
    );
    const forceHelper = jgs2DiagnosticsShader.slice(
      forceHelperStart,
      forceHelperEnd,
    );
    expect(forceHelper.indexOf("if (objectiveForceEnabled())")).toBeLessThan(
      forceHelper.indexOf("vertexObjectives[vertex]"),
    );

    const targetHelperStart = forceHelperEnd;
    const targetHelperEnd = jgs2DiagnosticsShader.indexOf(
      "fn loadDiagnosticRotation",
      targetHelperStart,
    );
    const targetHelper = jgs2DiagnosticsShader.slice(
      targetHelperStart,
      targetHelperEnd,
    );
    expect(targetHelper.indexOf("if (objectiveTargetEnabled())")).toBeLessThan(
      targetHelper.indexOf("vertexObjectives[vertex]"),
    );

    const vertexPassStart = jgs2DiagnosticsShader.indexOf(
      "fn evaluateDiagnosticVertices",
    );
    const vertexPassEnd = jgs2DiagnosticsShader.indexOf(
      "fn reduceDiagnostics",
      vertexPassStart,
    );
    const vertexPass = jgs2DiagnosticsShader.slice(
      vertexPassStart,
      vertexPassEnd,
    );
    const activeTargetBranch = vertexPass.indexOf(
      "if (targetRecord.w > 0.0)",
    );
    expect(activeTargetBranch).toBeGreaterThanOrEqual(0);
    expect(vertexPass.indexOf("finiteVec3(targetRecord.xyz)")).toBeGreaterThan(
      activeTargetBranch,
    );
    expect(vertexPass.indexOf("current - targetRecord.xyz")).toBeGreaterThan(
      activeTargetBranch,
    );
  });
});

describe("JGS2 GPU oracle objective capacity", () => {
  it("runs inactive with a one-record dummy and rejects active flags before dispatch", async () => {
    vi.stubGlobal("GPUBufferUsage", {
      STORAGE: 1,
      COPY_SRC: 2,
      UNIFORM: 4,
      COPY_DST: 8,
      MAP_READ: 16,
    });
    vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });
    vi.stubGlobal("GPUMapMode", { READ: 1 });

    const queueWriteBuffer = vi.fn();
    const queueSubmit = vi.fn();
    const sourceBuffer = { size: 4 } as GPUBuffer;
    const objectiveDummyDestroy = vi.fn();
    const objectiveDummy = {
      size: 32,
      destroy: objectiveDummyDestroy,
    } as unknown as GPUBuffer;
    const computePass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
      end: vi.fn(),
    } as unknown as GPUComputePassEncoder;
    const device = {
      limits: {
        maxStorageBuffersPerShaderStage: 7,
        maxStorageBufferBindingSize: 1 << 20,
        maxBufferSize: 1 << 20,
      },
      createBuffer: (descriptor: GPUBufferDescriptor) => {
        const mapped = new ArrayBuffer(Number(descriptor.size));
        return {
          size: Number(descriptor.size),
          destroy: vi.fn(),
          mapAsync: vi.fn(async () => undefined),
          getMappedRange: vi.fn(() => mapped),
          unmap: vi.fn(),
        } as unknown as GPUBuffer;
      },
      createBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
      createPipelineLayout: vi.fn(() => ({} as GPUPipelineLayout)),
      createShaderModule: vi.fn(
        () =>
          ({
            getCompilationInfo: vi.fn(async () => ({ messages: [] })),
          }) as unknown as GPUShaderModule,
      ),
      createComputePipelineAsync: vi.fn(
        async () => ({} as GPUComputePipeline),
      ),
      createBindGroup: vi.fn(() => ({} as GPUBindGroup)),
      createCommandEncoder: vi.fn(
        () =>
          ({
            beginComputePass: vi.fn(() => computePass),
            copyBufferToBuffer: vi.fn(),
            finish: vi.fn(() => ({} as GPUCommandBuffer)),
          }) as unknown as GPUCommandEncoder,
      ),
      queue: {
        writeBuffer: queueWriteBuffer,
        submit: queueSubmit,
      },
    } as unknown as GPUDevice;
    let evaluator: JGS2GpuOracleEvaluator | undefined;

    try {
      evaluator = await JGS2GpuOracleEvaluator.create(device, 12, 0, {
        dynamic: sourceBuffer,
        vertices: sourceBuffer,
        tets: sourceBuffer,
        stiffness: sourceBuffer,
        adjacency: sourceBuffer,
        objectives: objectiveDummy,
      });
      const settings = {
        currentPositionOffset: 0,
        predictedPositionOffset: 12,
        finalUpdateOffset: 24,
        timestep: 1 / 60,
        floorHeight: 0,
        floorStiffness: 0,
        rotationEpsilon: 1e-7,
      };

      await expect(evaluator.evaluate(settings)).resolves.toMatchObject({
        components: { externalForce: 0, quadraticTarget: 0 },
      });
      const writesAfterInactiveEvaluation = queueWriteBuffer.mock.calls.length;
      expect(() =>
        evaluator!.evaluate({ ...settings, objectiveFlags: 1 }),
      ).toThrow(/full per-vertex objective source buffer/);
      expect(queueWriteBuffer).toHaveBeenCalledTimes(
        writesAfterInactiveEvaluation,
      );
      expect(queueSubmit).toHaveBeenCalledTimes(1);
      expect(objectiveDummyDestroy).not.toHaveBeenCalled();
    } finally {
      evaluator?.destroy();
      vi.unstubAllGlobals();
    }
  });
});

describe("JGS2 diagnostic comparison math", () => {
  it("uses the roadmap's symmetric, scale-safe relative error", () => {
    expect(jgs2DiagnosticRelativeError([1, 2], [1, 1])).toBeCloseTo(
      1 / Math.sqrt(5),
      12,
    );
    expect(jgs2DiagnosticRelativeError([3, 4], [0, 0])).toBe(1);
    expect(jgs2DiagnosticRelativeError([0, 0], [0, 0])).toBe(0);
  });

  it("rejects mismatched comparison dimensions", () => {
    expect(() => jgs2DiagnosticRelativeError([1], [1, 2])).toThrow(
      /matching lengths/,
    );
  });
});
