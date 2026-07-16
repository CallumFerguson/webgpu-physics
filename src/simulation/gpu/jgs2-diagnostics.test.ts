import { describe, expect, it } from "vitest";

import {
  JGS2_DIAGNOSTICS_UNIFORM_BYTES,
  computeJGS2DiagnosticsLayout,
  decodeJGS2GpuOracleDiagnostics,
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
      summary: 18,
      vec4Count: 23,
      byteLength: 368,
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
      2, 1, 0, 0,
      10, 20, 0, 3,
      4, 18, 30, 0,
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

    const pinned = layout.vertexRecords + 7;
    writeVec4(packed, pinned, [0, 0, 0, 0]);
    writeVec4(packed, pinned + 4, [0, 1, 0, 0]);

    writeVec4(packed, layout.summary, [0.5, 5, 0.25, -0.1]);
    writeVec4(packed, layout.summary + 1, [Math.sqrt(14), 0.4, 0.75, 1]);
    writeVec4(packed, layout.summary + 2, [14, 1, 1, 1]);
    writeVec4(packed, layout.summary + 3, [1, 1, 1, 1]);
    writeVec4(packed, layout.summary + 4, [Math.sqrt(14), 5, Math.sqrt(14) / 5, 1]);

    const decoded = decodeJGS2GpuOracleDiagnostics(packed, 2, 1, layout);
    expect(decoded.energy).toBeCloseTo(5.75, 7);
    expect(decoded.components).toEqual({
      inertia: 0.5,
      elasticity: 5,
      floorContact: 0.25,
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
