import { describe, expect, it } from "vitest";

import {
  buildStaticIpcContactCandidates,
  evaluateIpcBarrier,
  ipcFrictionF0,
  ipcFrictionF1,
} from "./ipc-contact";

describe("IPC contact scalar kernels", () => {
  it("evaluates the barrier and matches finite-difference derivatives", () => {
    const activationDistance = 1.2;
    const distance = 0.37;
    const evaluation = evaluateIpcBarrier(distance, activationDistance);
    const step = 1e-5;
    const value = (sample: number) =>
      evaluateIpcBarrier(sample, activationDistance).value;
    const finiteDifferenceFirst =
      (value(distance + step) - value(distance - step)) / (2 * step);
    const finiteDifferenceSecond =
      (value(distance + step) -
        2 * value(distance) +
        value(distance - step)) /
      (step * step);

    expect(evaluation.valid).toBe(true);
    expect(evaluation.active).toBe(true);
    expect(evaluation.value).toBeCloseTo(
      -Math.pow(distance - activationDistance, 2) *
        Math.log(distance / activationDistance),
      14,
    );
    expect(evaluation.firstDerivative).toBeCloseTo(
      finiteDifferenceFirst,
      8,
    );
    expect(evaluation.secondDerivative).toBeCloseTo(
      finiteDifferenceSecond,
      4,
    );

    expect(evaluateIpcBarrier(activationDistance, activationDistance)).toEqual({
      valid: true,
      active: false,
      value: 0,
      firstDerivative: 0,
      secondDerivative: 0,
    });
    expect(evaluateIpcBarrier(2, activationDistance).value).toBe(0);
    expect(evaluateIpcBarrier(0, activationDistance)).toEqual({
      valid: false,
      active: true,
      value: Number.POSITIVE_INFINITY,
      firstDerivative: Number.NEGATIVE_INFINITY,
      secondDerivative: Number.POSITIVE_INFINITY,
    });
    expect(() => evaluateIpcBarrier(Number.NaN, activationDistance)).toThrow(
      /finite/,
    );
    expect(() => evaluateIpcBarrier(distance, 0)).toThrow(/positive/);
  });

  it("uses matching C1-smoothed friction f0 and f1 branches", () => {
    const smoothingScale = 0.2;
    const slidingMagnitude = 0.1;
    const step = 1e-6;
    const finiteDifferenceDerivative =
      (ipcFrictionF0(slidingMagnitude + step, smoothingScale) -
        ipcFrictionF0(slidingMagnitude - step, smoothingScale)) /
      (2 * step);

    expect(ipcFrictionF0(0, smoothingScale)).toBeCloseTo(
      smoothingScale / 3,
      14,
    );
    expect(ipcFrictionF1(0, smoothingScale)).toBe(0);
    expect(ipcFrictionF1(slidingMagnitude, smoothingScale)).toBeCloseTo(
      0.75,
      14,
    );
    expect(ipcFrictionF1(slidingMagnitude, smoothingScale)).toBeCloseTo(
      finiteDifferenceDerivative,
      9,
    );
    expect(ipcFrictionF0(smoothingScale, smoothingScale)).toBe(
      smoothingScale,
    );
    expect(ipcFrictionF1(smoothingScale, smoothingScale)).toBe(1);
    expect(ipcFrictionF0(2 * smoothingScale, smoothingScale)).toBe(
      2 * smoothingScale,
    );
    expect(ipcFrictionF1(2 * smoothingScale, smoothingScale)).toBe(1);
    expect(() => ipcFrictionF0(-1, smoothingScale)).toThrow(/non-negative/);
    expect(() => ipcFrictionF1(0, 0)).toThrow(/positive/);
  });
});

describe("static IPC contact candidate enumeration", () => {
  it("is deterministic, GPU-packable, and filters self/one-ring pairs", () => {
    const forward = buildStaticIpcContactCandidates(6, {
      triangles: new Uint32Array([0, 1, 2, 3, 4, 5]),
      edges: new Uint32Array([
        0, 1, 1, 2, 2, 0,
        3, 4, 4, 5, 5, 3,
      ]),
    });
    const reorderedWithDuplicates = buildStaticIpcContactCandidates(6, {
      triangles: new Uint32Array([5, 3, 4, 2, 1, 0, 4, 5, 3]),
      edges: new Uint32Array([
        5, 4, 4, 3, 3, 5,
        2, 1, 1, 0, 0, 2,
        1, 2,
      ]),
    });

    expect(forward.vertexTriangleCount).toBe(6);
    expect(forward.edgeEdgeCount).toBe(9);
    expect(forward.vertexTriangleCandidates).toEqual([
      [0, 3, 4, 5],
      [1, 3, 4, 5],
      [2, 3, 4, 5],
      [3, 0, 1, 2],
      [4, 0, 1, 2],
      [5, 0, 1, 2],
    ]);
    expect(forward.edgeEdgeCandidates[0]).toEqual([0, 1, 3, 4]);
    expect(forward.edgeEdgeCandidates.at(-1)).toEqual([1, 2, 4, 5]);
    expect(forward.packedIndices.length).toBe((6 + 9) * 4);
    expect([...forward.packedIndices.slice(0, 8)]).toEqual([
      0, 3, 4, 5,
      1, 3, 4, 5,
    ]);
    expect(reorderedWithDuplicates).toEqual(forward);

    const connectedPatch = buildStaticIpcContactCandidates(4, {
      triangles: new Uint32Array([0, 1, 2, 2, 1, 3]),
      edges: new Uint32Array(0),
    });
    expect(connectedPatch.vertexTriangleCandidates).toEqual([]);
    expect(connectedPatch.edgeEdgeCandidates).toEqual([]);
    expect(connectedPatch.packedIndices).toHaveLength(0);
  });
});
