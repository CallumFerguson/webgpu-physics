import { describe, expect, it } from "vitest";

import { buildScene, toJGS2GpuInput } from "../scenes";
import { diagnosticsFromState } from "./diagnostics";

const TEST_CONTEXT = {
  frame: 0,
  lastStepIterations: 0,
  runtime: {
    parityMode: true,
    velocityDamping: 1,
    contactTangentialDamping: 0,
    horizontalBodyCorrection: false,
  },
} as const;

describe("deterministic readback diagnostics", () => {
  it("reports momentum and honest validity flags", () => {
    const scene = buildScene("drop");
    const positions = toJGS2GpuInput(scene).positions.slice();
    const velocities = new Float32Array(positions.length);
    let totalMass = 0;
    const center = [0, 0, 0];
    for (let vertex = 0; vertex < scene.mesh.bodyIds.length; vertex += 1) {
      const mass = scene.lumpedMasses[vertex] ?? 0;
      totalMass += mass;
      for (let axis = 0; axis < 3; axis += 1) {
        center[axis] += mass * (positions[vertex * 4 + axis] ?? 0);
      }
    }
    for (let axis = 0; axis < 3; axis += 1) {
      center[axis] /= totalMass;
    }

    const translation = [2, -1, 0.5];
    const angularVelocity = [0.25, -0.5, 0.75];
    for (let vertex = 0; vertex < scene.mesh.bodyIds.length; vertex += 1) {
      const offset = vertex * 4;
      const rx = (positions[offset] ?? 0) - (center[0] ?? 0);
      const ry = (positions[offset + 1] ?? 0) - (center[1] ?? 0);
      const rz = (positions[offset + 2] ?? 0) - (center[2] ?? 0);
      velocities[offset] =
        (translation[0] ?? 0) +
        (angularVelocity[1] ?? 0) * rz -
        (angularVelocity[2] ?? 0) * ry;
      velocities[offset + 1] =
        (translation[1] ?? 0) +
        (angularVelocity[2] ?? 0) * rx -
        (angularVelocity[0] ?? 0) * rz;
      velocities[offset + 2] =
        (translation[2] ?? 0) +
        (angularVelocity[0] ?? 0) * ry -
        (angularVelocity[1] ?? 0) * rx;
    }

    const diagnostics = diagnosticsFromState(scene, positions, velocities, {
      frame: 3,
      lastStepIterations: 4,
      runtime: {
        parityMode: true,
        velocityDamping: 1,
        contactTangentialDamping: 0,
        horizontalBodyCorrection: false,
      },
    });

    expect(diagnostics.source).toBe("cpu-readback");
    expect(diagnostics.finite).toBe(true);
    expect(diagnostics.lastStepIterations).toBe(4);
    expect(diagnostics.totalLinearMomentumValid).toBe(true);
    expect(diagnostics.totalAngularMomentumValid).toBe(true);
    expect(diagnostics.bodies).toHaveLength(1);
    expect(diagnostics.bodies[0]!.mass).toBeCloseTo(totalMass, 10);
    for (let axis = 0; axis < 3; axis += 1) {
      expect(diagnostics.totalLinearMomentum[axis]).toBeCloseTo(
        totalMass * (translation[axis] ?? 0),
        4,
      );
      expect(diagnostics.bodies[0]!.linearMomentum[axis]).toBeCloseTo(
        diagnostics.totalLinearMomentum[axis],
        10,
      );
    }
    expect(
      Math.hypot(...diagnostics.bodies[0]!.angularMomentum),
    ).toBeGreaterThan(0);

    const [cx, cy, cz] = diagnostics.bodies[0]!.centerOfMass;
    const [px, py, pz] = diagnostics.totalLinearMomentum;
    const orbital = [cy * pz - cz * py, cz * px - cx * pz, cx * py - cy * px];
    for (let axis = 0; axis < 3; axis += 1) {
      expect(diagnostics.totalAngularMomentum[axis]).toBeCloseTo(
        (orbital[axis] ?? 0) +
          diagnostics.bodies[0]!.angularMomentum[axis]!,
        5,
      );
    }

    expect(diagnostics.minTetDeterminantValid).toBe(true);
    expect(diagnostics.minimumContactDistance).toBe(0);
    expect(diagnostics.minimumContactDistanceValid).toBe(false);
    expect(diagnostics.activeContactCount).toBe(0);
    expect(diagnostics.activeContactCountValid).toBe(false);
    expect(diagnostics.candidateBufferOverflow).toBe(false);
    expect(diagnostics.candidateBufferOverflowValid).toBe(false);
    expect(diagnostics.relativeResidual).toBe(0);
    expect(diagnostics.relativeResidualValid).toBe(false);
    expect(diagnostics.maximumUpdate).toBe(0);
    expect(diagnostics.maximumUpdateValid).toBe(false);
  });

  it("uses explicit finite sentinels for scenes without pins or tetrahedra", () => {
    const baseScene = buildScene("minimal");
    const scene = {
      ...baseScene,
      mesh: {
        ...baseScene.mesh,
        fixed: new Uint8Array(baseScene.mesh.fixed.length),
        tetrahedra: new Uint32Array(),
        materialIds: new Uint16Array(),
      },
    };
    const positions = toJGS2GpuInput(baseScene).positions.slice();
    const diagnostics = diagnosticsFromState(
      scene,
      positions,
      new Float32Array(positions.length),
      TEST_CONTEXT,
    );

    expect(diagnostics.finite).toBe(true);
    expect(diagnostics.pinnedMaxError).toBe(0);
    expect(diagnostics.pinnedMaxErrorValid).toBe(false);
    expect(Number.isFinite(diagnostics.pinnedMaxError)).toBe(true);
    expect(diagnostics.minTetDeterminant).toBe(0);
    expect(diagnostics.minTetDeterminantValid).toBe(false);
    expect(Number.isFinite(diagnostics.minTetDeterminant)).toBe(true);
  });

  it("marks the pinned maximum valid iff at least one vertex is fixed", () => {
    const scene = buildScene("minimal");
    const positions = toJGS2GpuInput(scene).positions.slice();
    const diagnostics = diagnosticsFromState(
      scene,
      positions,
      new Float32Array(positions.length),
      TEST_CONTEXT,
    );

    expect(scene.mesh.fixed.some((fixed) => fixed !== 0)).toBe(true);
    expect(diagnostics.pinnedMaxErrorValid).toBe(true);
    expect(Number.isFinite(diagnostics.pinnedMaxError)).toBe(true);
  });

  it("returns empty bodies and invalid finite momentum sentinels without mass", () => {
    const baseScene = buildScene("minimal");
    const scene = {
      ...baseScene,
      lumpedMasses: new Float64Array(baseScene.lumpedMasses.length),
    };
    const positions = toJGS2GpuInput(baseScene).positions.slice();
    const diagnostics = diagnosticsFromState(
      scene,
      positions,
      new Float32Array(positions.length),
      TEST_CONTEXT,
    );

    expect(diagnostics.finite).toBe(true);
    expect(diagnostics.bodies).toEqual([]);
    expect(diagnostics.totalLinearMomentum).toEqual([0, 0, 0]);
    expect(diagnostics.totalLinearMomentumValid).toBe(false);
    expect(diagnostics.totalAngularMomentum).toEqual([0, 0, 0]);
    expect(diagnostics.totalAngularMomentumValid).toBe(false);
    expect(diagnostics.totalLinearMomentum.every(Number.isFinite)).toBe(true);
    expect(diagnostics.totalAngularMomentum.every(Number.isFinite)).toBe(true);
  });
});
