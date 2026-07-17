import { describe, expect, it } from "vitest";

import { jgs2IpcContactWgsl } from "./ipc-contact-wgsl";

function functionRegion(name: string, nextName?: string): string {
  const start = jgs2IpcContactWgsl.indexOf(`fn ${name}(`);
  expect(start, `WGSL function ${name}`).toBeGreaterThanOrEqual(0);
  const end = nextName
    ? jgs2IpcContactWgsl.indexOf(`fn ${nextName}(`, start + 1)
    : jgs2IpcContactWgsl.length;
  expect(end, `WGSL function following ${name}`).toBeGreaterThan(start);
  return jgs2IpcContactWgsl.slice(start, end);
}

function barrierValue(distance: number, activationDistance: number): number {
  if (!(distance > 0 && distance < activationDistance)) return 0;
  const offset = distance - activationDistance;
  return -(offset * offset) * Math.log(distance / activationDistance);
}

function barrierFirstDerivative(
  distance: number,
  activationDistance: number,
): number {
  if (!(distance > 0 && distance < activationDistance)) return 0;
  const offset = distance - activationDistance;
  return (
    -2 * offset * Math.log(distance / activationDistance) -
    (offset * offset) / distance
  );
}

function barrierSecondDerivative(
  distance: number,
  activationDistance: number,
): number {
  if (!(distance > 0 && distance < activationDistance)) return 0;
  const offset = distance - activationDistance;
  return (
    -2 * Math.log(distance / activationDistance) -
    (4 * offset) / distance +
    (offset * offset) / (distance * distance)
  );
}

function safeStepCap(
  sourceDistance: number,
  minDistance: number,
  displacementBound: number,
  safetyFactor: number,
): number {
  if (
    !Number.isFinite(sourceDistance) ||
    !Number.isFinite(minDistance) ||
    !Number.isFinite(displacementBound) ||
    !Number.isFinite(safetyFactor) ||
    minDistance < 0 ||
    sourceDistance <= minDistance ||
    displacementBound < 0
  ) {
    return 0;
  }
  if (displacementBound === 0) return 1;
  return Math.min(
    1,
    Math.max(
      0,
      Math.min(1, Math.max(0, safetyFactor)) *
        ((sourceDistance - minDistance) / displacementBound),
    ),
  );
}

function frictionF0(slipNorm: number, threshold: number): number {
  const slip = Math.max(slipNorm, 0);
  if (!(threshold > 0) || slip >= threshold) return slip;
  const normalized = slip / threshold;
  const normalizedSquared = normalized * normalized;
  return threshold * (
    -(normalizedSquared * normalized) / 3 + normalizedSquared + 1 / 3
  );
}

function frictionF1(slipNorm: number, threshold: number): number {
  const slip = Math.max(slipNorm, 0);
  if (!(threshold > 0) || slip >= threshold) return slip > 0 ? 1 : 0;
  const normalized = slip / threshold;
  return normalized * (2 - normalized);
}

function frictionDissipation(
  slipNorm: number,
  threshold: number,
  coefficient: number,
  normalForce: number,
): number {
  const slip = Math.max(slipNorm, 0);
  const scale = Math.max(coefficient, 0) * Math.max(normalForce, 0);
  if (!(threshold > 0)) return scale * slip;
  if (slip < threshold) {
    const normalized = slip / threshold;
    return scale * threshold * normalized * normalized * (1 - normalized / 3);
  }
  return scale * (slip - threshold / 3);
}

describe("IPC contact WGSL library", () => {
  it("names every WGSL declaration with the collision-safe prefix", () => {
    const declarations = [
      ...jgs2IpcContactWgsl.matchAll(
        /\b(?:const|struct|fn)\s+([a-zA-Z_]\w*)/g,
      ),
    ].map((match) => match[1]);
    expect(declarations.length).toBeGreaterThan(20);
    expect(declarations.every((name) => name!.startsWith("jgs2_ipc_"))).toBe(
      true,
    );
  });

  it("returns a shared weighted contact record for robust PT and EE queries", () => {
    const pointTriangle = functionRegion(
      "jgs2_ipc_point_triangle_contact",
      "jgs2_ipc_edge_edge_contact",
    );
    expect(pointTriangle).toContain("area_squared > area_threshold");
    expect(pointTriangle).toContain("candidate01");
    expect(pointTriangle).toContain("candidate12");
    expect(pointTriangle).toContain("candidate20");
    expect(pointTriangle).toContain(
      "vec4f(1.0, -barycentrics.x, -barycentrics.y, -barycentrics.z)",
    );

    const edgeEdge = functionRegion(
      "jgs2_ipc_edge_edge_contact",
      "jgs2_ipc_barrier_active",
    );
    expect(edgeEdge).toContain("a_degenerate && !b_degenerate");
    expect(edgeEdge).toContain("denominator > parallel_threshold");
    expect(edgeEdge).toContain("1.0 - parameter_a");
    expect(edgeEdge).toContain("-(1.0 - parameter_b)");
    expect(edgeEdge).not.toMatch(/\binverse\s*\(/);
  });

  it("implements the IPC barrier and its analytic derivatives", () => {
    const activationDistance = 2.3;
    for (const distance of [0.17, 0.61, 1.4, 2.1]) {
      const epsilon = 1e-6;
      const numericalFirst =
        (barrierValue(distance + epsilon, activationDistance) -
          barrierValue(distance - epsilon, activationDistance)) /
        (2 * epsilon);
      const numericalSecond =
        (barrierFirstDerivative(distance + epsilon, activationDistance) -
          barrierFirstDerivative(distance - epsilon, activationDistance)) /
        (2 * epsilon);
      expect(barrierFirstDerivative(distance, activationDistance)).toBeCloseTo(
        numericalFirst,
        5,
      );
      expect(barrierSecondDerivative(distance, activationDistance)).toBeCloseTo(
        numericalSecond,
        5,
      );
      expect(barrierValue(distance, activationDistance)).toBeGreaterThan(0);
      expect(barrierFirstDerivative(distance, activationDistance)).toBeLessThan(0);
      expect(barrierSecondDerivative(distance, activationDistance)).toBeGreaterThan(0);
    }
    expect(barrierValue(activationDistance, activationDistance)).toBe(0);
    expect(barrierFirstDerivative(3, activationDistance)).toBe(0);
    expect(barrierSecondDerivative(0, activationDistance)).toBe(0);

    const barrier = functionRegion(
      "jgs2_ipc_barrier_value",
      "jgs2_ipc_barrier_first_derivative",
    );
    expect(barrier).toContain("-(offset * offset) * log_ratio");
    expect(barrier).toContain("log(distance) - log(activation_distance)");
  });

  it("projects each per-vertex normal Hessian block to PSD", () => {
    const hessian = functionRegion(
      "jgs2_ipc_psd_normal_hessian_scalar",
      "jgs2_ipc_psd_normal_hessian",
    );
    expect(hessian).toContain(
      "barrier_stiffness * vertex_weight * vertex_weight",
    );
    expect(hessian).toContain("barrier_second_derivative <= 0.0");
    expect(hessian).toContain("return 0.0");

    for (const weight of [-1, -0.35, 0, 0.4, 1]) {
      const scale = 250_000 * weight * weight * Math.max(17.5, 0);
      expect(scale).toBeGreaterThanOrEqual(0);
    }
  });

  it("caps motion with the supplied distance Lipschitz bound", () => {
    const cases = [
      [0.04, 0.001, 0.2, 0.9],
      [0.2, 0.01, 0.05, 0.9],
      [0.02, 0.005, 3, 1],
    ] as const;
    for (const [distance, minimum, bound, safety] of cases) {
      const alpha = safeStepCap(distance, minimum, bound, safety);
      expect(alpha).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeLessThanOrEqual(1);
      expect(alpha * bound).toBeLessThanOrEqual(
        Math.min(bound, Math.min(1, Math.max(0, safety)) * (distance - minimum)) +
          1e-12,
      );
    }
    expect(safeStepCap(0.04, 0.001, 0, 0.9)).toBe(1);
    expect(safeStepCap(0.001, 0.001, 1, 0.9)).toBe(0);
    expect(safeStepCap(Number.NaN, 0, 1, 0.9)).toBe(0);
  });

  it("keeps lagged friction C1 and nonnegative through zero slip", () => {
    const threshold = 0.03;
    expect(frictionF0(0, threshold)).toBeCloseTo(threshold / 3, 14);
    expect(frictionF1(0, threshold)).toBe(0);
    expect(frictionF0(threshold, threshold)).toBeCloseTo(threshold, 14);
    expect(frictionF1(threshold, threshold)).toBe(1);

    for (const slip of [0.001, 0.008, 0.02, 0.029, 0.04]) {
      const epsilon = 1e-7;
      const numericalDerivative =
        (frictionF0(slip + epsilon, threshold) -
          frictionF0(slip - epsilon, threshold)) /
        (2 * epsilon);
      expect(frictionF1(slip, threshold)).toBeCloseTo(numericalDerivative, 6);
      const dissipation = frictionDissipation(slip, threshold, 0.4, 120);
      expect(dissipation).toBeGreaterThanOrEqual(0);
      expect(dissipation).toBeCloseTo(
        0.4 * 120 * (frictionF0(slip, threshold) - frictionF0(0, threshold)),
        12,
      );
    }
    expect(frictionDissipation(0, threshold, 0.4, 120)).toBe(0);

    const dissipation = functionRegion(
      "jgs2_ipc_lagged_friction_dissipation",
    );
    expect(dissipation).toContain("max(friction_coefficient, 0.0)");
    expect(dissipation).toContain("max(lagged_normal_force, 0.0)");
  });
});
