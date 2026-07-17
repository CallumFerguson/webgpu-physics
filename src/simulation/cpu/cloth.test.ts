import { describe, expect, it } from "vitest";
import { centralDifferenceGradient, relativeError } from "./finite-difference";
import {
  evaluateQuadraticDihedralBending,
  evaluateStVKTriangleMembrane,
  precomputeQuadraticDihedralRest,
  precomputeStVKTriangleRest,
  type StVKMembraneMaterial,
} from "./cloth";

const membraneMaterial: StVKMembraneMaterial = {
  youngModulus: 8_000,
  poissonRatio: 0.3,
  thickness: 0.02,
};

const restTriangle = new Float64Array([
  0, 0, 0,
  1, 0, 0,
  0.2, 0.8, 0,
]);

const restHinge = new Float64Array([
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
  1, -1, 0,
]);

describe("CPU StVK cloth reference", () => {
  it("has zero membrane and bending energy and gradient at rest", () => {
    const membrane = evaluateStVKTriangleMembrane(
      precomputeStVKTriangleRest(restTriangle),
      restTriangle,
      membraneMaterial,
    );
    expect(membrane.energy).toBeLessThan(1e-24);
    expect(Math.hypot(...membrane.gradient)).toBeLessThan(1e-12);
    expect(membrane.deformationGradient).toEqual(
      new Float64Array([1, 0, 0, 1, 0, 0]),
    );

    const bending = evaluateQuadraticDihedralBending(
      precomputeQuadraticDihedralRest(restHinge),
      restHinge,
      2.5,
    );
    expect(bending.energy).toBe(0);
    expect(bending.angleDifference).toBe(0);
    expect(bending.gradient).toEqual(new Float64Array(12));
  });

  it("matches finite differences for a deformed membrane triangle", () => {
    const rest = precomputeStVKTriangleRest(restTriangle);
    const positions = new Float64Array([
      -0.03, 0.02, 0.04,
      1.08, -0.04, 0.12,
      0.16, 0.91, -0.08,
    ]);
    const analytic = evaluateStVKTriangleMembrane(
      rest,
      positions,
      membraneMaterial,
    );
    const finite = centralDifferenceGradient(
      (trial) => evaluateStVKTriangleMembrane(rest, trial, membraneMaterial).energy,
      positions,
    );
    expect(relativeError(analytic.gradient, finite)).toBeLessThan(2e-8);
  });

  it("matches finite differences for a bent, nonplanar hinge", () => {
    const rest = precomputeQuadraticDihedralRest(restHinge);
    const positions = new Float64Array([
      -0.02, 0.01, 0.03,
      1.04, -0.03, -0.02,
      0.08, 0.94, 0.22,
      0.91, -1.07, -0.14,
    ]);
    const analytic = evaluateQuadraticDihedralBending(rest, positions, 3.25);
    const finite = centralDifferenceGradient(
      (trial) => evaluateQuadraticDihedralBending(rest, trial, 3.25).energy,
      positions,
    );
    expect(analytic.energy).toBeGreaterThan(0);
    expect(relativeError(analytic.gradient, finite)).toBeLessThan(2e-8);
  });

  it("rejects degenerate geometry and invalid material inputs", () => {
    expect(() =>
      precomputeStVKTriangleRest(new Float64Array(9)),
    ).toThrow(/nondegenerate/i);
    expect(() =>
      precomputeQuadraticDihedralRest(
        new Float64Array([
          0, 0, 0,
          1, 0, 0,
          0.5, 0, 0,
          1, -1, 0,
        ]),
      ),
    ).toThrow(/nondegenerate/i);
    expect(() =>
      evaluateStVKTriangleMembrane(
        precomputeStVKTriangleRest(restTriangle),
        restTriangle,
        { ...membraneMaterial, thickness: 0 },
      ),
    ).toThrow(/thickness/i);
    expect(() =>
      evaluateQuadraticDihedralBending(
        precomputeQuadraticDihedralRest(restHinge),
        restHinge,
        -1,
      ),
    ).toThrow(/nonnegative/i);
  });
});
