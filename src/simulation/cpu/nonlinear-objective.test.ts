import { describe, expect, it } from "vitest";

import {
  centralDifferenceGradient,
  centralDifferenceHessian,
  relativeError,
} from "./finite-difference";
import {
  createNonlinearPerVertexObjective,
  evaluateNonlinearPerVertexObjective,
  nonlinearPerVertexObjectiveEnergyDelta,
  type NonlinearPerVertexObjectiveInput,
} from "./nonlinear-objective";
import type { TetrahedralMesh } from "./types";

const MESH: TetrahedralMesh = {
  positions: new Float64Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]),
  tetrahedra: new Uint32Array([0, 1, 2, 3]),
  materialIds: new Uint16Array([0]),
  fixed: new Uint8Array([1, 0, 0, 0]),
  bodyIds: new Uint16Array(4),
};

function emptyInput(): NonlinearPerVertexObjectiveInput {
  return {
    externalForces: new Float64Array(12),
    targetPositions: MESH.positions.slice(),
    targetStiffnesses: new Float64Array(4),
  };
}

describe("nonlinear per-vertex objective", () => {
  it("matches finite differences and its closed-form energy delta", () => {
    const input = emptyInput();
    input.externalForces.set([2.5, -1.25, 0.75], 3);
    input.targetPositions.set([1.2, -0.3, 0.15], 3);
    input.targetStiffnesses[1] = 37;
    const objective = createNonlinearPerVertexObjective(MESH, input);
    const position = new Float64Array([0.91, 0.14, -0.08]);
    const analytic = evaluateNonlinearPerVertexObjective(
      objective,
      1,
      position,
    );
    const energy = (coordinates: Float64Array) =>
      evaluateNonlinearPerVertexObjective(objective, 1, coordinates).energy;
    const gradient = (coordinates: Float64Array) =>
      evaluateNonlinearPerVertexObjective(objective, 1, coordinates).gradient;
    const expectedHessian = new Float64Array([
      37, 0, 0,
      0, 37, 0,
      0, 0, 37,
    ]);

    expect(
      relativeError(
        analytic.gradient,
        centralDifferenceGradient(energy, position),
      ),
    ).toBeLessThan(1e-8);
    expect(
      relativeError(
        expectedHessian,
        centralDifferenceHessian(gradient, position),
      ),
    ).toBeLessThan(1e-8);

    const displacement = new Float64Array([0.03, -0.02, 0.015]);
    const moved = Float64Array.from(
      position,
      (value, coordinate) => value + displacement[coordinate]!,
    );
    expect(
      nonlinearPerVertexObjectiveEnergyDelta(
        objective,
        1,
        position,
        displacement,
      ),
    ).toBeCloseTo(energy(moved) - energy(position), 13);
  });

  it("snapshots inputs and makes zero stiffness an exact release", () => {
    const input = emptyInput();
    input.targetPositions.set(
      [Number.MAX_VALUE, -Number.MAX_VALUE, Number.MAX_VALUE],
      6,
    );
    const objective = createNonlinearPerVertexObjective(MESH, input);
    input.externalForces.fill(123);
    input.targetPositions.fill(123);
    input.targetStiffnesses.fill(123);

    const position = new Float64Array([0.2, -0.1, 0.4]);
    const evaluation = evaluateNonlinearPerVertexObjective(
      objective,
      2,
      position,
    );
    expect(objective.active).toBe(false);
    expect(evaluation.energy).toBe(0);
    expect(evaluation.targetEnergy).toBe(0);
    expect(evaluation.gradient).toEqual(new Float64Array(3));
    expect(evaluation.targetGradient).toEqual(new Float64Array(3));
    expect(
      nonlinearPerVertexObjectiveEnergyDelta(
        objective,
        2,
        position,
        new Float64Array([1, 2, 3]),
      ),
    ).toBe(0);
  });

  it("is equivalent to putting a constant force in the paper predictor", () => {
    const input = emptyInput();
    const force = new Float64Array([2, -3, 4]);
    input.externalForces.set(force, 3);
    const objective = createNonlinearPerVertexObjective(MESH, input);
    const timestep = 0.125;
    const mass = 5;
    const inertia = mass / (timestep * timestep);
    const predictor = new Float64Array([0.8, -0.2, 0.4]);
    const shiftedPredictor = Float64Array.from(
      predictor,
      (value, coordinate) =>
        value +
        (timestep * timestep * force[coordinate]!) / mass,
    );
    const explicit = (position: Float64Array) => {
      let energy = evaluateNonlinearPerVertexObjective(
        objective,
        1,
        position,
      ).energy;
      const gradient = evaluateNonlinearPerVertexObjective(
        objective,
        1,
        position,
      ).gradient;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const difference = position[coordinate]! - predictor[coordinate]!;
        energy += 0.5 * inertia * difference * difference;
        gradient[coordinate] += inertia * difference;
      }
      return { energy, gradient };
    };
    const shifted = (position: Float64Array) => {
      let energy = 0;
      const gradient = new Float64Array(3);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const difference =
          position[coordinate]! - shiftedPredictor[coordinate]!;
        energy += 0.5 * inertia * difference * difference;
        gradient[coordinate] = inertia * difference;
      }
      return { energy, gradient };
    };
    const first = new Float64Array([0.9, -0.1, 0.2]);
    const second = new Float64Array([1.1, -0.35, 0.55]);

    expect(relativeError(explicit(first).gradient, shifted(first).gradient)).toBeLessThan(
      1e-14,
    );
    expect(relativeError(explicit(second).gradient, shifted(second).gradient)).toBeLessThan(
      1e-14,
    );
    expect(
      (explicit(first).energy - shifted(first).energy) -
        (explicit(second).energy - shifted(second).energy),
    ).toBeCloseTo(0, 12);
  });

  it("rejects malformed, non-finite, negative, and fixed-vertex data", () => {
    const short = emptyInput();
    expect(() =>
      createNonlinearPerVertexObjective(MESH, {
        ...short,
        externalForces: new Float64Array(11),
      }),
    ).toThrow(
      /expected 12/i,
    );

    const nonFinite = emptyInput();
    nonFinite.targetPositions[5] = Number.NaN;
    expect(() =>
      createNonlinearPerVertexObjective(MESH, nonFinite),
    ).toThrow(/must be finite/i);

    const negative = emptyInput();
    negative.targetStiffnesses[1] = -1;
    expect(() => createNonlinearPerVertexObjective(MESH, negative)).toThrow(
      /nonnegative/i,
    );

    const fixedForce = emptyInput();
    fixedForce.externalForces[0] = 1;
    expect(() =>
      createNonlinearPerVertexObjective(MESH, fixedForce),
    ).toThrow(/fixed vertex 0.*external force/i);

    const fixedTarget = emptyInput();
    fixedTarget.targetStiffnesses[0] = 1;
    expect(() =>
      createNonlinearPerVertexObjective(MESH, fixedTarget),
    ).toThrow(/fixed vertex 0.*quadratic target/i);
  });
});
