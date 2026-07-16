import { describe, expect, it } from "vitest";

import {
  createCompositeEnergyModel,
  createLinearExternalForceTerm,
  createQuadraticTargetTerm,
  type DifferentiableEnergyTerm,
} from "./energy";
import {
  centralDifferenceGradient,
  centralDifferenceHessian,
  relativeError,
} from "./finite-difference";

describe("composable differentiable energy contract", () => {
  it("combines material, force, constraint, and contact contributions", () => {
    const material: DifferentiableEnergyTerm = {
      id: "material.reference",
      kind: "material",
      dimension: 2,
      evaluate: (coordinates) => ({
        energy: 0.5 * (2 * coordinates[0]! ** 2 + 3 * coordinates[1]! ** 2),
        gradient: new Float64Array([
          2 * coordinates[0]!,
          3 * coordinates[1]!,
        ]),
        hessian: new Float64Array([2, 0, 0, 3]),
      }),
    };
    const model = createCompositeEnergyModel([
      material,
      createLinearExternalForceTerm({
        id: "external.gravity",
        force: new Float64Array([0.25, -0.5]),
      }),
      createQuadraticTargetTerm({
        id: "constraint.handle",
        kind: "constraint",
        target: new Float64Array([0.2, -0.1]),
        stiffness: new Float64Array([4, 0]),
      }),
      createQuadraticTargetTerm({
        id: "contact.reference",
        kind: "contact",
        target: new Float64Array([0, 0.3]),
        stiffness: new Float64Array([0, 5]),
      }),
    ]);
    const coordinates = new Float64Array([0.7, -0.4]);
    const evaluation = model.evaluate(coordinates);

    expect(evaluation.contributions.map(({ id, kind }) => [id, kind])).toEqual([
      ["material.reference", "material"],
      ["external.gravity", "external-force"],
      ["constraint.handle", "constraint"],
      ["contact.reference", "contact"],
    ]);
    expect(
      relativeError(
        evaluation.gradient,
        centralDifferenceGradient(model.energy, coordinates),
      ),
    ).toBeLessThan(1e-9);
    expect(
      relativeError(
        evaluation.hessian,
        centralDifferenceHessian(model.gradient, coordinates),
      ),
    ).toBeLessThan(1e-10);
  });

  it("rejects duplicate IDs, shape mismatches, and non-finite term output", () => {
    const force = createLinearExternalForceTerm({
      id: "external.force",
      force: new Float64Array([1, 2]),
    });
    expect(() => createCompositeEnergyModel([force, force])).toThrow(
      /Duplicate/,
    );
    expect(() =>
      createCompositeEnergyModel([
        force,
        createLinearExternalForceTerm({
          id: "external.other",
          force: new Float64Array([1]),
        }),
      ]),
    ).toThrow(/dimension/);

    const invalid: DifferentiableEnergyTerm = {
      id: "material.invalid",
      kind: "material",
      dimension: 2,
      evaluate: () => ({
        energy: Number.NaN,
        gradient: new Float64Array(2),
        hessian: new Float64Array(4),
      }),
    };
    const model = createCompositeEnergyModel([invalid]);
    expect(() => model.evaluate(new Float64Array(2))).toThrow(/invalid/);
  });
});
