import { describe, expect, it } from "vitest";

import {
  fitNonnegativeColumnWeights,
  selectGreedyNonnegativeColumns,
} from "./cubature";

describe("exact nonnegative column fitting", () => {
  it("fits a caller-selected subset and reports its residual", () => {
    const fit = fitNonnegativeColumnWeights(
      [new Float64Array([1, 0]), new Float64Array([1, 1])],
      new Float64Array([2, 1]),
    );

    expect([...fit.weights]).toEqual([1, 1]);
    expect([...fit.residualVector]).toEqual([0, 0]);
    expect(fit.residualNorm).toBe(0);
    expect(fit.normalizedResidual).toBe(0);
  });

  it("keeps the exact fit nonnegative when an unconstrained coefficient would be negative", () => {
    const fit = fitNonnegativeColumnWeights(
      [new Float64Array([1, 0]), new Float64Array([1, 1])],
      new Float64Array([0, 1]),
    );

    expect([...fit.weights]).toEqual([0, 0.5]);
    expect([...fit.weights].every((weight) => weight >= 0)).toBe(true);
    expect(fit.normalizedResidual).toBeCloseTo(Math.sqrt(0.5), 14);
  });
});

describe("generic greedy nonnegative column selection", () => {
  it("selects an exact deterministic nonnegative representation", () => {
    const result = selectGreedyNonnegativeColumns({
      columns: [
        new Float64Array([1, 0, 0]),
        new Float64Array([0, 1, 0]),
        new Float64Array([0, 0, 1]),
      ],
      target: new Float64Array([2, 3, 0]),
      maximumColumns: 2,
    });

    expect(result.selectedColumnIndices).toEqual([1, 0]);
    expect([...result.weights]).toEqual([3, 2]);
    expect([...result.residualVector]).toEqual([0, 0, 0]);
    expect(result.targetNorm).toBeCloseTo(Math.sqrt(13), 14);
    expect(result.residualNorm).toBe(0);
    expect(result.normalizedResidual).toBe(0);
  });

  it("uses original column order for exact ties and ignores zero columns", () => {
    const first = selectGreedyNonnegativeColumns({
      columns: [
        new Float64Array([0, 0]),
        new Float64Array([1, 0]),
        new Float64Array([1, 0]),
      ],
      target: new Float64Array([1, 0]),
      maximumColumns: 1,
    });
    const second = selectGreedyNonnegativeColumns({
      columns: [
        new Float64Array([0, 0]),
        new Float64Array([1, 0]),
        new Float64Array([1, 0]),
      ],
      target: new Float64Array([1, 0]),
      maximumColumns: 1,
    });

    expect(first.selectedColumnIndices).toEqual([1]);
    expect(first.selectedColumnIndices).toEqual(second.selectedColumnIndices);
    expect([...first.weights]).toEqual([1]);
    expect(first.normalizedResidual).toBe(0);
  });

  it("reports the normalized residual without hiding a sample-budget error", () => {
    const result = selectGreedyNonnegativeColumns({
      columns: [
        new Float64Array([1, 0]),
        new Float64Array([0, 1]),
      ],
      target: new Float64Array([1, 1]),
      maximumColumns: 1,
    });

    expect(result.selectedColumnIndices).toEqual([0]);
    expect([...result.weights]).toEqual([1]);
    expect([...result.residualVector]).toEqual([0, 1]);
    expect(result.residualNorm).toBe(1);
    expect(result.normalizedResidual).toBeCloseTo(1 / Math.sqrt(2), 14);
  });

  it("uses the configured normalized stopping tolerance", () => {
    const input = {
      columns: [
        new Float64Array([1, 0]),
        new Float64Array([0, 1]),
      ],
      target: new Float64Array([1, 1]),
      maximumColumns: 2,
      refinementPasses: 0,
    } as const;
    const loose = selectGreedyNonnegativeColumns({
      ...input,
      normalizedResidualTolerance: 0.8,
    });
    const exact = selectGreedyNonnegativeColumns({
      ...input,
      normalizedResidualTolerance: 0,
    });

    expect(loose.selectedColumnIndices).toEqual([0]);
    expect(loose.normalizedResidual).toBeCloseTo(1 / Math.sqrt(2), 14);
    expect(exact.selectedColumnIndices).toEqual([0, 1]);
    expect(exact.normalizedResidual).toBe(0);
  });

  it("never introduces a negative weight", () => {
    const result = selectGreedyNonnegativeColumns({
      columns: [
        new Float64Array([-1, 0]),
        new Float64Array([0, 1]),
      ],
      target: new Float64Array([1, 1]),
      maximumColumns: 2,
    });

    expect(result.selectedColumnIndices).toEqual([1]);
    expect([...result.weights]).toEqual([1]);
    expect([...result.weights].every((weight) => weight >= 0)).toBe(true);
    expect(result.normalizedResidual).toBeCloseTo(1 / Math.sqrt(2), 14);
  });

  it("defines zero-target and missing-candidate residuals", () => {
    const zeroTarget = selectGreedyNonnegativeColumns({
      columns: [new Float64Array([1, 0])],
      target: new Float64Array([0, 0]),
      maximumColumns: 1,
    });
    const noCandidates = selectGreedyNonnegativeColumns({
      columns: [],
      target: new Float64Array([2, -1]),
      maximumColumns: 4,
    });

    expect(zeroTarget.selectedColumnIndices).toEqual([]);
    expect(zeroTarget.normalizedResidual).toBe(0);
    expect(noCandidates.selectedColumnIndices).toEqual([]);
    expect(noCandidates.residualNorm).toBeCloseTo(Math.sqrt(5), 14);
    expect(noCandidates.normalizedResidual).toBe(1);
  });

  it("rejects malformed dimensions, values, and selection settings", () => {
    const valid = {
      columns: [new Float64Array([1, 0])],
      target: new Float64Array([1, 0]),
      maximumColumns: 1,
    } as const;

    expect(() =>
      selectGreedyNonnegativeColumns({ ...valid, target: new Float64Array(0) }),
    ).toThrow(/target must not be empty/i);
    expect(() =>
      selectGreedyNonnegativeColumns({
        ...valid,
        columns: [new Float64Array([1])],
      }),
    ).toThrow(/must contain 2 entries/i);
    expect(() =>
      selectGreedyNonnegativeColumns({
        ...valid,
        columns: [new Float64Array([1, Number.NaN])],
      }),
    ).toThrow(/must be finite/i);
    expect(() =>
      selectGreedyNonnegativeColumns({
        ...valid,
        target: new Float64Array([1, Number.POSITIVE_INFINITY]),
      }),
    ).toThrow(/must be finite/i);
    expect(() =>
      selectGreedyNonnegativeColumns({ ...valid, maximumColumns: -1 }),
    ).toThrow(/maximumColumns/i);
    expect(() =>
      selectGreedyNonnegativeColumns({ ...valid, maximumColumns: 1.5 }),
    ).toThrow(/maximumColumns/i);
    expect(() =>
      selectGreedyNonnegativeColumns({
        ...valid,
        normalizedResidualTolerance: -0.01,
      }),
    ).toThrow(/normalizedResidualTolerance/i);
    expect(() =>
      selectGreedyNonnegativeColumns({ ...valid, refinementPasses: 0.5 }),
    ).toThrow(/refinementPasses/i);

    const tooManyColumns = Array.from({ length: 13 }, (_unused, column) => {
      const result = new Float64Array(13);
      result[column] = 1;
      return result;
    });
    expect(() =>
      selectGreedyNonnegativeColumns({
        columns: tooManyColumns,
        target: new Float64Array(13).fill(1),
        maximumColumns: 13,
      }),
    ).toThrow(/at most 12 selected columns/i);
  });
});
