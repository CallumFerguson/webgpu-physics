import { describe, expect, it } from "vitest";

import {
  decodeDenseGpuEquilibriumOracleSteps,
  packDenseGpuEquilibriumOracleInput,
} from "./equilibrium-oracle";

describe("dense GPU equilibrium-oracle input packing", () => {
  it("validates, counts, and snapshots consecutive dense bases", () => {
    const gradient = new Float32Array([1, 2, 3]);
    const hessian = new Float32Array([
      4, 0, 0,
      0, 5, 0,
      0, 0, 6,
    ]);
    const bases = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      2, 0, 0,
      0, 2, 0,
      0, 0, 2,
    ]);
    const packed = packDenseGpuEquilibriumOracleInput({
      gradient,
      hessian,
      bases,
    });

    expect(packed.dimension).toBe(3);
    expect(packed.basisCount).toBe(2);
    expect(packed.gradient).not.toBe(gradient);
    expect(packed.hessian).not.toBe(hessian);
    expect(packed.bases).not.toBe(bases);
    gradient[0] = 99;
    expect(packed.gradient[0]).toBe(1);
  });

  it("rejects mismatched dimensions and non-finite values", () => {
    const validBasis = new Float32Array(9);
    expect(() =>
      packDenseGpuEquilibriumOracleInput({
        gradient: new Float32Array([1, 2, 3]),
        hessian: new Float32Array(8),
        bases: validBasis,
      }),
    ).toThrow(/Hessian/);
    expect(() =>
      packDenseGpuEquilibriumOracleInput({
        gradient: new Float32Array([1, 2, 3]),
        hessian: new Float32Array(9),
        bases: new Float32Array(8),
      }),
    ).toThrow(/bases/i);

    const nonFiniteHessian = new Float32Array(9);
    nonFiniteHessian[4] = Number.NaN;
    expect(() =>
      packDenseGpuEquilibriumOracleInput({
        gradient: new Float32Array([1, 2, 3]),
        hessian: nonFiniteHessian,
        bases: validBasis,
      }),
    ).toThrow(/hessian\[4\]/i);
  });
});

describe("dense GPU equilibrium-oracle result decoding", () => {
  it("removes one padding lane from each local three-vector", () => {
    expect(
      decodeDenseGpuEquilibriumOracleSteps(
        new Float32Array([1, 2, 3, 1, 4, 5, 6, 1]),
        2,
      ),
    ).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it("rejects malformed padded output", () => {
    expect(() =>
      decodeDenseGpuEquilibriumOracleSteps(new Float32Array(7), 2),
    ).toThrow(/8 values/);
    expect(() =>
      decodeDenseGpuEquilibriumOracleSteps(new Float32Array(0), 0),
    ).toThrow(/positive/);
    expect(() =>
      decodeDenseGpuEquilibriumOracleSteps(
        new Float32Array([1, 2, 3, 0]),
        1,
      ),
    ).toThrow(/without pivot regularization/);
  });
});
