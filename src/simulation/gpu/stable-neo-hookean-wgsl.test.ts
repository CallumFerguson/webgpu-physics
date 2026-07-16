import { describe, expect, it } from "vitest";

import { stableNeoHookeanWgsl } from "./stable-neo-hookean-wgsl";

type Matrix3 = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

const IDENTITY: Matrix3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function dot(a: Matrix3, b: Matrix3): number {
  let result = 0;
  for (let index = 0; index < 9; index += 1) {
    result += a[index]! * b[index]!;
  }
  return result;
}

function determinant(f: Matrix3): number {
  return (
    f[0] * (f[4] * f[8] - f[5] * f[7]) -
    f[3] * (f[1] * f[8] - f[2] * f[7]) +
    f[6] * (f[1] * f[5] - f[2] * f[4])
  );
}

function cofactor(f: Matrix3): Matrix3 {
  return [
    f[4] * f[8] - f[5] * f[7],
    f[5] * f[6] - f[3] * f[8],
    f[3] * f[7] - f[4] * f[6],
    f[2] * f[7] - f[1] * f[8],
    f[0] * f[8] - f[2] * f[6],
    f[1] * f[6] - f[0] * f[7],
    f[1] * f[5] - f[2] * f[4],
    f[2] * f[3] - f[0] * f[5],
    f[0] * f[4] - f[1] * f[3],
  ];
}

function directionalCofactor(f: Matrix3, h: Matrix3): Matrix3 {
  return [
    h[4] * f[8] + f[4] * h[8] - h[5] * f[7] - f[5] * h[7],
    h[5] * f[6] + f[5] * h[6] - h[3] * f[8] - f[3] * h[8],
    h[3] * f[7] + f[3] * h[7] - h[4] * f[6] - f[4] * h[6],
    h[2] * f[7] + f[2] * h[7] - h[1] * f[8] - f[1] * h[8],
    h[0] * f[8] + f[0] * h[8] - h[2] * f[6] - f[2] * h[6],
    h[1] * f[6] + f[1] * h[6] - h[0] * f[7] - f[0] * h[7],
    h[1] * f[5] + f[1] * h[5] - h[2] * f[4] - f[2] * h[4],
    h[2] * f[3] + f[2] * h[3] - h[0] * f[5] - f[0] * h[5],
    h[0] * f[4] + f[0] * h[4] - h[1] * f[3] - f[1] * h[3],
  ];
}

function energyDensity(
  f: Matrix3,
  adjustedLambda: number,
  adjustedMu: number,
): number {
  const firstInvariant = dot(f, f);
  const determinantOffset = determinant(f) - 1;
  return (
    0.5 * adjustedMu * (firstInvariant - 3) -
    0.75 * adjustedMu * determinantOffset +
    0.5 * adjustedLambda * determinantOffset * determinantOffset -
    0.5 * adjustedMu * Math.log((firstInvariant + 1) * 0.25)
  );
}

function firstPiola(
  f: Matrix3,
  adjustedLambda: number,
  adjustedMu: number,
): Matrix3 {
  const firstInvariant = dot(f, f);
  const shearScale = adjustedMu * (1 - 1 / (firstInvariant + 1));
  const volumeScale = adjustedLambda * (determinant(f) - 1) - 0.75 * adjustedMu;
  const cof = cofactor(f);
  return f.map(
    (value, index) => shearScale * value + volumeScale * cof[index]!,
  ) as unknown as Matrix3;
}

function tangentProduct(
  f: Matrix3,
  h: Matrix3,
  adjustedLambda: number,
  adjustedMu: number,
): Matrix3 {
  const firstInvariant = dot(f, f);
  const invariantDenominator = firstInvariant + 1;
  const cof = cofactor(f);
  const dCof = directionalCofactor(f, h);
  const shearScale = adjustedMu * (1 - 1 / invariantDenominator);
  const directionalShearScale =
    (2 * adjustedMu * dot(f, h)) /
    (invariantDenominator * invariantDenominator);
  const volumeScale = adjustedLambda * (determinant(f) - 1) - 0.75 * adjustedMu;
  const directionalVolumeScale = adjustedLambda * dot(cof, h);
  return f.map(
    (value, index) =>
      shearScale * h[index]! +
      directionalShearScale * value +
      directionalVolumeScale * cof[index]! +
      volumeScale * dCof[index]!,
  ) as unknown as Matrix3;
}

function addScaled(f: Matrix3, h: Matrix3, scale: number): Matrix3 {
  return f.map(
    (value, index) => value + scale * h[index]!,
  ) as unknown as Matrix3;
}

function subtractScaledDifference(
  positive: Matrix3,
  negative: Matrix3,
  denominator: number,
): Matrix3 {
  return positive.map(
    (value, index) => (value - negative[index]!) / denominator,
  ) as unknown as Matrix3;
}

function relativeError(actual: Matrix3, expected: Matrix3): number {
  const difference = actual.map(
    (value, index) => value - expected[index]!,
  ) as unknown as Matrix3;
  return Math.sqrt(dot(difference, difference)) / Math.max(1, Math.sqrt(dot(expected, expected)));
}

function functionBody(name: string): string {
  const match = stableNeoHookeanWgsl.match(
    new RegExp(`fn ${name}\\([^]*?\\n\\}`),
  );
  expect(match, `WGSL function ${name}`).not.toBeNull();
  return match![0].replace(/\s+/g, " ");
}

describe("stable Neo-Hookean WGSL library", () => {
  it("names every declared function with the collision-safe prefix", () => {
    const names = [...stableNeoHookeanWgsl.matchAll(/\bfn\s+([a-zA-Z_]\w*)/g)].map(
      (match) => match[1],
    );
    expect(names).toEqual([
      "snh_frobenius_inner",
      "snh_determinant",
      "snh_cofactor",
      "snh_directional_cofactor",
      "snh_energy_density",
      "snh_first_piola",
      "snh_tangent_product",
    ]);
    expect(names.every((name) => name!.startsWith("snh_"))).toBe(true);
  });

  it("uses column-major polynomial cofactors without an inverse", () => {
    const body = functionBody("snh_cofactor");
    expect(body).toContain("cross(f[1], f[2])");
    expect(body).toContain("cross(f[2], f[0])");
    expect(body).toContain("cross(f[0], f[1])");
    expect(body).not.toMatch(/\binverse\s*\(/);

    const directionalBody = functionBody("snh_directional_cofactor");
    expect(directionalBody).toContain(
      "cross(h[1], f[2]) + cross(f[1], h[2])",
    );
    expect(directionalBody).toContain(
      "cross(h[2], f[0]) + cross(f[2], h[0])",
    );
    expect(directionalBody).toContain(
      "cross(h[0], f[1]) + cross(f[0], h[1])",
    );
  });

  it("contains the expanded, rest-normalized energy and exact derivatives", () => {
    const energy = functionBody("snh_energy_density");
    expect(energy).toContain("0.5 * adjusted_mu * (first_invariant - 3.0)");
    expect(energy).toContain("- 0.75 * adjusted_mu * determinant_offset");
    expect(energy).toContain(
      "+ 0.5 * adjusted_lambda * determinant_offset * determinant_offset",
    );
    expect(energy).toContain(
      "- 0.5 * adjusted_mu * log((first_invariant + 1.0) * 0.25)",
    );
    expect(energy).not.toContain("alpha");

    const piola = functionBody("snh_first_piola");
    expect(piola).toContain(
      "adjusted_mu * (1.0 - 1.0 / invariant_denominator)",
    );
    expect(piola).toContain("- 0.75 * adjusted_mu");
    expect(piola).toContain("shear_scale * f + volume_scale * cofactor");

    const tangent = functionBody("snh_tangent_product");
    expect(tangent).toContain("2.0 * adjusted_mu * snh_frobenius_inner(f, h)");
    expect(tangent).toContain(
      "adjusted_lambda * snh_frobenius_inner(cofactor, h)",
    );
    expect(tangent).toContain("volume_scale * directional_cofactor");
  });

  it("has zero rest energy/stress and remains finite at collapse and inversion", () => {
    const adjustedLambda = 75_000;
    const adjustedMu = 42_000;
    expect(energyDensity(IDENTITY, adjustedLambda, adjustedMu)).toBe(0);
    expect(Math.hypot(...firstPiola(IDENTITY, adjustedLambda, adjustedMu))).toBeLessThan(
      1e-11,
    );

    const collapsed: Matrix3 = [1, 0, 0, 0, 0, 0, 0, 0, 0];
    const inverted: Matrix3 = [-0.8, 0.1, 0, 0.2, 1.1, 0.3, 0, -0.1, 0.9];
    for (const deformation of [collapsed, inverted]) {
      expect(Number.isFinite(energyDensity(deformation, adjustedLambda, adjustedMu))).toBe(true);
      expect(
        firstPiola(deformation, adjustedLambda, adjustedMu).every(Number.isFinite),
      ).toBe(true);
    }
  });

  it("matches finite differences of energy and first Piola", () => {
    const adjustedLambda = 63_000;
    const adjustedMu = 37_000;
    const direction: Matrix3 = [
      0.17, -0.23, 0.09,
      0.31, -0.12, 0.26,
      -0.08, 0.14, 0.19,
    ];
    const deformations: readonly Matrix3[] = [
      IDENTITY,
      [1.13, 0.07, -0.03, -0.21, 0.91, 0.16, 0.08, -0.12, 1.04],
      [-0.72, 0.11, 0.04, 0.09, 1.08, -0.17, -0.06, 0.14, 0.83],
    ];
    const step = 1e-5;

    for (const deformation of deformations) {
      const positive = addScaled(deformation, direction, step);
      const negative = addScaled(deformation, direction, -step);
      const energyDerivative =
        (energyDensity(positive, adjustedLambda, adjustedMu) -
          energyDensity(negative, adjustedLambda, adjustedMu)) /
        (2 * step);
      expect(energyDerivative).toBeCloseTo(
        dot(firstPiola(deformation, adjustedLambda, adjustedMu), direction),
        4,
      );

      const finiteDifference = subtractScaledDifference(
        firstPiola(positive, adjustedLambda, adjustedMu),
        firstPiola(negative, adjustedLambda, adjustedMu),
        2 * step,
      );
      const exact = tangentProduct(
        deformation,
        direction,
        adjustedLambda,
        adjustedMu,
      );
      expect(relativeError(exact, finiteDifference)).toBeLessThan(2e-10);
    }
  });

  it("recovers the requested infinitesimal Hooke response after adjustment", () => {
    const hookeLambda = 31_000;
    const hookeMu = 19_000;
    const adjustedLambda = hookeLambda + (5 / 6) * hookeMu;
    const adjustedMu = (4 / 3) * hookeMu;
    const h: Matrix3 = [
      0.21, -0.08, 0.13,
      0.17, -0.11, 0.06,
      -0.04, 0.19, 0.27,
    ];
    const trace = h[0] + h[4] + h[8];
    const expected = h.map((value, index) => {
      const row = index % 3;
      const column = Math.floor(index / 3);
      return (
        hookeMu * (value + h[row * 3 + column]!) +
        (row === column ? hookeLambda * trace : 0)
      );
    }) as unknown as Matrix3;
    expect(
      relativeError(
        tangentProduct(IDENTITY, h, adjustedLambda, adjustedMu),
        expected,
      ),
    ).toBeLessThan(1e-14);
  });
});
