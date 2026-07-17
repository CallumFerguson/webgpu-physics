import { describe, expect, it } from "vitest";

import {
  JGS2_GLOBALIZATION_ARMIJO_C1,
  JGS2_GLOBALIZATION_BACKTRACK_FACTOR,
  JGS2_GLOBALIZATION_DETERMINANT_FLOOR,
  JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_NON_DESCENT,
  JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
  JGS2_GLOBALIZATION_LOCAL_STATUS_SHIFT_LIMIT,
  JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT,
  JGS2_GLOBALIZATION_MAX_BACKTRACKS,
  JGS2_GLOBALIZATION_MAX_NORMALIZED_SHIFT,
  JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER,
  JGS2_GLOBALIZATION_RELATIVE_EIGENVALUE_FLOOR,
} from "./jgs2-globalization";
import {
  JGS2_GLOBALIZATION_STATUS_CODES,
  jgs2GlobalizationWgsl,
} from "./jgs2-globalization-wgsl";

function wgslFloatLiteral(value: number): string {
  if (!Number.isFinite(value)) {
    throw new RangeError("WGSL policy constants must be finite.");
  }
  const rendered = value.toString();
  return Number.isInteger(value) ? `${rendered}.0` : rendered;
}

function functionRegion(name: string, nextName?: string): string {
  const start = jgs2GlobalizationWgsl.indexOf(`fn ${name}(`);
  expect(start, `WGSL function ${name}`).toBeGreaterThanOrEqual(0);
  const end = nextName
    ? jgs2GlobalizationWgsl.indexOf(`fn ${nextName}(`, start + 1)
    : jgs2GlobalizationWgsl.length;
  expect(end, `WGSL function following ${name}`).toBeGreaterThan(start);
  return jgs2GlobalizationWgsl.slice(start, end);
}

describe("JGS2 globalization WGSL library", () => {
  it("freezes the Phase 1 shift, Armijo, feasibility, and trial constants", () => {
    const floatPolicies = [
      [
        "jgs2_globalization_position_resolution_multiplier",
        JGS2_GLOBALIZATION_POSITION_RESOLUTION_MULTIPLIER,
      ],
      [
        "jgs2_globalization_relative_eigenvalue_floor",
        JGS2_GLOBALIZATION_RELATIVE_EIGENVALUE_FLOOR,
      ],
      [
        "jgs2_globalization_max_normalized_shift",
        JGS2_GLOBALIZATION_MAX_NORMALIZED_SHIFT,
      ],
      ["jgs2_globalization_armijo_c1", JGS2_GLOBALIZATION_ARMIJO_C1],
      [
        "jgs2_globalization_backtrack_factor",
        JGS2_GLOBALIZATION_BACKTRACK_FACTOR,
      ],
      [
        "jgs2_globalization_determinant_floor",
        JGS2_GLOBALIZATION_DETERMINANT_FLOOR,
      ],
    ] as const;
    for (const [name, value] of floatPolicies) {
      expect(jgs2GlobalizationWgsl).toContain(
        `${name}: f32 = ${wgslFloatLiteral(value)}`,
      );
    }
    expect(jgs2GlobalizationWgsl).toContain(
      `jgs2_globalization_max_backtracks: u32 = ` +
        `${JGS2_GLOBALIZATION_MAX_BACKTRACKS}u`,
    );
    expect(jgs2GlobalizationWgsl).toContain(
      `jgs2_globalization_trial_count: u32 = ` +
        `${JGS2_GLOBALIZATION_MAX_BACKTRACKS + 1}u`,
    );
  });

  it("exports stable, non-overlapping solve status codes", () => {
    expect(JGS2_GLOBALIZATION_STATUS_CODES).toEqual({
      accepted: JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
      zeroGradient: JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT,
      invalidInput: JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
      shiftLimitExceeded: JGS2_GLOBALIZATION_LOCAL_STATUS_SHIFT_LIMIT,
      factorizationFailed: JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED,
      nonDescentDirection: JGS2_GLOBALIZATION_LOCAL_STATUS_NON_DESCENT,
    });
    expect(
      new Set(Object.values(JGS2_GLOBALIZATION_STATUS_CODES)).size,
    ).toBe(Object.keys(JGS2_GLOBALIZATION_STATUS_CODES).length);
    for (const [name, code] of Object.entries(
      JGS2_GLOBALIZATION_STATUS_CODES,
    )) {
      const snakeName = name.replace(
        /[A-Z]/g,
        (letter) => `_${letter.toLowerCase()}`,
      );
      expect(jgs2GlobalizationWgsl).toContain(
        `jgs2_globalization_status_${snakeName}: u32 =\n  ${code}u`,
      );
    }
  });

  it("prefixes every declared WGSL symbol", () => {
    const declarations = [
      ...jgs2GlobalizationWgsl.matchAll(
        /\b(?:const|struct|fn)\s+([a-zA-Z_]\w*)/g,
      ),
    ].map((match) => match[1]);
    expect(declarations.length).toBeGreaterThan(20);
    expect(
      declarations.every((name) => name!.startsWith("jgs2_globalization_")),
    ).toBe(true);
  });

  it("normalizes before the closed-form symmetric spectrum calculation", () => {
    const classification = functionRegion(
      "jgs2_globalization_classify_shift",
      "jgs2_globalization_scaled_length3",
    );
    expect(classification).toContain(
      "jgs2_globalization_frobenius_over_sqrt_three(symmetric)",
    );
    expect(classification).toContain(
      "let normalized_hessian = mat3x3f(",
    );
    expect(classification).toContain(
      "jgs2_globalization_relative_eigenvalue_floor -",
    );
    expect(classification).toContain(
      "normalized_shift > jgs2_globalization_max_normalized_shift",
    );
  });

  it("uses an unclamped Cholesky solve and fails closed on every pivot", () => {
    const solve = functionRegion(
      "jgs2_globalization_solve",
      "jgs2_globalization_log_one_plus",
    );
    expect(solve).toContain("let l00 = sqrt(pivot0)");
    expect(solve).toContain("let l11 = sqrt(pivot1)");
    expect(solve).toContain("let l22 = sqrt(pivot2)");
    expect(solve).not.toMatch(/sqrt\s*\(\s*max\s*\(/);
    expect(solve).not.toMatch(/max\s*\(\s*pivot[012]/);
    expect(solve.match(/status_factorization_failed/g)?.length).toBeGreaterThanOrEqual(
      4,
    );
    expect(solve).toContain("gradient_dot_direction < 0.0");
    expect(solve).toContain("jgs2_globalization_status_zero_gradient");
    expect(solve).toContain("jgs2_globalization_shifted_relative_residual");
  });

  it("uses factored invariant and determinant deltas for material energy", () => {
    const energy = functionRegion(
      "jgs2_globalization_stable_neo_hookean_energy_density_delta",
    );
    expect(energy).toContain(
      "jgs2_globalization_determinant_delta",
    );
    expect(energy).toContain("final_deformation + initial");
    expect(energy).toContain("jgs2_globalization_log_one_plus");
    expect(energy).toContain(
      "(final_determinant + initial_determinant - 2.0)",
    );
    expect(energy).not.toContain(
      "jgs2_globalization_determinant(final) -",
    );
  });
});
