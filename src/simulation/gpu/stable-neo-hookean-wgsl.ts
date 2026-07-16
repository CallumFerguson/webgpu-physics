/**
 * Standalone stable Neo-Hookean material routines for composition into WGSL
 * compute shaders. Every symbol is prefixed so the source can be concatenated
 * with independently authored shader libraries.
 *
 * Matrices use WGSL's column-major convention: `f[column][row]`. Callers must
 * pass the adjusted Smith/Pixar parameters
 *
 *   adjustedMu = (4 / 3) * hookeMu
 *   adjustedLambda = hookeLambda + (5 / 6) * hookeMu
 *
 * rather than the unmodified Hooke-law Lame constants.
 */
export const stableNeoHookeanWgsl = /* wgsl */ `
fn snh_frobenius_inner(a: mat3x3f, b: mat3x3f) -> f32 {
  return dot(a[0], b[0]) + dot(a[1], b[1]) + dot(a[2], b[2]);
}

fn snh_determinant(f: mat3x3f) -> f32 {
  return dot(f[0], cross(f[1], f[2]));
}

// Polynomial d(det(F)) / dF. Unlike det(F) * transpose(inverse(F)), this is
// finite at singular F and remains defined for inverted configurations.
fn snh_cofactor(f: mat3x3f) -> mat3x3f {
  return mat3x3f(
    cross(f[1], f[2]),
    cross(f[2], f[0]),
    cross(f[0], f[1]),
  );
}

// Exact directional derivative D(cofactor(F))[h], written by WGSL columns.
fn snh_directional_cofactor(f: mat3x3f, h: mat3x3f) -> mat3x3f {
  return mat3x3f(
    cross(h[1], f[2]) + cross(f[1], h[2]),
    cross(h[2], f[0]) + cross(f[2], h[0]),
    cross(h[0], f[1]) + cross(f[0], h[1]),
  );
}

// Corrected stable Neo-Hookean density, expanded around J = 1 and normalized
// so both the energy and first Piola stress are exactly zero at every rotation.
fn snh_energy_density(
  f: mat3x3f,
  adjusted_lambda: f32,
  adjusted_mu: f32,
) -> f32 {
  let first_invariant = snh_frobenius_inner(f, f);
  let determinant_offset = snh_determinant(f) - 1.0;
  return 0.5 * adjusted_mu * (first_invariant - 3.0)
    - 0.75 * adjusted_mu * determinant_offset
    + 0.5 * adjusted_lambda * determinant_offset * determinant_offset
    - 0.5 * adjusted_mu * log((first_invariant + 1.0) * 0.25);
}

fn snh_first_piola(
  f: mat3x3f,
  adjusted_lambda: f32,
  adjusted_mu: f32,
) -> mat3x3f {
  let first_invariant = snh_frobenius_inner(f, f);
  let invariant_denominator = first_invariant + 1.0;
  let cofactor = snh_cofactor(f);
  let shear_scale = adjusted_mu * (1.0 - 1.0 / invariant_denominator);
  let volume_scale = adjusted_lambda * (snh_determinant(f) - 1.0)
    - 0.75 * adjusted_mu;
  return shear_scale * f + volume_scale * cofactor;
}

// Apply the exact material tangent DP(F)[h] without constructing a 9x9 matrix.
fn snh_tangent_product(
  f: mat3x3f,
  h: mat3x3f,
  adjusted_lambda: f32,
  adjusted_mu: f32,
) -> mat3x3f {
  let first_invariant = snh_frobenius_inner(f, f);
  let invariant_denominator = first_invariant + 1.0;
  let cofactor = snh_cofactor(f);
  let directional_cofactor = snh_directional_cofactor(f, h);
  let shear_scale = adjusted_mu * (1.0 - 1.0 / invariant_denominator);
  let directional_shear_scale =
    2.0 * adjusted_mu * snh_frobenius_inner(f, h)
    / (invariant_denominator * invariant_denominator);
  let volume_scale = adjusted_lambda * (snh_determinant(f) - 1.0)
    - 0.75 * adjusted_mu;
  let directional_volume_scale =
    adjusted_lambda * snh_frobenius_inner(cofactor, h);
  return shear_scale * h
    + directional_shear_scale * f
    + directional_volume_scale * cofactor
    + volume_scale * directional_cofactor;
}
`;
