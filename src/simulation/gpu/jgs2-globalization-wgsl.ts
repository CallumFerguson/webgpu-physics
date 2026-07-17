/**
 * Stable, collision-safe WGSL helpers for the Phase 1 nonlinear JGS2 local
 * solve. The material Hessian is symmetrized and scaled before its spectrum
 * is evaluated. The only permitted positive-definite treatment is the
 * reported uniform diagonal shift; failed Cholesky pivots are never clamped.
 */

import {
  JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED,
  JGS2_GLOBALIZATION_LOCAL_STATUS_NON_DESCENT,
  JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
  JGS2_GLOBALIZATION_LOCAL_STATUS_SHIFT_LIMIT,
  JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT,
} from "./jgs2-globalization";

export const JGS2_GLOBALIZATION_STATUS_CODES = Object.freeze({
  accepted: JGS2_GLOBALIZATION_LOCAL_STATUS_ACCEPTED,
  zeroGradient: JGS2_GLOBALIZATION_LOCAL_STATUS_ZERO_GRADIENT,
  invalidInput: JGS2_GLOBALIZATION_LOCAL_STATUS_NONFINITE,
  shiftLimitExceeded: JGS2_GLOBALIZATION_LOCAL_STATUS_SHIFT_LIMIT,
  factorizationFailed: JGS2_GLOBALIZATION_LOCAL_STATUS_CHOLESKY_FAILED,
  nonDescentDirection: JGS2_GLOBALIZATION_LOCAL_STATUS_NON_DESCENT,
} as const);

export const jgs2GlobalizationWgsl = /* wgsl */ `
const jgs2_globalization_f32_max: f32 = 3.0e38;
const jgs2_globalization_f32_unit_roundoff: f32 = 0.000000059604644775390625;
const jgs2_globalization_position_resolution_multiplier: f32 = 8.0;
const jgs2_globalization_relative_eigenvalue_floor: f32 = 0.0000152587890625;
const jgs2_globalization_max_normalized_shift: f32 = 0.001;
const jgs2_globalization_armijo_c1: f32 = 0.0001;
const jgs2_globalization_backtrack_factor: f32 = 0.5;
const jgs2_globalization_max_backtracks: u32 = 12u;
const jgs2_globalization_trial_count: u32 = 13u;
const jgs2_globalization_determinant_floor: f32 = 0.0001;
const jgs2_globalization_two_pi_over_three: f32 = 2.0943951023931953;

const jgs2_globalization_status_accepted: u32 =
  ${JGS2_GLOBALIZATION_STATUS_CODES.accepted}u;
const jgs2_globalization_status_zero_gradient: u32 =
  ${JGS2_GLOBALIZATION_STATUS_CODES.zeroGradient}u;
const jgs2_globalization_status_invalid_input: u32 =
  ${JGS2_GLOBALIZATION_STATUS_CODES.invalidInput}u;
const jgs2_globalization_status_shift_limit_exceeded: u32 =
  ${JGS2_GLOBALIZATION_STATUS_CODES.shiftLimitExceeded}u;
const jgs2_globalization_status_factorization_failed: u32 =
  ${JGS2_GLOBALIZATION_STATUS_CODES.factorizationFailed}u;
const jgs2_globalization_status_non_descent_direction: u32 =
  ${JGS2_GLOBALIZATION_STATUS_CODES.nonDescentDirection}u;

struct jgs2_globalization_shift_result {
  normalized_hessian: mat3x3f,
  scale: f32,
  minimum_eigenvalue_normalized: f32,
  normalized_shift: f32,
  maximum_relative_asymmetry: f32,
  status: u32,
}

struct jgs2_globalization_solve_result {
  direction: vec3f,
  gradient_dot_direction: f32,
  scale: f32,
  minimum_eigenvalue_normalized: f32,
  normalized_shift: f32,
  maximum_relative_asymmetry: f32,
  shifted_relative_residual: f32,
  status: u32,
}

fn jgs2_globalization_finite_scalar(value: f32) -> bool {
  return value == value && abs(value) <= jgs2_globalization_f32_max;
}

fn jgs2_globalization_finite_vec3(value: vec3f) -> bool {
  return jgs2_globalization_finite_scalar(value.x) &&
    jgs2_globalization_finite_scalar(value.y) &&
    jgs2_globalization_finite_scalar(value.z);
}

fn jgs2_globalization_finite_mat3(value: mat3x3f) -> bool {
  return jgs2_globalization_finite_vec3(value[0]) &&
    jgs2_globalization_finite_vec3(value[1]) &&
    jgs2_globalization_finite_vec3(value[2]);
}

fn jgs2_globalization_zero_mat3() -> mat3x3f {
  return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
}

fn jgs2_globalization_determinant(matrix: mat3x3f) -> f32 {
  return dot(matrix[0], cross(matrix[1], matrix[2]));
}

fn jgs2_globalization_frobenius_inner(
  left: mat3x3f,
  right: mat3x3f,
) -> f32 {
  return dot(left[0], right[0]) + dot(left[1], right[1]) +
    dot(left[2], right[2]);
}

fn jgs2_globalization_symmetrize(matrix: mat3x3f) -> mat3x3f {
  let a01 = 0.5 * (matrix[0][1] + matrix[1][0]);
  let a02 = 0.5 * (matrix[0][2] + matrix[2][0]);
  let a12 = 0.5 * (matrix[1][2] + matrix[2][1]);
  return mat3x3f(
    vec3f(matrix[0][0], a01, a02),
    vec3f(a01, matrix[1][1], a12),
    vec3f(a02, a12, matrix[2][2]),
  );
}

fn jgs2_globalization_maximum_relative_asymmetry(matrix: mat3x3f) -> f32 {
  let maximum_magnitude = max(
    max(max(abs(matrix[0][0]), abs(matrix[0][1])), abs(matrix[0][2])),
    max(
      max(max(abs(matrix[1][0]), abs(matrix[1][1])), abs(matrix[1][2])),
      max(max(abs(matrix[2][0]), abs(matrix[2][1])), abs(matrix[2][2])),
    ),
  );
  if (!(maximum_magnitude > 0.0)) {
    return 0.0;
  }
  let maximum_difference = max(
    abs(matrix[0][1] - matrix[1][0]),
    max(
      abs(matrix[0][2] - matrix[2][0]),
      abs(matrix[1][2] - matrix[2][1]),
    ),
  );
  return maximum_difference / maximum_magnitude;
}

// Compute ||A||_F / sqrt(3) after normalizing by the largest entry. This
// prevents the intermediate squares from overflowing or underflowing.
fn jgs2_globalization_frobenius_over_sqrt_three(
  matrix: mat3x3f,
) -> f32 {
  let maximum_entry = max(
    max(max(abs(matrix[0][0]), abs(matrix[1][1])), abs(matrix[2][2])),
    max(
      max(abs(matrix[0][1]), abs(matrix[0][2])),
      abs(matrix[1][2]),
    ),
  );
  if (!(maximum_entry > 0.0)) {
    return 0.0;
  }
  // Divide each entry directly. Forming 1 / maximum_entry first can overflow
  // when an otherwise representable matrix is subnormal.
  let a00 = matrix[0][0] / maximum_entry;
  let a11 = matrix[1][1] / maximum_entry;
  let a22 = matrix[2][2] / maximum_entry;
  let a01 = matrix[0][1] / maximum_entry;
  let a02 = matrix[0][2] / maximum_entry;
  let a12 = matrix[1][2] / maximum_entry;
  let normalized_squared =
    a00 * a00 + a11 * a11 + a22 * a22 +
    2.0 * (a01 * a01 + a02 * a02 + a12 * a12);
  return maximum_entry * sqrt(normalized_squared / 3.0);
}

// Closed-form spectrum of an already scaled symmetric 3x3 matrix.
fn jgs2_globalization_minimum_symmetric_eigenvalue_normalized(
  matrix: mat3x3f,
) -> f32 {
  let a00 = matrix[0][0];
  let a11 = matrix[1][1];
  let a22 = matrix[2][2];
  let a01 = matrix[0][1];
  let a02 = matrix[0][2];
  let a12 = matrix[1][2];
  let off_diagonal_squared = a01 * a01 + a02 * a02 + a12 * a12;
  if (off_diagonal_squared == 0.0) {
    return min(a00, min(a11, a22));
  }

  let mean = (a00 + a11 + a22) / 3.0;
  let d00 = a00 - mean;
  let d11 = a11 - mean;
  let d22 = a22 - mean;
  let centered_squared =
    d00 * d00 + d11 * d11 + d22 * d22 +
    2.0 * off_diagonal_squared;
  let spectral_scale = sqrt(centered_squared / 6.0);
  if (!(spectral_scale > 0.0)) {
    return min(a00, min(a11, a22));
  }

  let normalized_centered = mat3x3f(
    vec3f(d00, a01, a02) / spectral_scale,
    vec3f(a01, d11, a12) / spectral_scale,
    vec3f(a02, a12, d22) / spectral_scale,
  );
  let half_determinant = clamp(
    0.5 * jgs2_globalization_determinant(normalized_centered),
    -1.0,
    1.0,
  );
  let angle = acos(half_determinant) / 3.0;
  let largest = mean + 2.0 * spectral_scale * cos(angle);
  let smallest = mean + 2.0 * spectral_scale *
    cos(angle + jgs2_globalization_two_pi_over_three);
  let middle = 3.0 * mean - largest - smallest;
  return min(largest, min(middle, smallest));
}

fn jgs2_globalization_invalid_shift_result(
  status: u32,
) -> jgs2_globalization_shift_result {
  return jgs2_globalization_shift_result(
    jgs2_globalization_zero_mat3(),
    0.0,
    0.0,
    0.0,
    0.0,
    status,
  );
}

fn jgs2_globalization_classify_shift(
  hessian: mat3x3f,
  inertia_scale: f32,
) -> jgs2_globalization_shift_result {
  if (!jgs2_globalization_finite_mat3(hessian) ||
      !jgs2_globalization_finite_scalar(inertia_scale) ||
      !(inertia_scale > 0.0)) {
    return jgs2_globalization_invalid_shift_result(
      jgs2_globalization_status_invalid_input,
    );
  }

  let symmetric = jgs2_globalization_symmetrize(hessian);
  let largest_absolute_diagonal = max(
    abs(symmetric[0][0]),
    max(abs(symmetric[1][1]), abs(symmetric[2][2])),
  );
  let scale = max(
    inertia_scale,
    max(
      largest_absolute_diagonal,
      jgs2_globalization_frobenius_over_sqrt_three(symmetric),
    ),
  );
  if (!jgs2_globalization_finite_scalar(scale) || !(scale > 0.0)) {
    return jgs2_globalization_invalid_shift_result(
      jgs2_globalization_status_invalid_input,
    );
  }

  let normalized_hessian = mat3x3f(
    symmetric[0] / scale,
    symmetric[1] / scale,
    symmetric[2] / scale,
  );
  let minimum_eigenvalue_normalized =
    jgs2_globalization_minimum_symmetric_eigenvalue_normalized(
      normalized_hessian,
    );
  if (!jgs2_globalization_finite_scalar(minimum_eigenvalue_normalized)) {
    return jgs2_globalization_invalid_shift_result(
      jgs2_globalization_status_invalid_input,
    );
  }
  let normalized_shift = max(
    0.0,
    jgs2_globalization_relative_eigenvalue_floor -
      minimum_eigenvalue_normalized,
  );
  let status = select(
    jgs2_globalization_status_accepted,
    jgs2_globalization_status_shift_limit_exceeded,
    normalized_shift > jgs2_globalization_max_normalized_shift,
  );
  return jgs2_globalization_shift_result(
    normalized_hessian,
    scale,
    minimum_eigenvalue_normalized,
    normalized_shift,
    jgs2_globalization_maximum_relative_asymmetry(hessian),
    status,
  );
}

fn jgs2_globalization_scaled_length3(value: vec3f) -> f32 {
  let maximum_component = max(abs(value.x), max(abs(value.y), abs(value.z)));
  if (!(maximum_component > 0.0)) {
    return 0.0;
  }
  let normalized = value / maximum_component;
  return maximum_component * sqrt(dot(normalized, normalized));
}

fn jgs2_globalization_shifted_relative_residual(
  shifted_normalized_hessian: mat3x3f,
  direction: vec3f,
  right_hand_side: vec3f,
) -> f32 {
  let residual = shifted_normalized_hessian * direction - right_hand_side;
  let residual_norm = jgs2_globalization_scaled_length3(residual);
  let right_hand_side_norm = jgs2_globalization_scaled_length3(
    right_hand_side,
  );
  return residual_norm / max(1.0, right_hand_side_norm);
}

fn jgs2_globalization_solve_result_from_shift(
  shift: jgs2_globalization_shift_result,
  direction: vec3f,
  gradient_dot_direction: f32,
  shifted_relative_residual: f32,
  status: u32,
) -> jgs2_globalization_solve_result {
  return jgs2_globalization_solve_result(
    direction,
    gradient_dot_direction,
    shift.scale,
    shift.minimum_eigenvalue_normalized,
    shift.normalized_shift,
    shift.maximum_relative_asymmetry,
    shifted_relative_residual,
    status,
  );
}

fn jgs2_globalization_solve(
  hessian: mat3x3f,
  gradient: vec3f,
  inertia_scale: f32,
) -> jgs2_globalization_solve_result {
  let shift = jgs2_globalization_classify_shift(hessian, inertia_scale);
  if (shift.status != jgs2_globalization_status_accepted) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      shift.status,
    );
  }
  if (!jgs2_globalization_finite_vec3(gradient)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_invalid_input,
    );
  }
  let gradient_scale = max(
    abs(gradient.x),
    max(abs(gradient.y), abs(gradient.z)),
  );
  if (!(gradient_scale > 0.0)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_zero_gradient,
    );
  }

  let matrix = mat3x3f(
    vec3f(
      shift.normalized_hessian[0][0] + shift.normalized_shift,
      shift.normalized_hessian[0][1],
      shift.normalized_hessian[0][2],
    ),
    vec3f(
      shift.normalized_hessian[1][0],
      shift.normalized_hessian[1][1] + shift.normalized_shift,
      shift.normalized_hessian[1][2],
    ),
    vec3f(
      shift.normalized_hessian[2][0],
      shift.normalized_hessian[2][1],
      shift.normalized_hessian[2][2] + shift.normalized_shift,
    ),
  );
  let right_hand_side = -gradient / shift.scale;
  if (!jgs2_globalization_finite_vec3(right_hand_side)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_invalid_input,
    );
  }

  // Cholesky of the one uniformly shifted matrix. A bad pivot fails closed;
  // max(pivot, epsilon) would silently introduce a second modification.
  let pivot0 = matrix[0][0];
  if (!jgs2_globalization_finite_scalar(pivot0) || !(pivot0 > 0.0)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_factorization_failed,
    );
  }
  let l00 = sqrt(pivot0);
  let l10 = matrix[0][1] / l00;
  let l20 = matrix[0][2] / l00;
  let pivot1 = matrix[1][1] - l10 * l10;
  if (!jgs2_globalization_finite_scalar(pivot1) || !(pivot1 > 0.0)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_factorization_failed,
    );
  }
  let l11 = sqrt(pivot1);
  let l21 = (matrix[1][2] - l20 * l10) / l11;
  let pivot2 = matrix[2][2] - l20 * l20 - l21 * l21;
  if (!jgs2_globalization_finite_scalar(pivot2) || !(pivot2 > 0.0)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_factorization_failed,
    );
  }
  let l22 = sqrt(pivot2);

  let y0 = right_hand_side.x / l00;
  let y1 = (right_hand_side.y - l10 * y0) / l11;
  let y2 = (right_hand_side.z - l20 * y0 - l21 * y1) / l22;
  let x2 = y2 / l22;
  let x1 = (y1 - l21 * x2) / l11;
  let x0 = (y0 - l10 * x1 - l20 * x2) / l00;
  let direction = vec3f(x0, x1, x2);
  if (!jgs2_globalization_finite_vec3(direction)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_factorization_failed,
    );
  }

  let gradient_dot_direction = dot(gradient, direction);
  if (!jgs2_globalization_finite_scalar(gradient_dot_direction)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      0.0,
      0.0,
      jgs2_globalization_status_non_descent_direction,
    );
  }
  if (!(gradient_dot_direction < 0.0)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      gradient_dot_direction,
      0.0,
      jgs2_globalization_status_non_descent_direction,
    );
  }

  let shifted_relative_residual =
    jgs2_globalization_shifted_relative_residual(
      matrix,
      direction,
      right_hand_side,
    );
  if (!jgs2_globalization_finite_scalar(shifted_relative_residual)) {
    return jgs2_globalization_solve_result_from_shift(
      shift,
      vec3f(0.0),
      gradient_dot_direction,
      0.0,
      jgs2_globalization_status_factorization_failed,
    );
  }
  return jgs2_globalization_solve_result_from_shift(
    shift,
    direction,
    gradient_dot_direction,
    shifted_relative_residual,
    jgs2_globalization_status_accepted,
  );
}

// WGSL has no log1p. The short series preserves small invariant changes that
// would disappear when 1 + z rounds to one in f32.
fn jgs2_globalization_log_one_plus(value: f32) -> f32 {
  if (abs(value) <= 0.001) {
    let value2 = value * value;
    return value * (1.0 + value * (
      -0.5 + value * (0.3333333333333333 + value * (
        -0.25 + 0.2 * value
      ))
    ));
  }
  return log(1.0 + value);
}

// Cancellation-resistant det(F1) - det(F0), expanded as mixed triple
// products of D = F1 - F0 rather than subtracting two nearby determinants.
fn jgs2_globalization_determinant_delta(
  initial: mat3x3f,
  final_deformation: mat3x3f,
) -> f32 {
  let delta = final_deformation - initial;
  return
    dot(delta[0], cross(initial[1], initial[2])) +
    dot(initial[0], cross(delta[1], initial[2])) +
    dot(initial[0], cross(initial[1], delta[2])) +
    dot(delta[0], cross(delta[1], initial[2])) +
    dot(delta[0], cross(initial[1], delta[2])) +
    dot(initial[0], cross(delta[1], delta[2])) +
    dot(delta[0], cross(delta[1], delta[2]));
}

// Corrected stable Neo-Hookean density difference psi(F1) - psi(F0).
// adjusted_lambda and adjusted_mu are the same Smith/Pixar parameters used by
// snh_energy_density. The factored invariant and determinant differences make
// this suitable for an f32 Armijo comparison close to the source pose.
fn jgs2_globalization_stable_neo_hookean_energy_density_delta(
  initial: mat3x3f,
  final_deformation: mat3x3f,
  adjusted_lambda: f32,
  adjusted_mu: f32,
) -> f32 {
  let deformation_delta = final_deformation - initial;
  let initial_invariant = jgs2_globalization_frobenius_inner(
    initial,
    initial,
  );
  let invariant_delta = jgs2_globalization_frobenius_inner(
    deformation_delta,
    final_deformation + initial,
  );
  let determinant_delta = jgs2_globalization_determinant_delta(
    initial,
    final_deformation,
  );
  let initial_determinant = jgs2_globalization_determinant(initial);
  let final_determinant = jgs2_globalization_determinant(final_deformation);
  let log_ratio = jgs2_globalization_log_one_plus(
    invariant_delta / (initial_invariant + 1.0),
  );
  return
    0.5 * adjusted_mu * invariant_delta -
    0.75 * adjusted_mu * determinant_delta +
    0.5 * adjusted_lambda * determinant_delta *
      (final_determinant + initial_determinant - 2.0) -
    0.5 * adjusted_mu * log_ratio;
}
`;
