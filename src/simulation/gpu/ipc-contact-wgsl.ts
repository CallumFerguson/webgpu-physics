/**
 * Standalone IPC contact math for composition into JGS2 WGSL shaders.
 *
 * Contact weights always satisfy
 *
 *   gap = sum_i(weights[i] * position[i]) = point_a - point_b.
 *
 * Point-triangle contacts use `(point, triangle0, triangle1, triangle2)`;
 * edge-edge contacts use `(edgeA0, edgeA1, edgeB0, edgeB1)`. The returned
 * normal points from `point_b` toward `point_a`. At exactly zero separation,
 * the normal is a deterministic geometric fallback and must not be used to
 * infer contact sidedness.
 *
 * The barrier routines implement the IPC barrier only on its mathematical
 * domain `0 < distance < activation_distance`. They return zero outside that
 * domain; callers must reject non-positive distances and use the safe-step
 * cap before evaluating trial configurations.
 */
export const jgs2IpcContactWgsl = /* wgsl */ `
const jgs2_ipc_f32_max: f32 = 3.0e38;
const jgs2_ipc_length_squared_floor: f32 = 1.0e-20;
const jgs2_ipc_parallel_relative_epsilon: f32 = 1.0e-6;

struct jgs2_ipc_segment_result {
  point: vec3f,
  parameter: f32,
  squared_distance: f32,
}

struct jgs2_ipc_contact_data {
  point_a: vec3f,
  distance: f32,
  point_b: vec3f,
  squared_distance: f32,
  normal: vec3f,
  valid: u32,
  weights: vec4f,
}

fn jgs2_ipc_finite_scalar(value: f32) -> bool {
  return value == value && abs(value) <= jgs2_ipc_f32_max;
}

fn jgs2_ipc_finite_vec3(value: vec3f) -> bool {
  return jgs2_ipc_finite_scalar(value.x) &&
    jgs2_ipc_finite_scalar(value.y) &&
    jgs2_ipc_finite_scalar(value.z);
}

fn jgs2_ipc_finite_vec4(value: vec4f) -> bool {
  return jgs2_ipc_finite_scalar(value.x) &&
    jgs2_ipc_finite_scalar(value.y) &&
    jgs2_ipc_finite_scalar(value.z) &&
    jgs2_ipc_finite_scalar(value.w);
}

fn jgs2_ipc_safe_unit(value: vec3f, fallback: vec3f) -> vec3f {
  let length_squared = dot(value, value);
  if (
    jgs2_ipc_finite_scalar(length_squared) &&
    length_squared > jgs2_ipc_length_squared_floor
  ) {
    return value * inverseSqrt(length_squared);
  }

  let fallback_length_squared = dot(fallback, fallback);
  if (
    jgs2_ipc_finite_scalar(fallback_length_squared) &&
    fallback_length_squared > jgs2_ipc_length_squared_floor
  ) {
    return fallback * inverseSqrt(fallback_length_squared);
  }
  return vec3f(1.0, 0.0, 0.0);
}

fn jgs2_ipc_orthogonal_unit(direction: vec3f) -> vec3f {
  let absolute_direction = abs(direction);
  var axis = vec3f(1.0, 0.0, 0.0);
  if (absolute_direction.y <= absolute_direction.x &&
      absolute_direction.y <= absolute_direction.z) {
    axis = vec3f(0.0, 1.0, 0.0);
  } else if (absolute_direction.z <= absolute_direction.x &&
             absolute_direction.z <= absolute_direction.y) {
    axis = vec3f(0.0, 0.0, 1.0);
  }
  return jgs2_ipc_safe_unit(cross(direction, axis), vec3f(1.0, 0.0, 0.0));
}

fn jgs2_ipc_invalid_contact() -> jgs2_ipc_contact_data {
  return jgs2_ipc_contact_data(
    vec3f(0.0),
    0.0,
    vec3f(0.0),
    0.0,
    vec3f(1.0, 0.0, 0.0),
    0u,
    vec4f(0.0),
  );
}

fn jgs2_ipc_make_contact(
  point_a: vec3f,
  point_b: vec3f,
  weights: vec4f,
  fallback_normal: vec3f,
) -> jgs2_ipc_contact_data {
  if (!jgs2_ipc_finite_vec3(point_a) ||
      !jgs2_ipc_finite_vec3(point_b) ||
      !jgs2_ipc_finite_vec4(weights)) {
    return jgs2_ipc_invalid_contact();
  }

  let gap = point_a - point_b;
  let squared_distance = dot(gap, gap);
  if (!jgs2_ipc_finite_scalar(squared_distance) || squared_distance < 0.0) {
    return jgs2_ipc_invalid_contact();
  }
  let distance = sqrt(squared_distance);
  let normal = jgs2_ipc_safe_unit(gap, fallback_normal);
  return jgs2_ipc_contact_data(
    point_a,
    distance,
    point_b,
    squared_distance,
    normal,
    1u,
    weights,
  );
}

fn jgs2_ipc_closest_point_on_segment(
  query: vec3f,
  start: vec3f,
  end: vec3f,
) -> jgs2_ipc_segment_result {
  let edge = end - start;
  let edge_length_squared = dot(edge, edge);
  var parameter = 0.0;
  if (jgs2_ipc_finite_scalar(edge_length_squared) &&
      edge_length_squared > jgs2_ipc_length_squared_floor) {
    parameter = clamp(dot(query - start, edge) / edge_length_squared, 0.0, 1.0);
  }
  let point = start + parameter * edge;
  let difference = query - point;
  return jgs2_ipc_segment_result(point, parameter, dot(difference, difference));
}

// Robust point-triangle closest point. Interior projection is used only when
// the scaled area test passes; otherwise the nearest of all three segments is
// selected, which also covers a triangle collapsed to a point.
fn jgs2_ipc_point_triangle_contact(
  point: vec3f,
  triangle0: vec3f,
  triangle1: vec3f,
  triangle2: vec3f,
) -> jgs2_ipc_contact_data {
  let edge01 = triangle1 - triangle0;
  let edge02 = triangle2 - triangle0;
  let edge12 = triangle2 - triangle1;
  let edge01_squared = dot(edge01, edge01);
  let edge02_squared = dot(edge02, edge02);
  let edge12_squared = dot(edge12, edge12);
  let largest_edge_squared = max(edge01_squared, max(edge02_squared, edge12_squared));
  let area_normal = cross(edge01, edge02);
  let area_squared = dot(area_normal, area_normal);
  let area_threshold = max(
    jgs2_ipc_length_squared_floor * jgs2_ipc_length_squared_floor,
    jgs2_ipc_parallel_relative_epsilon * largest_edge_squared * largest_edge_squared,
  );

  var longest_edge = edge01;
  if (edge02_squared > edge01_squared && edge02_squared >= edge12_squared) {
    longest_edge = edge02;
  } else if (edge12_squared > edge01_squared && edge12_squared > edge02_squared) {
    longest_edge = edge12;
  }
  let fallback_normal = jgs2_ipc_safe_unit(
    area_normal,
    jgs2_ipc_orthogonal_unit(longest_edge),
  );

  if (jgs2_ipc_finite_scalar(area_squared) && area_squared > area_threshold) {
    let dot00 = edge01_squared;
    let dot01 = dot(edge01, edge02);
    let dot11 = edge02_squared;
    let point_offset = point - triangle0;
    let dot20 = dot(point_offset, edge01);
    let dot21 = dot(point_offset, edge02);
    let denominator = dot00 * dot11 - dot01 * dot01;
    let barycentric1 = (dot11 * dot20 - dot01 * dot21) / denominator;
    let barycentric2 = (dot00 * dot21 - dot01 * dot20) / denominator;
    let barycentric0 = 1.0 - barycentric1 - barycentric2;
    if (barycentric0 >= 0.0 && barycentric1 >= 0.0 && barycentric2 >= 0.0) {
      let closest = barycentric0 * triangle0 +
        barycentric1 * triangle1 + barycentric2 * triangle2;
      return jgs2_ipc_make_contact(
        point,
        closest,
        vec4f(1.0, -barycentric0, -barycentric1, -barycentric2),
        fallback_normal,
      );
    }
  }

  let candidate01 = jgs2_ipc_closest_point_on_segment(point, triangle0, triangle1);
  let candidate12 = jgs2_ipc_closest_point_on_segment(point, triangle1, triangle2);
  let candidate20 = jgs2_ipc_closest_point_on_segment(point, triangle2, triangle0);
  var closest = candidate01;
  var barycentrics = vec3f(1.0 - candidate01.parameter, candidate01.parameter, 0.0);
  if (candidate12.squared_distance < closest.squared_distance) {
    closest = candidate12;
    barycentrics = vec3f(0.0, 1.0 - candidate12.parameter, candidate12.parameter);
  }
  if (candidate20.squared_distance < closest.squared_distance) {
    closest = candidate20;
    barycentrics = vec3f(candidate20.parameter, 0.0, 1.0 - candidate20.parameter);
  }
  return jgs2_ipc_make_contact(
    point,
    closest.point,
    vec4f(1.0, -barycentrics.x, -barycentrics.y, -barycentrics.z),
    fallback_normal,
  );
}

// Closest points on two finite segments, including point-segment,
// point-point, and nearly parallel degeneracies.
fn jgs2_ipc_edge_edge_contact(
  edge_a0: vec3f,
  edge_a1: vec3f,
  edge_b0: vec3f,
  edge_b1: vec3f,
) -> jgs2_ipc_contact_data {
  let direction_a = edge_a1 - edge_a0;
  let direction_b = edge_b1 - edge_b0;
  let offset = edge_a0 - edge_b0;
  let length_a_squared = dot(direction_a, direction_a);
  let length_b_squared = dot(direction_b, direction_b);
  let length_scale = max(length_a_squared, length_b_squared);
  let degenerate_threshold = max(
    jgs2_ipc_length_squared_floor,
    jgs2_ipc_parallel_relative_epsilon * length_scale,
  );
  let a_degenerate = !(length_a_squared > degenerate_threshold);
  let b_degenerate = !(length_b_squared > degenerate_threshold);
  var parameter_a = 0.0;
  var parameter_b = 0.0;

  if (a_degenerate && !b_degenerate) {
    parameter_b = clamp(dot(direction_b, offset) / length_b_squared, 0.0, 1.0);
  } else if (!a_degenerate) {
    let projection_a = dot(direction_a, offset);
    if (b_degenerate) {
      parameter_a = clamp(-projection_a / length_a_squared, 0.0, 1.0);
    } else {
      let coupling = dot(direction_a, direction_b);
      let projection_b = dot(direction_b, offset);
      let denominator = length_a_squared * length_b_squared - coupling * coupling;
      let parallel_threshold = jgs2_ipc_parallel_relative_epsilon *
        length_a_squared * length_b_squared;
      if (denominator > parallel_threshold) {
        parameter_a = clamp(
          (coupling * projection_b - projection_a * length_b_squared) /
            denominator,
          0.0,
          1.0,
        );
      }
      let parameter_b_unclamped =
        (coupling * parameter_a + projection_b) / length_b_squared;
      if (parameter_b_unclamped < 0.0) {
        parameter_b = 0.0;
        parameter_a = clamp(-projection_a / length_a_squared, 0.0, 1.0);
      } else if (parameter_b_unclamped > 1.0) {
        parameter_b = 1.0;
        parameter_a = clamp(
          (coupling - projection_a) / length_a_squared,
          0.0,
          1.0,
        );
      } else {
        parameter_b = parameter_b_unclamped;
      }
    }
  }

  let point_a = edge_a0 + parameter_a * direction_a;
  let point_b = edge_b0 + parameter_b * direction_b;
  var longest_direction = direction_a;
  if (length_b_squared > length_a_squared) {
    longest_direction = direction_b;
  }
  let fallback_normal = jgs2_ipc_safe_unit(
    cross(direction_a, direction_b),
    jgs2_ipc_orthogonal_unit(longest_direction),
  );
  return jgs2_ipc_make_contact(
    point_a,
    point_b,
    vec4f(
      1.0 - parameter_a,
      parameter_a,
      -(1.0 - parameter_b),
      -parameter_b,
    ),
    fallback_normal,
  );
}

fn jgs2_ipc_barrier_active(distance: f32, activation_distance: f32) -> bool {
  return jgs2_ipc_finite_scalar(distance) &&
    jgs2_ipc_finite_scalar(activation_distance) &&
    distance > 0.0 && distance < activation_distance;
}

// B(d, dHat) = -(d - dHat)^2 log(d / dHat).
fn jgs2_ipc_barrier_value(distance: f32, activation_distance: f32) -> f32 {
  if (!jgs2_ipc_barrier_active(distance, activation_distance)) {
    return 0.0;
  }
  let offset = distance - activation_distance;
  let log_ratio = log(distance) - log(activation_distance);
  return -(offset * offset) * log_ratio;
}

// dB/dd = -2(d - dHat) log(d / dHat) - (d - dHat)^2 / d.
fn jgs2_ipc_barrier_first_derivative(
  distance: f32,
  activation_distance: f32,
) -> f32 {
  if (!jgs2_ipc_barrier_active(distance, activation_distance)) {
    return 0.0;
  }
  let offset = distance - activation_distance;
  let log_ratio = log(distance) - log(activation_distance);
  return -2.0 * offset * log_ratio - offset * offset / distance;
}

// d2B/dd2 = -2 log(d / dHat) - 4(d - dHat)/d + (d - dHat)^2/d^2.
fn jgs2_ipc_barrier_second_derivative(
  distance: f32,
  activation_distance: f32,
) -> f32 {
  if (!jgs2_ipc_barrier_active(distance, activation_distance)) {
    return 0.0;
  }
  let offset = distance - activation_distance;
  let log_ratio = log(distance) - log(activation_distance);
  return -2.0 * log_ratio - 4.0 * offset / distance +
    offset * offset / (distance * distance);
}

// Scalar multiplying a contact normal in one vertex's barrier gradient.
fn jgs2_ipc_normal_gradient_scalar(
  vertex_weight: f32,
  barrier_stiffness: f32,
  barrier_first_derivative: f32,
) -> f32 {
  if (!jgs2_ipc_finite_scalar(vertex_weight) ||
      !jgs2_ipc_finite_scalar(barrier_stiffness) ||
      !jgs2_ipc_finite_scalar(barrier_first_derivative) ||
      barrier_stiffness < 0.0) {
    return 0.0;
  }
  return vertex_weight * barrier_stiffness * barrier_first_derivative;
}

// Projected normal curvature kappa * weight^2 * max(B'', 0). Dropping the
// tangential distance Hessian makes this local block positive semidefinite.
fn jgs2_ipc_psd_normal_hessian_scalar(
  vertex_weight: f32,
  barrier_stiffness: f32,
  barrier_second_derivative: f32,
) -> f32 {
  if (!jgs2_ipc_finite_scalar(vertex_weight) ||
      !jgs2_ipc_finite_scalar(barrier_stiffness) ||
      !jgs2_ipc_finite_scalar(barrier_second_derivative) ||
      barrier_stiffness <= 0.0 || vertex_weight == 0.0 ||
      barrier_second_derivative <= 0.0) {
    return 0.0;
  }
  let scalar = barrier_stiffness * vertex_weight * vertex_weight *
    barrier_second_derivative;
  if (!jgs2_ipc_finite_scalar(scalar)) {
    return jgs2_ipc_f32_max;
  }
  return scalar;
}

fn jgs2_ipc_psd_normal_hessian(normal: vec3f, scalar: f32) -> mat3x3f {
  let unit_normal = jgs2_ipc_safe_unit(normal, vec3f(1.0, 0.0, 0.0));
  let nonnegative_scalar = max(scalar, 0.0);
  return mat3x3f(
    nonnegative_scalar * unit_normal.x * unit_normal,
    nonnegative_scalar * unit_normal.y * unit_normal,
    nonnegative_scalar * unit_normal.z * unit_normal,
  );
}

// Conservative Lipschitz bound: d(alpha) >= d(0) - alpha * L. The safety
// factor should be in [0, 1]; 0.9 leaves a strict margin above min_distance.
fn jgs2_ipc_candidate_safe_step_cap(
  source_distance: f32,
  min_distance: f32,
  displacement_lipschitz_bound: f32,
  safety_factor: f32,
) -> f32 {
  if (!jgs2_ipc_finite_scalar(source_distance) ||
      !jgs2_ipc_finite_scalar(min_distance) ||
      !jgs2_ipc_finite_scalar(displacement_lipschitz_bound) ||
      !jgs2_ipc_finite_scalar(safety_factor) ||
      min_distance < 0.0 || source_distance <= min_distance ||
      displacement_lipschitz_bound < 0.0) {
    return 0.0;
  }
  if (displacement_lipschitz_bound == 0.0) {
    return 1.0;
  }
  let safe_fraction = clamp(safety_factor, 0.0, 1.0) *
    (source_distance - min_distance) / displacement_lipschitz_bound;
  return clamp(safe_fraction, 0.0, 1.0);
}

// IPC's C1 smoothing primitive. Its derivative is jgs2_ipc_friction_f1.
fn jgs2_ipc_friction_f0(slip_norm: f32, smoothing_threshold: f32) -> f32 {
  let nonnegative_slip = max(slip_norm, 0.0);
  if (!(smoothing_threshold > 0.0) || nonnegative_slip >= smoothing_threshold) {
    return nonnegative_slip;
  }
  let normalized_slip = nonnegative_slip / smoothing_threshold;
  return smoothing_threshold * (
    -normalized_slip * normalized_slip * normalized_slip / 3.0 +
    normalized_slip * normalized_slip + 1.0 / 3.0
  );
}

fn jgs2_ipc_friction_f1(slip_norm: f32, smoothing_threshold: f32) -> f32 {
  let nonnegative_slip = max(slip_norm, 0.0);
  if (!(smoothing_threshold > 0.0) || nonnegative_slip >= smoothing_threshold) {
    return select(0.0, 1.0, nonnegative_slip > 0.0);
  }
  let normalized_slip = nonnegative_slip / smoothing_threshold;
  return normalized_slip * (2.0 - normalized_slip);
}

// Lagged normal force is held fixed for a nonlinear solve. This is the
// nonnegative potential mu * lambda_n * (f0(s) - f0(0)), evaluated in a form
// that avoids cancellation near zero slip.
fn jgs2_ipc_lagged_friction_dissipation(
  slip_norm: f32,
  smoothing_threshold: f32,
  friction_coefficient: f32,
  lagged_normal_force: f32,
) -> f32 {
  if (!jgs2_ipc_finite_scalar(slip_norm) ||
      !jgs2_ipc_finite_scalar(smoothing_threshold) ||
      !jgs2_ipc_finite_scalar(friction_coefficient) ||
      !jgs2_ipc_finite_scalar(lagged_normal_force)) {
    return 0.0;
  }
  let nonnegative_slip = max(slip_norm, 0.0);
  let scale = max(friction_coefficient, 0.0) * max(lagged_normal_force, 0.0);
  if (!(smoothing_threshold > 0.0)) {
    return scale * nonnegative_slip;
  }
  if (nonnegative_slip < smoothing_threshold) {
    let normalized_slip = nonnegative_slip / smoothing_threshold;
    return scale * smoothing_threshold * normalized_slip * normalized_slip *
      (1.0 - normalized_slip / 3.0);
  }
  return scale * (nonnegative_slip - smoothing_threshold / 3.0);
}
`;
