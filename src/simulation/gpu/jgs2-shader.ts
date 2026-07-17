import { JGS2_CUBATURE_RECORD_WORDS } from "./layout";
import {
  JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT,
  JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT,
  JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT,
  JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT,
  JGS2_GLOBALIZATION_VALIDITY_BITS_MASK,
} from "./jgs2-globalization";
import { jgs2GlobalizationWgsl } from "./jgs2-globalization-wgsl";
import { stableNeoHookeanWgsl } from "./stable-neo-hookean-wgsl";

export const JGS2_WORKGROUP_SIZE = 128;

export const jgs2Shader = /* wgsl */ `
const WORKGROUP_SIZE: u32 = ${JGS2_WORKGROUP_SIZE}u;
const CUBATURE_RECORD_WORDS: u32 = ${JGS2_CUBATURE_RECORD_WORDS}u;
const EMPTY_TET: u32 = 0xffffffffu;
const LOCAL_GLOBALIZATION_STRIDE: u32 = 4u;
const TET_GLOBALIZATION_STRIDE: u32 = 2u;
const CONVERGENCE_COMPONENT_STRIDE: u32 = 5u;
const GLOBALIZATION_CONTROL_STRIDE: u32 = 4u;
const GLOBALIZATION_HISTORY_STRIDE: u32 = 8u;

const LOCAL_STATUS_ACCEPTED: u32 = 0u;
const LOCAL_STATUS_PINNED: u32 = 1u;
const LOCAL_STATUS_ZERO_GRADIENT: u32 = 2u;
const LOCAL_STATUS_SHIFT_LIMIT: u32 = 3u;
const LOCAL_STATUS_NON_DESCENT: u32 = 4u;
const LOCAL_STATUS_LINE_SEARCH_FAILED: u32 = 5u;
const LOCAL_STATUS_NONFINITE: u32 = 6u;
const LOCAL_STATUS_CHOLESKY_FAILED: u32 = 7u;

const SOURCE_GEOMETRY_VALID_BIT: u32 =
  ${JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT}u;
const CANDIDATE_GEOMETRY_VALID_BIT: u32 =
  ${JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT}u;
const ACCEPTED_MINIMUM_VALID_BIT: u32 =
  ${JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT}u;
const LOCAL_NUMERICS_VALID_BIT: u32 =
  ${JGS2_GLOBALIZATION_LOCAL_NUMERICS_VALID_BIT}u;
const VALIDITY_BITS_MASK: u32 = ${JGS2_GLOBALIZATION_VALIDITY_BITS_MASK}u;

const CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT: u32 = 16u;
const CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT: u32 = 32u;
const CANDIDATE_REDUCTION_VALIDITY_BITS_MASK: u32 =
  SOURCE_GEOMETRY_VALID_BIT |
  CANDIDATE_GEOMETRY_VALID_BIT |
  LOCAL_NUMERICS_VALID_BIT |
  CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT |
  CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT;

struct VertexStatic {
  restMass: vec4f,
  color: vec4f,
  info: vec4u,
}

struct TetStatic {
  indices: vec4u,
  invDm0: vec4f,
  invDm1: vec4f,
  invDm2: vec4f,
  attributes: vec4f,
}

struct SimParams {
  // vertex count, tet count, cubature K, body count
  counts: vec4u,
  // posA, posB, predicted, velocity offsets in vec4 elements
  offsets0: vec4u,
  // old, vertex rotation, tet rotation, iteration source position
  offsets1: vec4u,
  // iteration target position, per-body correction, final update, reserved
  offsets2: vec4u,
  // dt, inverse dt, inverse dt squared, maximum local step
  time: vec4f,
  // gravity xyz, floor height
  gravityFloor: vec4f,
  // floor stiffness, relative regularization, polar epsilon, velocity damping
  solver: vec4f,
  // grounded tangential damping rate (s^-1), contact margin, reserved
  contact: vec4f,
  // local diagnostics, tet diagnostics, assembled vertex energy, convergence gradients
  offsets3: vec4u,
  // globalization control, history, history capacity, enabled
  offsets4: vec4u,
  // residual tolerance, normalized update tolerance, initial scene scale, reserved
  convergence: vec4f,
}

// 40 bytes/lane * 128 lanes = 5,120 bytes.
struct CandidateReductionLane {
  minimumDeterminants: vec2f,
  energies: vec2f,
  maximumNormalizedShift: f32,
  localFailureCount: u32,
  maximumBacktracks: u32,
  skippedGeometryTrials: u32,
  totalEnergyEvaluations: u32,
  validityBits: u32,
}

// 32 bytes/lane * 128 lanes = 4,096 bytes. Together the two reduction
// scratch arrays consume 9,216 bytes, below WebGPU's 16 KiB minimum limit.
struct ConvergenceReductionLane {
  firstFourChannels: vec4f,
  finalChannels: vec2f,
  maximumUpdate: f32,
  valid: u32,
}

@group(0) @binding(0)
var<storage, read_write> dynamicData: array<vec4f>;

@group(0) @binding(1)
var<storage, read> vertexData: array<VertexStatic>;

@group(0) @binding(2)
var<storage, read> tetData: array<TetStatic>;

@group(0) @binding(3)
var<storage, read> restStiffness: array<f32>;

@group(0) @binding(4)
var<storage, read> adjacency: array<u32>;

@group(0) @binding(5)
var<storage, read> cubatureWords: array<u32>;

@group(0) @binding(6)
var<uniform> params: SimParams;

var<workgroup> candidateReductionLanes:
  array<CandidateReductionLane, WORKGROUP_SIZE>;
var<workgroup> convergenceReductionLanes:
  array<ConvergenceReductionLane, WORKGROUP_SIZE>;

${stableNeoHookeanWgsl}
${jgs2GlobalizationWgsl}

fn zeroMat3() -> mat3x3f {
  return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
}

fn identityMat3() -> mat3x3f {
  return mat3x3f(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0),
    vec3f(0.0, 0.0, 1.0),
  );
}

fn finiteSquaredLength(value: vec3f) -> bool {
  let squared = dot(value, value);
  return squared >= 0.0 && squared < 1.0e30;
}

fn globalizationFiniteVec4(value: vec4f) -> bool {
  return jgs2_globalization_finite_scalar(value.x) &&
    jgs2_globalization_finite_scalar(value.y) &&
    jgs2_globalization_finite_scalar(value.z) &&
    jgs2_globalization_finite_scalar(value.w);
}

fn globalizationValidityBits(packed: f32) -> u32 {
  if (
    !jgs2_globalization_finite_scalar(packed) ||
    packed < 0.0 ||
    packed > f32(VALIDITY_BITS_MASK) ||
    packed != floor(packed)
  ) {
    return 0u;
  }
  return u32(packed);
}

fn finiteNonnegativeInteger(value: f32, maximum: f32) -> bool {
  return jgs2_globalization_finite_scalar(value) &&
    value >= 0.0 && value <= maximum && value == floor(value);
}

fn safeNormalize(value: vec3f, fallback: vec3f) -> vec3f {
  let squared = dot(value, value);
  if (squared > params.solver.z * params.solver.z && squared < 1.0e30) {
    return value * inverseSqrt(squared);
  }
  return fallback;
}

fn perpendicularTo(axis: vec3f) -> vec3f {
  var candidate = vec3f(1.0, 0.0, 0.0);
  if (abs(axis.x) > 0.8) {
    candidate = vec3f(0.0, 1.0, 0.0);
  }
  return safeNormalize(cross(axis, candidate), vec3f(0.0, 0.0, 1.0));
}

fn orthonormalize(matrix: mat3x3f) -> mat3x3f {
  let column0 = safeNormalize(matrix[0], vec3f(1.0, 0.0, 0.0));
  let column1Candidate = matrix[1] - column0 * dot(column0, matrix[1]);
  var column1 = safeNormalize(column1Candidate, perpendicularTo(column0));
  var column2 = safeNormalize(cross(column0, column1), vec3f(0.0, 0.0, 1.0));

  if (dot(column2, matrix[2]) < 0.0) {
    column1 = -column1;
    column2 = -column2;
  }
  column1 = safeNormalize(cross(column2, column0), perpendicularTo(column0));
  return mat3x3f(column0, column1, column2);
}

fn matrixDeterminant(matrix: mat3x3f) -> f32 {
  return dot(matrix[0], cross(matrix[1], matrix[2]));
}

fn inverseTranspose(matrix: mat3x3f, determinantValue: f32) -> mat3x3f {
  let reciprocal = 1.0 / determinantValue;
  return reciprocal * mat3x3f(
    cross(matrix[1], matrix[2]),
    cross(matrix[2], matrix[0]),
    cross(matrix[0], matrix[1]),
  );
}

fn polarRotation(matrix: mat3x3f) -> mat3x3f {
  var rotation = matrix;
  for (var iteration = 0u; iteration < 7u; iteration += 1u) {
    let determinantValue = matrixDeterminant(rotation);
    let magnitude = abs(determinantValue);
    if (!(magnitude > params.solver.z) || magnitude > 1.0e20) {
      break;
    }
    rotation = 0.5 * (rotation + inverseTranspose(rotation, determinantValue));
  }
  return orthonormalize(rotation);
}

fn loadPosition(offset: u32, vertex: u32) -> vec3f {
  return dynamicData[offset + vertex].xyz;
}

fn loadRotation(offset: u32, index: u32) -> mat3x3f {
  let base = offset + index * 3u;
  return mat3x3f(
    dynamicData[base].xyz,
    dynamicData[base + 1u].xyz,
    dynamicData[base + 2u].xyz,
  );
}

fn storeRotation(offset: u32, index: u32, rotation: mat3x3f) {
  let base = offset + index * 3u;
  dynamicData[base] = vec4f(rotation[0], 0.0);
  dynamicData[base + 1u] = vec4f(rotation[1], 0.0);
  dynamicData[base + 2u] = vec4f(rotation[2], 0.0);
}

fn stiffnessBlock(tet: u32, rowVertex: u32, columnVertex: u32) -> mat3x3f {
  let base = tet * 144u;
  let row0 = rowVertex * 3u;
  let column0 = columnVertex * 3u;
  return mat3x3f(
    vec3f(
      restStiffness[base + row0 * 12u + column0],
      restStiffness[base + (row0 + 1u) * 12u + column0],
      restStiffness[base + (row0 + 2u) * 12u + column0],
    ),
    vec3f(
      restStiffness[base + row0 * 12u + column0 + 1u],
      restStiffness[base + (row0 + 1u) * 12u + column0 + 1u],
      restStiffness[base + (row0 + 2u) * 12u + column0 + 1u],
    ),
    vec3f(
      restStiffness[base + row0 * 12u + column0 + 2u],
      restStiffness[base + (row0 + 1u) * 12u + column0 + 2u],
      restStiffness[base + (row0 + 2u) * 12u + column0 + 2u],
    ),
  );
}

fn localTetSlot(indices: vec4u, vertex: u32) -> u32 {
  if (indices.x == vertex) { return 0u; }
  if (indices.y == vertex) { return 1u; }
  if (indices.z == vertex) { return 2u; }
  if (indices.w == vertex) { return 3u; }
  return 4u;
}

fn usesStableNeoHookean(tet: u32) -> bool {
  return tetData[tet].attributes.w >= 0.5;
}

fn tetrahedronInverseDm(tet: u32) -> mat3x3f {
  let item = tetData[tet];
  return mat3x3f(item.invDm0.xyz, item.invDm1.xyz, item.invDm2.xyz);
}

fn tetrahedronDeformationGradientAt(
  tet: u32,
  positionOffset: u32,
) -> mat3x3f {
  let item = tetData[tet];
  let position0 = loadPosition(positionOffset, item.indices.x);
  let deformedShape = mat3x3f(
    loadPosition(positionOffset, item.indices.y) - position0,
    loadPosition(positionOffset, item.indices.z) - position0,
    loadPosition(positionOffset, item.indices.w) - position0,
  );
  return deformedShape * tetrahedronInverseDm(tet);
}

fn tetrahedronDeformationGradient(tet: u32) -> mat3x3f {
  return tetrahedronDeformationGradientAt(tet, params.offsets1.w);
}

fn tetrahedronShapeGradients(tet: u32) -> array<vec3f, 4> {
  let inverseDmRows = transpose(tetrahedronInverseDm(tet));
  var gradients: array<vec3f, 4>;
  gradients[1] = inverseDmRows[0];
  gradients[2] = inverseDmRows[1];
  gradients[3] = inverseDmRows[2];
  gradients[0] = -gradients[1] - gradients[2] - gradients[3];
  return gradients;
}

fn stableTetrahedronGradientAt(
  tet: u32,
  positionOffset: u32,
) -> array<vec3f, 4> {
  let item = tetData[tet];
  let firstPiola = snh_first_piola(
    tetrahedronDeformationGradientAt(tet, positionOffset),
    item.attributes.y,
    item.attributes.z,
  );
  let gradients = tetrahedronShapeGradients(tet);
  var result: array<vec3f, 4>;
  for (var local = 0u; local < 4u; local += 1u) {
    result[local] = item.attributes.x * firstPiola * gradients[local];
  }
  return result;
}

fn stableTetrahedronGradient(tet: u32) -> array<vec3f, 4> {
  return stableTetrahedronGradientAt(tet, params.offsets1.w);
}

fn stableTetrahedronHessianProduct(
  tet: u32,
  directions: array<vec3f, 4>,
) -> array<vec3f, 4> {
  let item = tetData[tet];
  let gradients = tetrahedronShapeGradients(tet);
  var deformationDirection = zeroMat3();
  for (var local = 0u; local < 4u; local += 1u) {
    deformationDirection += mat3x3f(
      directions[local] * gradients[local].x,
      directions[local] * gradients[local].y,
      directions[local] * gradients[local].z,
    );
  }
  let stressDirection = snh_tangent_product(
    tetrahedronDeformationGradient(tet),
    deformationDirection,
    item.attributes.y,
    item.attributes.z,
  );
  var result: array<vec3f, 4>;
  for (var local = 0u; local < 4u; local += 1u) {
    result[local] = item.attributes.x * stressDirection * gradients[local];
  }
  return result;
}

fn stableTetrahedronLocalHessian(tet: u32, localVertex: u32) -> mat3x3f {
  let item = tetData[tet];
  let gradients = tetrahedronShapeGradients(tet);
  let gradient = gradients[localVertex];
  let deformationGradient = tetrahedronDeformationGradient(tet);
  var columns: array<vec3f, 3>;
  for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
    var axis = vec3f(0.0);
    axis[coordinate] = 1.0;
    let direction = mat3x3f(
      axis * gradient.x,
      axis * gradient.y,
      axis * gradient.z,
    );
    columns[coordinate] = item.attributes.x * snh_tangent_product(
      deformationGradient,
      direction,
      item.attributes.y,
      item.attributes.z,
    ) * gradient;
  }
  return mat3x3f(columns[0], columns[1], columns[2]);
}

fn tetGradient(tet: u32) -> array<vec3f, 4> {
  if (usesStableNeoHookean(tet)) {
    return stableTetrahedronGradient(tet);
  }
  let indices = tetData[tet].indices;
  var current: array<vec3f, 4>;
  var rest: array<vec3f, 4>;
  current[0] = loadPosition(params.offsets1.w, indices.x);
  current[1] = loadPosition(params.offsets1.w, indices.y);
  current[2] = loadPosition(params.offsets1.w, indices.z);
  current[3] = loadPosition(params.offsets1.w, indices.w);
  rest[0] = vertexData[indices.x].restMass.xyz;
  rest[1] = vertexData[indices.y].restMass.xyz;
  rest[2] = vertexData[indices.z].restMass.xyz;
  rest[3] = vertexData[indices.w].restMass.xyz;

  let currentCenter = 0.25 * (current[0] + current[1] + current[2] + current[3]);
  let restCenter = 0.25 * (rest[0] + rest[1] + rest[2] + rest[3]);
  let rotation = loadRotation(params.offsets1.z, tet);
  let inverseRotation = transpose(rotation);
  var displacement: array<vec3f, 4>;
  var gradient: array<vec3f, 4>;

  for (var local = 0u; local < 4u; local += 1u) {
    displacement[local] =
      inverseRotation * (current[local] - currentCenter) - (rest[local] - restCenter);
  }
  for (var row = 0u; row < 4u; row += 1u) {
    var localGradient = vec3f(0.0);
    for (var column = 0u; column < 4u; column += 1u) {
      localGradient += stiffnessBlock(tet, row, column) * displacement[column];
    }
    gradient[row] = rotation * localGradient;
  }
  return gradient;
}

fn cubatureFloat(index: u32) -> f32 {
  return bitcast<f32>(cubatureWords[index]);
}

fn cubatureBlock(recordBase: u32, localVertex: u32) -> mat3x3f {
  let base = recordBase + 2u + localVertex * 9u;
  // CPU storage is row-major. WGSL matrix constructors take columns.
  return mat3x3f(
    vec3f(cubatureFloat(base), cubatureFloat(base + 3u), cubatureFloat(base + 6u)),
    vec3f(cubatureFloat(base + 1u), cubatureFloat(base + 4u), cubatureFloat(base + 7u)),
    vec3f(cubatureFloat(base + 2u), cubatureFloat(base + 5u), cubatureFloat(base + 8u)),
  );
}

fn outerProductColumns(left: vec3f, right: vec3f) -> mat3x3f {
  return mat3x3f(
    left * right.x,
    left * right.y,
    left * right.z,
  );
}

fn currentCubatureBasis(
  recordBase: u32,
  sourceVertex: u32,
  indices: vec4u,
) -> array<mat3x3f, 4> {
  let sourceRotation = loadRotation(params.offsets1.y, sourceVertex);
  var basis: array<mat3x3f, 4>;
  basis[0] = loadRotation(params.offsets1.y, indices.x) *
    cubatureBlock(recordBase, 0u) * transpose(sourceRotation);
  basis[1] = loadRotation(params.offsets1.y, indices.y) *
    cubatureBlock(recordBase, 1u) * transpose(sourceRotation);
  basis[2] = loadRotation(params.offsets1.y, indices.z) *
    cubatureBlock(recordBase, 2u) * transpose(sourceRotation);
  basis[3] = loadRotation(params.offsets1.y, indices.w) *
    cubatureBlock(recordBase, 3u) * transpose(sourceRotation);
  return basis;
}

fn exactSourceTrialDeformationGradient(
  tet: u32,
  sourceVertex: u32,
  displacement: vec3f,
) -> mat3x3f {
  let slot = localTetSlot(tetData[tet].indices, sourceVertex);
  if (slot >= 4u) {
    return tetrahedronDeformationGradient(tet);
  }
  let gradients = tetrahedronShapeGradients(tet);
  return tetrahedronDeformationGradient(tet) +
    outerProductColumns(displacement, gradients[slot]);
}

fn projectedTrialDeformationGradient(
  tet: u32,
  basis: array<mat3x3f, 4>,
  displacement: vec3f,
) -> mat3x3f {
  let gradients = tetrahedronShapeGradients(tet);
  var result = tetrahedronDeformationGradient(tet);
  for (var local = 0u; local < 4u; local += 1u) {
    result += outerProductColumns(basis[local] * displacement, gradients[local]);
  }
  return result;
}

fn quadraticEnergyDelta(
  basePosition: vec3f,
  targetPosition: vec3f,
  displacement: vec3f,
  weight: f32,
) -> f32 {
  let residual = basePosition - targetPosition;
  return weight * (
    dot(residual, displacement) + 0.5 * dot(displacement, displacement)
  );
}

fn floorEnergyDelta(baseY: f32, displacementY: f32) -> f32 {
  if (!(params.solver.x > 0.0)) {
    return 0.0;
  }
  let initialPenetration = min(0.0, baseY - params.gravityFloor.w);
  let finalPenetration = min(
    0.0,
    baseY + displacementY - params.gravityFloor.w,
  );
  return 0.5 * params.solver.x *
    (finalPenetration - initialPenetration) *
    (finalPenetration + initialPenetration);
}

fn stableMaterialEnergyDelta(
  tet: u32,
  initial: mat3x3f,
  finalDeformation: mat3x3f,
) -> f32 {
  let attributes = tetData[tet].attributes;
  return attributes.x *
    jgs2_globalization_stable_neo_hookean_energy_density_delta(
      initial,
      finalDeformation,
      attributes.y,
      attributes.z,
    );
}

// Geometry is deliberately a separate routine. jgs2Solve calls it in an if
// branch before the restricted energy routine for every Armijo candidate.
struct RestrictedGeometryResult {
  minimumDeformationDeterminant: f32,
  valid: u32,
}

fn restrictedMinimumDeformationDeterminant(
  sourceVertex: u32,
  displacement: vec3f,
) -> RestrictedGeometryResult {
  let sourceItem = vertexData[sourceVertex];
  var minimumDeterminant = jgs2_globalization_f32_max;
  let preflightGeometry = dynamicData[params.offsets4.x];
  let preflightValidity = globalizationValidityBits(preflightGeometry.w);
  if (
    (preflightValidity & ACCEPTED_MINIMUM_VALID_BIT) != 0u &&
    jgs2_globalization_finite_scalar(preflightGeometry.z)
  ) {
    // Unchanged tetrahedra retain the accepted source-pose minimum. Changed
    // incident/projected tetrahedra below can only lower this conservative
    // restricted minimum.
    minimumDeterminant = preflightGeometry.z;
  }
  for (var adjacent = 0u; adjacent < sourceItem.info.y; adjacent += 1u) {
    let tet = adjacency[sourceItem.info.x + adjacent];
    let determinantValue = snh_determinant(
      exactSourceTrialDeformationGradient(tet, sourceVertex, displacement),
    );
    if (!jgs2_globalization_finite_scalar(determinantValue)) {
      return RestrictedGeometryResult(0.0, 0u);
    }
    minimumDeterminant = min(minimumDeterminant, determinantValue);
  }

  for (var sample = 0u; sample < params.counts.z; sample += 1u) {
    let record = sourceVertex * params.counts.z + sample;
    let recordBase = record * CUBATURE_RECORD_WORDS;
    let tet = cubatureWords[recordBase];
    let weight = cubatureFloat(recordBase + 1u);
    if (tet == EMPTY_TET || tet >= params.counts.y || !(weight > 0.0)) {
      continue;
    }
    let indices = tetData[tet].indices;
    let basis = currentCubatureBasis(recordBase, sourceVertex, indices);
    let determinantValue = snh_determinant(
      projectedTrialDeformationGradient(tet, basis, displacement),
    );
    if (!jgs2_globalization_finite_scalar(determinantValue)) {
      return RestrictedGeometryResult(0.0, 0u);
    }
    minimumDeterminant = min(minimumDeterminant, determinantValue);
  }
  return RestrictedGeometryResult(minimumDeterminant, 1u);
}

fn restrictedEnergyDelta(
  sourceVertex: u32,
  displacement: vec3f,
) -> f32 {
  let sourceItem = vertexData[sourceVertex];
  let sourcePosition = loadPosition(params.offsets1.w, sourceVertex);
  let sourceInertia = sourceItem.restMass.w * params.time.z;
  var result = quadraticEnergyDelta(
    sourcePosition,
    loadPosition(params.offsets0.z, sourceVertex),
    displacement,
    sourceInertia,
  ) + floorEnergyDelta(sourcePosition.y, displacement.y);

  for (var adjacent = 0u; adjacent < sourceItem.info.y; adjacent += 1u) {
    let tet = adjacency[sourceItem.info.x + adjacent];
    let initial = tetrahedronDeformationGradient(tet);
    let finalDeformation = exactSourceTrialDeformationGradient(
      tet,
      sourceVertex,
      displacement,
    );
    result += stableMaterialEnergyDelta(tet, initial, finalDeformation);
  }

  for (var sample = 0u; sample < params.counts.z; sample += 1u) {
    let record = sourceVertex * params.counts.z + sample;
    let recordBase = record * CUBATURE_RECORD_WORDS;
    let tet = cubatureWords[recordBase];
    let weight = cubatureFloat(recordBase + 1u);
    if (tet == EMPTY_TET || tet >= params.counts.y || !(weight > 0.0)) {
      continue;
    }
    let indices = tetData[tet].indices;
    let basis = currentCubatureBasis(recordBase, sourceVertex, indices);
    let initial = tetrahedronDeformationGradient(tet);
    let projected = projectedTrialDeformationGradient(
      tet,
      basis,
      displacement,
    );
    var complementary = stableMaterialEnergyDelta(
      tet,
      initial,
      projected,
    );
    for (var local = 0u; local < 4u; local += 1u) {
      let vertex = indices[local];
      let localDisplacement = basis[local] * displacement;
      complementary += quadraticEnergyDelta(
        loadPosition(params.offsets1.w, vertex),
        loadPosition(params.offsets0.z, vertex),
        localDisplacement,
        distributedInertiaWeight(vertex),
      );
    }

    let sourceSlot = localTetSlot(indices, sourceVertex);
    if (sourceSlot < 4u) {
      complementary -= stableMaterialEnergyDelta(
        tet,
        initial,
        exactSourceTrialDeformationGradient(
          tet,
          sourceVertex,
          displacement,
        ),
      );
      complementary -= quadraticEnergyDelta(
        sourcePosition,
        loadPosition(params.offsets0.z, sourceVertex),
        displacement,
        distributedInertiaWeight(sourceVertex),
      );
    }
    result += weight * complementary;
  }
  return result;
}

fn cubatureHessian(
  tet: u32,
  basis: array<mat3x3f, 4>,
) -> mat3x3f {
  var reducedColumns: array<vec3f, 3>;

  if (usesStableNeoHookean(tet)) {
    for (var column = 0u; column < 3u; column += 1u) {
      var direction: array<vec3f, 4>;
      for (var local = 0u; local < 4u; local += 1u) {
        direction[local] = basis[local][column];
      }
      let product = stableTetrahedronHessianProduct(tet, direction);
      var reducedColumn = vec3f(0.0);
      for (var row = 0u; row < 3u; row += 1u) {
        var entry = 0.0;
        for (var local = 0u; local < 4u; local += 1u) {
          entry += dot(basis[local][row], product[local]);
        }
        reducedColumn[row] = entry;
      }
      reducedColumns[column] = reducedColumn;
    }
    return mat3x3f(
      reducedColumns[0],
      reducedColumns[1],
      reducedColumns[2],
    );
  }

  let rotation = loadRotation(params.offsets1.z, tet);
  let inverseRotation = transpose(rotation);

  // Apply the rotated 12x12 stiffness to each of the three basis columns.
  for (var column = 0u; column < 3u; column += 1u) {
    var localDirection: array<vec3f, 4>;
    var worldProduct: array<vec3f, 4>;
    for (var local = 0u; local < 4u; local += 1u) {
      localDirection[local] = inverseRotation * basis[local][column];
    }
    for (var rowVertex = 0u; rowVertex < 4u; rowVertex += 1u) {
      var localProduct = vec3f(0.0);
      for (var columnVertex = 0u; columnVertex < 4u; columnVertex += 1u) {
        localProduct +=
          stiffnessBlock(tet, rowVertex, columnVertex) * localDirection[columnVertex];
      }
      worldProduct[rowVertex] = rotation * localProduct;
    }

    var reducedColumn = vec3f(0.0);
    for (var row = 0u; row < 3u; row += 1u) {
      var entry = 0.0;
      for (var local = 0u; local < 4u; local += 1u) {
        entry += dot(basis[local][row], worldProduct[local]);
      }
      reducedColumn[row] = entry;
    }
    reducedColumns[column] = reducedColumn;
  }
  return mat3x3f(reducedColumns[0], reducedColumns[1], reducedColumns[2]);
}

fn distributedInertiaWeight(vertex: u32) -> f32 {
  let item = vertexData[vertex];
  if (item.info.z != 0u) {
    return 0.0;
  }
  let incidentCount = max(item.info.y, 1u);
  return item.restMass.w * params.time.z / f32(incidentCount);
}

fn distributedInertiaGradient(vertex: u32) -> vec3f {
  let weight = distributedInertiaWeight(vertex);
  return weight * (
    loadPosition(params.offsets1.w, vertex) -
    loadPosition(params.offsets0.z, vertex)
  );
}

fn cubatureInertiaHessian(
  indices: vec4u,
  basis: array<mat3x3f, 4>,
) -> mat3x3f {
  var result = zeroMat3();
  for (var local = 0u; local < 4u; local += 1u) {
    let vertex = indices[local];
    result += distributedInertiaWeight(vertex) *
      transpose(basis[local]) * basis[local];
  }
  return result;
}

fn regularizedSolve(matrix: mat3x3f, rightHandSide: vec3f) -> vec3f {
  let diagonalScale = max(
    1.0,
    max(abs(matrix[0][0]), max(abs(matrix[1][1]), abs(matrix[2][2]))),
  );
  let shift = max(params.solver.y * diagonalScale, 1.0e-9);
  let a00 = matrix[0][0] + shift;
  let a11 = matrix[1][1] + shift;
  let a22 = matrix[2][2] + shift;
  let a01 = 0.5 * (matrix[0][1] + matrix[1][0]);
  let a02 = 0.5 * (matrix[0][2] + matrix[2][0]);
  let a12 = 0.5 * (matrix[1][2] + matrix[2][1]);

  // Modified Cholesky keeps the tiny local solve finite even if an element
  // Hessian is numerically indefinite in f32.
  let l00 = sqrt(max(a00, shift));
  let l10 = a01 / l00;
  let l20 = a02 / l00;
  let l11 = sqrt(max(a11 - l10 * l10, shift));
  let l21 = (a12 - l20 * l10) / l11;
  let l22 = sqrt(max(a22 - l20 * l20 - l21 * l21, shift));

  let y0 = rightHandSide.x / l00;
  let y1 = (rightHandSide.y - l10 * y0) / l11;
  let y2 = (rightHandSide.z - l20 * y0 - l21 * y1) / l22;
  let x2 = y2 / l22;
  let x1 = (y1 - l21 * x2) / l11;
  let x0 = (y0 - l10 * x1 - l20 * x2) / l00;
  let result = vec3f(x0, x1, x2);
  if (!finiteSquaredLength(result)) {
    return vec3f(0.0);
  }
  return result;
}

fn localGlobalizationBase(vertex: u32) -> u32 {
  return params.offsets3.x + vertex * LOCAL_GLOBALIZATION_STRIDE;
}

fn storeLocalGlobalizationDiagnostic(
  vertex: u32,
  direction: vec3f,
  alpha: f32,
  minimumEigenvalue: f32,
  scale: f32,
  diagonalShift: f32,
  normalizedShift: f32,
  acceptedEnergyDelta: f32,
  armijoDeltaBound: f32,
  gradientDotDirection: f32,
  shiftedResidual: f32,
  minimumTrialDeterminant: f32,
  backtrackCount: u32,
  energyEvaluationCount: u32,
  status: u32,
) {
  let base = localGlobalizationBase(vertex);
  dynamicData[base] = vec4f(direction, alpha);
  dynamicData[base + 1u] = vec4f(
    minimumEigenvalue,
    scale,
    diagonalShift,
    normalizedShift,
  );
  dynamicData[base + 2u] = vec4f(
    acceptedEnergyDelta,
    armijoDeltaBound,
    gradientDotDirection,
    shiftedResidual,
  );
  dynamicData[base + 3u] = vec4f(
    minimumTrialDeterminant,
    f32(backtrackCount),
    f32(energyEvaluationCount),
    f32(status),
  );
}

fn vertexImplicitEnergyAt(positionOffset: u32, vertex: u32) -> f32 {
  if (vertexData[vertex].info.z != 0u) {
    return 0.0;
  }
  let position = loadPosition(positionOffset, vertex);
  let predicted = loadPosition(params.offsets0.z, vertex);
  let inertia = vertexData[vertex].restMass.w * params.time.z;
  let residual = position - predicted;
  var energy = 0.5 * inertia * dot(residual, residual);
  if (params.solver.x > 0.0) {
    let penetration = min(0.0, position.y - params.gravityFloor.w);
    energy += 0.5 * params.solver.x * penetration * penetration;
  }
  return energy;
}

fn storeAssembledVertexEnergy(vertex: u32) {
  let sourceEnergy = vertexImplicitEnergyAt(params.offsets1.w, vertex);
  let candidateEnergy = vertexImplicitEnergyAt(params.offsets2.x, vertex);
  let sourceValid = jgs2_globalization_finite_scalar(sourceEnergy);
  let candidateValid = jgs2_globalization_finite_scalar(candidateEnergy);
  dynamicData[params.offsets3.z + vertex] = vec4f(
    select(0.0, sourceEnergy, sourceValid),
    select(0.0, candidateEnergy, candidateValid),
    select(0.0, 1.0, sourceValid),
    select(0.0, 1.0, candidateValid),
  );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn predict(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }

  let oldPosition = loadPosition(params.offsets0.x, vertex);
  dynamicData[params.offsets1.x + vertex] = vec4f(oldPosition, 1.0);
  let pinned = vertexData[vertex].info.z != 0u;
  var predicted = vertexData[vertex].restMass.xyz;
  if (!pinned) {
    let velocity = dynamicData[params.offsets0.w + vertex].xyz;
    let dampedVelocity = params.solver.w * velocity;
    predicted = oldPosition + params.time.x * dampedVelocity +
      params.time.x * params.time.x * params.gravityFloor.xyz;
  } else {
    dynamicData[params.offsets0.w + vertex] = vec4f(0.0);
  }
  dynamicData[params.offsets0.z + vertex] = vec4f(predicted, 1.0);
  // Stable nonlinear solves start at the last accepted feasible pose. The
  // predicted position is only the implicit-Euler inertial target. Legacy
  // co-rotated regression scenes retain their historical predictor start.
  let iterationSource = select(predicted, oldPosition, params.offsets4.w != 0u);
  dynamicData[params.offsets0.y + vertex] = vec4f(iterationSource, 1.0);
  // A solve later in this frame makes the update record valid again.
  dynamicData[params.offsets2.z + vertex] = vec4f(0.0);
  if (vertex == 0u && params.offsets4.w != 0u) {
    for (var record = 0u; record < GLOBALIZATION_CONTROL_STRIDE; record += 1u) {
      dynamicData[params.offsets4.x + record] = vec4f(0.0);
    }
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn tetPolarRotation(@builtin(global_invocation_id) globalId: vec3u) {
  let tet = globalId.x;
  if (tet >= params.counts.y) {
    return;
  }
  if (usesStableNeoHookean(tet)) {
    storeRotation(params.offsets1.z, tet, identityMat3());
    return;
  }

  let item = tetData[tet];
  let position0 = loadPosition(params.offsets1.w, item.indices.x);
  let position1 = loadPosition(params.offsets1.w, item.indices.y);
  let position2 = loadPosition(params.offsets1.w, item.indices.z);
  let position3 = loadPosition(params.offsets1.w, item.indices.w);
  let deformedShape = mat3x3f(
    position1 - position0,
    position2 - position0,
    position3 - position0,
  );
  let inverseDm = mat3x3f(item.invDm0.xyz, item.invDm1.xyz, item.invDm2.xyz);
  storeRotation(params.offsets1.z, tet, polarRotation(deformedShape * inverseDm));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn vertexPolarRotation(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }

  let info = vertexData[vertex].info;
  if (info.y == 0u) {
    storeRotation(params.offsets1.y, vertex, identityMat3());
    return;
  }

  var average = zeroMat3();
  var totalVolume = 0.0;
  for (var adjacent = 0u; adjacent < info.y; adjacent += 1u) {
    let tet = adjacency[info.x + adjacent];
    let volumeWeight = tetData[tet].attributes.x;
    average += volumeWeight * tetrahedronDeformationGradient(tet);
    totalVolume += volumeWeight;
  }
  if (totalVolume > 0.0) {
    average = (1.0 / totalVolume) * average;
  } else {
    average = identityMat3();
  }
  storeRotation(params.offsets1.y, vertex, polarRotation(average));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn jgs2Solve(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }

  let vertexItem = vertexData[vertex];
  if (vertexItem.info.z != 0u) {
    dynamicData[params.offsets2.x + vertex] = vec4f(vertexItem.restMass.xyz, 1.0);
    dynamicData[params.offsets2.z + vertex] = vec4f(0.0, 0.0, 0.0, 1.0);
    if (params.offsets4.w != 0u) {
      storeLocalGlobalizationDiagnostic(
        vertex,
        vec3f(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0u,
        0u,
        LOCAL_STATUS_PINNED,
      );
      storeAssembledVertexEnergy(vertex);
    }
    return;
  }

  let position = loadPosition(params.offsets1.w, vertex);
  if (params.offsets4.w != 0u) {
    let preflightGeometry = dynamicData[params.offsets4.x];
    let preflightValidity = globalizationValidityBits(preflightGeometry.w);
    let acceptedMinimumValid =
      (preflightValidity & ACCEPTED_MINIMUM_VALID_BIT) != 0u &&
      jgs2_globalization_finite_scalar(preflightGeometry.z);
    let sourceFeasible = acceptedMinimumValid &&
      preflightGeometry.z > jgs2_globalization_determinant_floor;
    if (!sourceFeasible) {
      // The accepted global minimum describes the complete current source
      // pose. Reject before any material gradient/Hessian evaluation so an
      // invalid or already-infeasible source cannot enter local arithmetic.
      dynamicData[params.offsets2.x + vertex] =
        dynamicData[params.offsets1.w + vertex];
      dynamicData[params.offsets2.z + vertex] =
        vec4f(0.0, 0.0, 0.0, 1.0);
      storeLocalGlobalizationDiagnostic(
        vertex,
        vec3f(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        select(0.0, preflightGeometry.z, acceptedMinimumValid),
        0u,
        0u,
        select(
          LOCAL_STATUS_NONFINITE,
          LOCAL_STATUS_LINE_SEARCH_FAILED,
          acceptedMinimumValid,
        ),
      );
      storeAssembledVertexEnergy(vertex);
      return;
    }
  }
  let predicted = loadPosition(params.offsets0.z, vertex);
  let inertia = vertexItem.restMass.w * params.time.z;
  var gradient = inertia * (position - predicted);
  var hessian = inertia * identityMat3();

  // Exact local restriction: inertia plus every incident tetrahedron.
  for (var adjacent = 0u; adjacent < vertexItem.info.y; adjacent += 1u) {
    let tet = adjacency[vertexItem.info.x + adjacent];
    let slot = localTetSlot(tetData[tet].indices, vertex);
    if (slot < 4u) {
      let tetGradients = tetGradient(tet);
      let rotation = loadRotation(params.offsets1.z, tet);
      gradient += tetGradients[slot];
      if (usesStableNeoHookean(tet)) {
        hessian += stableTetrahedronLocalHessian(tet, slot);
      } else {
        hessian += rotation * stiffnessBlock(tet, slot, slot) *
          transpose(rotation);
      }
    }
  }

  // A one-sided quadratic penalty keeps the first demos deterministic and
  // avoids requiring a CPU collision pipeline.
  if (params.solver.x > 0.0 && position.y < params.gravityFloor.w) {
    gradient.y += params.solver.x * (position.y - params.gravityFloor.w);
    hessian[1][1] += params.solver.x;
  }

  // Cubature supplies the complementary gradient and Hessian in Eq. 15-17.
  for (var sample = 0u; sample < params.counts.z; sample += 1u) {
    let record = vertex * params.counts.z + sample;
    let recordBase = record * CUBATURE_RECORD_WORDS;
    let tet = cubatureWords[recordBase];
    let weight = cubatureFloat(recordBase + 1u);
    if (tet == EMPTY_TET || tet >= params.counts.y || !(weight > 0.0)) {
      continue;
    }
    let indices = tetData[tet].indices;

    let basis = currentCubatureBasis(recordBase, vertex, indices);

    let tetGradients = tetGradient(tet);
    var projectedGradient = vec3f(0.0);
    for (var local = 0u; local < 4u; local += 1u) {
      let sampleGradient = tetGradients[local] +
        distributedInertiaGradient(indices[local]);
      projectedGradient += transpose(basis[local]) * sampleGradient;
    }
    var projectedHessian =
      cubatureHessian(tet, basis) + cubatureInertiaHessian(indices, basis);

    // Cubature is trained on the remainder after the exact local restriction.
    // If this sample is incident to the source vertex, its full projection
    // contains the source elastic block and one distributed source-inertia
    // share. Both are already present in the exact gather above (the full
    // source inertia is the sum of its shares), so remove them before applying
    // the learned weight. Neighbor/cross terms from the same incident tet stay
    // in the remainder, which is required by Eq. 15-17.
    let sourceSlot = localTetSlot(indices, vertex);
    if (sourceSlot < 4u) {
      let elementRotation = loadRotation(params.offsets1.z, tet);
      projectedGradient -= tetGradients[sourceSlot] +
        distributedInertiaGradient(vertex);
      if (usesStableNeoHookean(tet)) {
        projectedHessian -=
          stableTetrahedronLocalHessian(tet, sourceSlot) +
          distributedInertiaWeight(vertex) * identityMat3();
      } else {
        projectedHessian -=
          elementRotation * stiffnessBlock(tet, sourceSlot, sourceSlot) *
            transpose(elementRotation) +
          distributedInertiaWeight(vertex) * identityMat3();
      }
    }

    gradient += weight * projectedGradient;
    hessian += weight * projectedHessian;
  }

  if (params.offsets4.w != 0u) {
    let directionResult = jgs2_globalization_solve(
      hessian,
      gradient,
      inertia,
    );
    var localStatus = LOCAL_STATUS_NONFINITE;
    let fullStepPosition = position + directionResult.direction;
    let fullStepDisplacement = fullStepPosition - position;
    let sourceCoordinateScale = max(
      abs(position.x),
      max(abs(position.y), abs(position.z)),
    );
    let positionResolution = max(
      params.convergence.z,
      max(sourceCoordinateScale, 1.0e-12),
    ) *
      jgs2_globalization_f32_unit_roundoff *
      jgs2_globalization_position_resolution_multiplier;
    let belowPositionResolution =
      directionResult.status == jgs2_globalization_status_accepted &&
      jgs2_globalization_finite_vec3(fullStepPosition) &&
      jgs2_globalization_finite_vec3(fullStepDisplacement) &&
      jgs2_globalization_scaled_length3(fullStepDisplacement) <=
        positionResolution;
    if (directionResult.status == jgs2_globalization_status_accepted) {
      localStatus = select(
        LOCAL_STATUS_LINE_SEARCH_FAILED,
        LOCAL_STATUS_ZERO_GRADIENT,
        belowPositionResolution,
      );
    } else if (
      directionResult.status == jgs2_globalization_status_zero_gradient
    ) {
      localStatus = LOCAL_STATUS_ZERO_GRADIENT;
    } else if (
      directionResult.status == jgs2_globalization_status_shift_limit_exceeded
    ) {
      localStatus = LOCAL_STATUS_SHIFT_LIMIT;
    } else if (
      directionResult.status == jgs2_globalization_status_factorization_failed
    ) {
      localStatus = LOCAL_STATUS_CHOLESKY_FAILED;
    } else if (
      directionResult.status == jgs2_globalization_status_non_descent_direction
    ) {
      localStatus = LOCAL_STATUS_NON_DESCENT;
    }

    var alpha = 0.0;
    var acceptedEnergyDelta = 0.0;
    var armijoDeltaBound = 0.0;
    var minimumTrialDeterminant = 0.0;
    var backtrackCount = 0u;
    var energyEvaluationCount = 0u;
    var acceptedDisplacement = vec3f(0.0);
    var acceptedPosition = position;
    if (
      directionResult.status == jgs2_globalization_status_accepted &&
      !belowPositionResolution
    ) {
      var trialAlpha = 1.0;
      for (
        var backtrack = 0u;
        backtrack <= jgs2_globalization_max_backtracks;
        backtrack += 1u
      ) {
        let nominalTrialDisplacement =
          trialAlpha * directionResult.direction;
        // Armijo and geometry must observe the exact f32 update that can be
        // stored, not the higher-level alpha*p expression before position
        // addition rounds. Keep the rounded position for the eventual write.
        let trialPosition = position + nominalTrialDisplacement;
        let trialDisplacement = trialPosition - position;
        if (
          !jgs2_globalization_finite_vec3(trialPosition) ||
          !jgs2_globalization_finite_vec3(trialDisplacement)
        ) {
          localStatus = LOCAL_STATUS_NONFINITE;
          minimumTrialDeterminant = 0.0;
          backtrackCount = backtrack;
          break;
        }
        let trialLength = jgs2_globalization_scaled_length3(
          trialDisplacement,
        );
        if (trialLength <= positionResolution) {
          // Further halving cannot produce a useful stored update. This is a
          // deliberate numerical zero, not a successful Armijo step.
          localStatus = LOCAL_STATUS_ZERO_GRADIENT;
          minimumTrialDeterminant = 0.0;
          backtrackCount = backtrack;
          break;
        }
        let trialGradientDotDisplacement = dot(
          gradient,
          trialDisplacement,
        );
        if (
          !jgs2_globalization_finite_scalar(
            trialGradientDotDisplacement,
          )
        ) {
          localStatus = LOCAL_STATUS_NONFINITE;
          minimumTrialDeterminant = 0.0;
          backtrackCount = backtrack;
          break;
        }
        if (!(trialGradientDotDisplacement < 0.0)) {
          localStatus = LOCAL_STATUS_ZERO_GRADIENT;
          minimumTrialDeterminant = 0.0;
          backtrackCount = backtrack;
          break;
        }
        // The geometry branch must remain before restrictedEnergyDelta. It is
        // both a correctness invariant and a test-visible evaluation count.
        let trialGeometry = restrictedMinimumDeformationDeterminant(
          vertex,
          trialDisplacement,
        );
        if (trialGeometry.valid == 0u) {
          // A nonfinite trial is an arithmetic failure with a finite zero
          // sentinel. Never relabel it as an ordinary infeasible geometry.
          localStatus = LOCAL_STATUS_NONFINITE;
          minimumTrialDeterminant = 0.0;
          backtrackCount = backtrack;
          break;
        }
        let trialMinimum = trialGeometry.minimumDeformationDeterminant;
        minimumTrialDeterminant = trialMinimum;
        if (
          trialMinimum > jgs2_globalization_determinant_floor
        ) {
          let trialEnergyDelta = restrictedEnergyDelta(
            vertex,
            trialDisplacement,
          );
          energyEvaluationCount += 1u;
          if (!jgs2_globalization_finite_scalar(trialEnergyDelta)) {
            // The geometry was finite and feasible, but the trial as a whole
            // is numerically invalid. Keep the finite geometry value so the
            // reducer can distinguish this from nonfinite geometry, and do
            // not age it into an ordinary line-search failure.
            localStatus = LOCAL_STATUS_NONFINITE;
            backtrackCount = backtrack;
            break;
          }
          let trialArmijoBound = jgs2_globalization_armijo_c1 *
            trialGradientDotDisplacement;
          if (
            trialEnergyDelta <= trialArmijoBound
          ) {
            alpha = trialAlpha;
            acceptedEnergyDelta = trialEnergyDelta;
            armijoDeltaBound = trialArmijoBound;
            backtrackCount = backtrack;
            localStatus = LOCAL_STATUS_ACCEPTED;
            acceptedDisplacement = trialDisplacement;
            acceptedPosition = trialPosition;
            break;
          }
        }
        trialAlpha *= jgs2_globalization_backtrack_factor;
      }
      if (localStatus == LOCAL_STATUS_LINE_SEARCH_FAILED) {
        backtrackCount = jgs2_globalization_max_backtracks;
      }
    }

    dynamicData[params.offsets2.x + vertex] =
      vec4f(acceptedPosition, 1.0);
    dynamicData[params.offsets2.z + vertex] =
      vec4f(jgs2_globalization_scaled_length3(acceptedDisplacement), 0.0, 0.0, 1.0);
    storeLocalGlobalizationDiagnostic(
      vertex,
      directionResult.direction,
      alpha,
      directionResult.minimum_eigenvalue_normalized * directionResult.scale,
      directionResult.scale,
      directionResult.normalized_shift * directionResult.scale,
      directionResult.normalized_shift,
      acceptedEnergyDelta,
      armijoDeltaBound,
      directionResult.gradient_dot_direction,
      directionResult.shifted_relative_residual,
      minimumTrialDeterminant,
      backtrackCount,
      energyEvaluationCount,
      localStatus,
    );
    storeAssembledVertexEnergy(vertex);
    return;
  }

  var delta = regularizedSolve(hessian, -gradient);
  let deltaLengthSquared = dot(delta, delta);
  let maxStep = params.time.w;
  if (maxStep > 0.0 && deltaLengthSquared > maxStep * maxStep) {
    delta *= maxStep * inverseSqrt(deltaLengthSquared);
  }
  dynamicData[params.offsets2.x + vertex] = vec4f(position + delta, 1.0);
  // Every nonlinear solve overwrites this record. After the command stream
  // finishes it therefore describes the final iteration, including after an
  // even-result copy or velocity finalization.
  dynamicData[params.offsets2.z + vertex] =
    vec4f(length(delta), 0.0, 0.0, 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn assembledVertexEnergy(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }
  storeAssembledVertexEnergy(vertex);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn candidateTetrahedron(@builtin(global_invocation_id) globalId: vec3u) {
  let tet = globalId.x;
  if (tet >= params.counts.y) {
    return;
  }
  let attributes = tetData[tet].attributes;
  let sourceF = tetrahedronDeformationGradientAt(tet, params.offsets1.w);
  let candidateF = tetrahedronDeformationGradientAt(tet, params.offsets2.x);
  let sourceJ = snh_determinant(sourceF);
  let candidateJ = snh_determinant(candidateF);
  let sourceGeometryValid = jgs2_globalization_finite_scalar(sourceJ);
  let candidateGeometryValid = jgs2_globalization_finite_scalar(candidateJ);
  var sourceEnergy = 0.0;
  var candidateEnergy = 0.0;
  var sourceEnergyValid = false;
  var candidateEnergyValid = false;
  // Preserve the same geometry-before-energy ordering as the local line
  // search. Infeasible assembled energy is unavailable rather than evaluated
  // through the inversion-tolerant material formula.
  if (
    sourceGeometryValid &&
    sourceJ > jgs2_globalization_determinant_floor
  ) {
    sourceEnergy = attributes.x * snh_energy_density(
      sourceF,
      attributes.y,
      attributes.z,
    );
    sourceEnergyValid = jgs2_globalization_finite_scalar(sourceEnergy);
  }
  if (
    candidateGeometryValid &&
    candidateJ > jgs2_globalization_determinant_floor
  ) {
    candidateEnergy = attributes.x * snh_energy_density(
      candidateF,
      attributes.y,
      attributes.z,
    );
    candidateEnergyValid = jgs2_globalization_finite_scalar(candidateEnergy);
  }
  let base = params.offsets3.y + tet * TET_GLOBALIZATION_STRIDE;
  dynamicData[base] = vec4f(
    select(0.0, sourceJ, sourceGeometryValid),
    select(0.0, candidateJ, candidateGeometryValid),
    select(0.0, 1.0, sourceGeometryValid),
    select(0.0, 1.0, candidateGeometryValid),
  );
  dynamicData[base + 1u] = vec4f(
    select(0.0, sourceEnergy, sourceEnergyValid),
    select(0.0, candidateEnergy, candidateEnergyValid),
    select(0.0, 1.0, sourceEnergyValid),
    select(0.0, 1.0, candidateEnergyValid),
  );
}

fn localStatusIsFailure(status: u32) -> bool {
  return status == LOCAL_STATUS_SHIFT_LIMIT ||
    status == LOCAL_STATUS_NON_DESCENT ||
    status == LOCAL_STATUS_LINE_SEARCH_FAILED ||
    status == LOCAL_STATUS_NONFINITE ||
    status == LOCAL_STATUS_CHOLESKY_FAILED;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reduceCandidate(
  @builtin(local_invocation_index) lane: u32,
) {
  var reduction = CandidateReductionLane(
    vec2f(jgs2_globalization_f32_max),
    vec2f(0.0),
    0.0,
    0u,
    0u,
    0u,
    0u,
    CANDIDATE_REDUCTION_VALIDITY_BITS_MASK,
  );

  // Fixed striding gives every record one deterministic lane owner.
  for (
    var tet = lane;
    tet < params.counts.y;
    tet += WORKGROUP_SIZE
  ) {
    let base = params.offsets3.y + tet * TET_GLOBALIZATION_STRIDE;
    let geometry = dynamicData[base];
    let energy = dynamicData[base + 1u];

    let sourceGeometryValid = geometry.z == 1.0 &&
      jgs2_globalization_finite_scalar(geometry.x);
    let candidateGeometryValid = geometry.w == 1.0 &&
      jgs2_globalization_finite_scalar(geometry.y);
    if (sourceGeometryValid) {
      reduction.minimumDeterminants.x = min(
        reduction.minimumDeterminants.x,
        geometry.x,
      );
    } else {
      reduction.validityBits &= ~SOURCE_GEOMETRY_VALID_BIT;
    }
    if (candidateGeometryValid) {
      reduction.minimumDeterminants.y = min(
        reduction.minimumDeterminants.y,
        geometry.y,
      );
    } else {
      reduction.validityBits &= ~CANDIDATE_GEOMETRY_VALID_BIT;
    }

    let sourceEnergyValid = energy.z == 1.0 &&
      jgs2_globalization_finite_scalar(energy.x);
    let candidateEnergyValid = energy.w == 1.0 &&
      jgs2_globalization_finite_scalar(energy.y);
    if (sourceEnergyValid) {
      let sum = reduction.energies.x + energy.x;
      if (jgs2_globalization_finite_scalar(sum)) {
        reduction.energies.x = sum;
      } else {
        reduction.energies.x = 0.0;
        reduction.validityBits &=
          ~CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT;
      }
    } else {
      reduction.validityBits &=
        ~CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT;
    }
    if (candidateEnergyValid) {
      let sum = reduction.energies.y + energy.y;
      if (jgs2_globalization_finite_scalar(sum)) {
        reduction.energies.y = sum;
      } else {
        reduction.energies.y = 0.0;
        reduction.validityBits &=
          ~CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT;
      }
    } else {
      reduction.validityBits &=
        ~CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT;
    }
  }

  for (
    var vertex = lane;
    vertex < params.counts.x;
    vertex += WORKGROUP_SIZE
  ) {
    let localBase = localGlobalizationBase(vertex);
    let local0 = dynamicData[localBase];
    let local1 = dynamicData[localBase + 1u];
    let local2 = dynamicData[localBase + 2u];
    let local3 = dynamicData[localBase + 3u];
    var localNumericsValid = globalizationFiniteVec4(local0) &&
      globalizationFiniteVec4(local1) &&
      globalizationFiniteVec4(local2) &&
      globalizationFiniteVec4(local3);

    let statusValid = finiteNonnegativeInteger(
      local3.w,
      f32(LOCAL_STATUS_CHOLESKY_FAILED),
    );
    var status = LOCAL_STATUS_NONFINITE;
    if (statusValid) {
      status = u32(local3.w);
    }
    localNumericsValid = localNumericsValid && statusValid &&
      status != LOCAL_STATUS_NONFINITE;
    if (localStatusIsFailure(status)) {
      reduction.localFailureCount += 1u;
    }

    let normalizedShiftValid =
      jgs2_globalization_finite_scalar(local1.w) && local1.w >= 0.0;
    if (normalizedShiftValid) {
      reduction.maximumNormalizedShift = max(
        reduction.maximumNormalizedShift,
        local1.w,
      );
    } else {
      localNumericsValid = false;
    }

    let backtracksValid = finiteNonnegativeInteger(
      local3.y,
      f32(jgs2_globalization_max_backtracks),
    );
    let energyEvaluationsValid = finiteNonnegativeInteger(
      local3.z,
      f32(jgs2_globalization_trial_count),
    );
    var backtracks = 0u;
    var energyEvaluations = 0u;
    if (backtracksValid) {
      backtracks = u32(local3.y);
    } else {
      localNumericsValid = false;
    }
    if (energyEvaluationsValid) {
      energyEvaluations = u32(local3.z);
    } else {
      localNumericsValid = false;
    }
    reduction.maximumBacktracks = max(
      reduction.maximumBacktracks,
      backtracks,
    );
    reduction.totalEnergyEvaluations += energyEvaluations;

    var evaluatedTrials = 0u;
    if (status == LOCAL_STATUS_ACCEPTED) {
      evaluatedTrials = backtracks + 1u;
    } else if (
      status == LOCAL_STATUS_LINE_SEARCH_FAILED &&
      backtracks == jgs2_globalization_max_backtracks
    ) {
      // A preflight source rejection also uses line-search-failed, but has no
      // trials and reports zero backtracks.
      evaluatedTrials = jgs2_globalization_trial_count;
    } else if (
      status == LOCAL_STATUS_ZERO_GRADIENT && backtracks > 0u
    ) {
      // A solver-level zero gradient has zero backtracks and no trial. A
      // storage-resolution no-op reached after halving retains the number of
      // preceding finite geometry trials in backtracks.
      evaluatedTrials = backtracks;
    } else if (status == LOCAL_STATUS_NONFINITE) {
      // The current nonfinite position/directional derivative/geometry is not
      // an ordinary infeasible trial, but prior finite geometry trials still
      // count. A positive stored min-J means geometry completed and only the
      // subsequent energy evaluation was nonfinite, so include that trial;
      // its energy evaluation will cancel from the skipped-geometry count.
      evaluatedTrials = backtracks;
      if (local3.x > jgs2_globalization_determinant_floor) {
        evaluatedTrials += 1u;
      }
    }
    reduction.skippedGeometryTrials += evaluatedTrials - min(
      evaluatedTrials,
      energyEvaluations,
    );
    if (!localNumericsValid) {
      reduction.validityBits &= ~LOCAL_NUMERICS_VALID_BIT;
    }

    let vertexEnergy = dynamicData[params.offsets3.z + vertex];
    let sourceVertexEnergyValid = vertexEnergy.z == 1.0 &&
      jgs2_globalization_finite_scalar(vertexEnergy.x);
    let candidateVertexEnergyValid = vertexEnergy.w == 1.0 &&
      jgs2_globalization_finite_scalar(vertexEnergy.y);
    if (sourceVertexEnergyValid) {
      let sum = reduction.energies.x + vertexEnergy.x;
      if (jgs2_globalization_finite_scalar(sum)) {
        reduction.energies.x = sum;
      } else {
        reduction.energies.x = 0.0;
        reduction.validityBits &=
          ~CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT;
      }
    } else {
      reduction.validityBits &=
        ~CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT;
    }
    if (candidateVertexEnergyValid) {
      let sum = reduction.energies.y + vertexEnergy.y;
      if (jgs2_globalization_finite_scalar(sum)) {
        reduction.energies.y = sum;
      } else {
        reduction.energies.y = 0.0;
        reduction.validityBits &=
          ~CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT;
      }
    } else {
      reduction.validityBits &=
        ~CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT;
    }
  }

  candidateReductionLanes[lane] = reduction;
  workgroupBarrier();
  var stride = WORKGROUP_SIZE / 2u;
  loop {
    if (lane < stride) {
      var left = candidateReductionLanes[lane];
      let right = candidateReductionLanes[lane + stride];
      left.minimumDeterminants = min(
        left.minimumDeterminants,
        right.minimumDeterminants,
      );
      let energySum = left.energies + right.energies;
      if (jgs2_globalization_finite_scalar(energySum.x)) {
        left.energies.x = energySum.x;
      } else {
        left.energies.x = 0.0;
        left.validityBits &=
          ~CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT;
      }
      if (jgs2_globalization_finite_scalar(energySum.y)) {
        left.energies.y = energySum.y;
      } else {
        left.energies.y = 0.0;
        left.validityBits &=
          ~CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT;
      }
      left.maximumNormalizedShift = max(
        left.maximumNormalizedShift,
        right.maximumNormalizedShift,
      );
      left.localFailureCount += right.localFailureCount;
      left.maximumBacktracks = max(
        left.maximumBacktracks,
        right.maximumBacktracks,
      );
      left.skippedGeometryTrials += right.skippedGeometryTrials;
      left.totalEnergyEvaluations += right.totalEnergyEvaluations;
      left.validityBits &= right.validityBits;
      candidateReductionLanes[lane] = left;
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride /= 2u;
  }

  if (lane == 0u) {
    let aggregate = candidateReductionLanes[0];
    var validityBits = aggregate.validityBits;
    // A determinant minimum is undefined for an empty tetrahedron set.
    if (params.counts.y == 0u) {
      validityBits &= ~SOURCE_GEOMETRY_VALID_BIT;
      validityBits &= ~CANDIDATE_GEOMETRY_VALID_BIT;
    }
    let sourceGeometryValid =
      (validityBits & SOURCE_GEOMETRY_VALID_BIT) != 0u;
    let candidateGeometryValid =
      (validityBits & CANDIDATE_GEOMETRY_VALID_BIT) != 0u;
    let sourceEnergyValid =
      (validityBits &
        CANDIDATE_REDUCTION_SOURCE_ENERGY_VALID_BIT) != 0u;
    let candidateEnergyValid =
      (validityBits &
        CANDIDATE_REDUCTION_CANDIDATE_ENERGY_VALID_BIT) != 0u;
    let sourceMinimum = aggregate.minimumDeterminants.x;
    let candidateMinimum = aggregate.minimumDeterminants.y;
    let sourceEnergy = aggregate.energies.x;
    let candidateEnergy = aggregate.energies.y;
    let sourceFeasible = sourceGeometryValid &&
      sourceMinimum > jgs2_globalization_determinant_floor;
    let candidateFeasible = candidateGeometryValid &&
      candidateMinimum > jgs2_globalization_determinant_floor;
    let accepted = sourceFeasible && candidateFeasible;
    let reverted = !accepted;
    let acceptedMinimum = select(sourceMinimum, candidateMinimum, accepted);
    let acceptedMinimumValid = select(
      sourceGeometryValid,
      candidateGeometryValid,
      accepted,
    );
    let acceptedEnergy = select(sourceEnergy, candidateEnergy, accepted);
    let acceptedEnergyValid = select(
      sourceEnergyValid,
      candidateEnergyValid,
      accepted,
    );
    if (acceptedMinimumValid) {
      validityBits |= ACCEPTED_MINIMUM_VALID_BIT;
    }
    let control = params.offsets4.x;
    var historyIndex = 0u;
    let packedHistoryIndex = dynamicData[control + 1u].w;
    if (
      finiteNonnegativeInteger(
        packedHistoryIndex,
        f32(params.offsets4.z),
      )
    ) {
      historyIndex = u32(packedHistoryIndex);
    } else {
      validityBits &= ~LOCAL_NUMERICS_VALID_BIT;
    }
    let controlValidityBits = validityBits & VALIDITY_BITS_MASK;
    dynamicData[control] = vec4f(
      select(0.0, sourceMinimum, sourceGeometryValid),
      select(0.0, candidateMinimum, candidateGeometryValid),
      select(0.0, acceptedMinimum, acceptedMinimumValid),
      f32(controlValidityBits),
    );
    dynamicData[control + 1u] = vec4f(
      select(0.0, 1.0, accepted),
      select(0.0, 1.0, reverted),
      f32(aggregate.localFailureCount),
      f32(historyIndex),
    );
    dynamicData[control + 2u] = vec4f(
      select(0.0, sourceEnergy, sourceEnergyValid),
      select(0.0, candidateEnergy, candidateEnergyValid),
      select(0.0, acceptedEnergy, acceptedEnergyValid),
      select(0.0, 1.0, sourceEnergyValid && candidateEnergyValid),
    );
    dynamicData[control + 3u] = vec4f(
      aggregate.maximumNormalizedShift,
      f32(aggregate.maximumBacktracks),
      f32(aggregate.skippedGeometryTrials),
      f32(aggregate.totalEnergyEvaluations),
    );
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn applyCandidate(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }
  let accepted = dynamicData[params.offsets4.x + 1u].x == 1.0;
  if (!accepted) {
    // Copy the complete vec4 so a rejected assembled pose is byte-identical
    // to its source; never retain a subset of locally accepted vertices.
    dynamicData[params.offsets2.x + vertex] =
      dynamicData[params.offsets1.w + vertex];
    dynamicData[params.offsets2.z + vertex] = vec4f(0.0, 0.0, 0.0, 1.0);
    return;
  }
  let update = loadPosition(params.offsets2.x, vertex) -
    loadPosition(params.offsets1.w, vertex);
  dynamicData[params.offsets2.z + vertex] = vec4f(
    jgs2_globalization_scaled_length3(update),
    0.0,
    0.0,
    1.0,
  );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn convergenceGradient(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }
  let base = params.offsets3.w + vertex * CONVERGENCE_COMPONENT_STRIDE;
  if (vertexData[vertex].info.z != 0u) {
    for (var component = 0u; component < CONVERGENCE_COMPONENT_STRIDE; component += 1u) {
      dynamicData[base + component] = vec4f(0.0);
    }
    return;
  }
  let position = loadPosition(params.offsets2.x, vertex);
  let predicted = loadPosition(params.offsets0.z, vertex);
  let inertia = vertexData[vertex].restMass.w * params.time.z *
    (position - predicted);
  var material = vec3f(0.0);
  let info = vertexData[vertex].info;
  for (var adjacent = 0u; adjacent < info.y; adjacent += 1u) {
    let tet = adjacency[info.x + adjacent];
    let slot = localTetSlot(tetData[tet].indices, vertex);
    if (slot < 4u) {
      material += stableTetrahedronGradientAt(
        tet,
        params.offsets2.x,
      )[slot];
    }
  }
  var contact = vec3f(0.0);
  if (params.solver.x > 0.0 && position.y < params.gravityFloor.w) {
    contact.y = params.solver.x * (position.y - params.gravityFloor.w);
  }
  dynamicData[base] = vec4f(inertia, 0.0);
  dynamicData[base + 1u] = vec4f(material, 0.0);
  dynamicData[base + 2u] = vec4f(0.0);
  dynamicData[base + 3u] = vec4f(0.0);
  dynamicData[base + 4u] = vec4f(contact, 0.0);
}

fn convergenceComponent(vertex: u32, component: u32) -> vec3f {
  return dynamicData[
    params.offsets3.w + vertex * CONVERGENCE_COMPONENT_STRIDE + component
  ].xyz;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reduceConvergence(
  @builtin(local_invocation_index) lane: u32,
) {
  var scaleReduction = ConvergenceReductionLane(
    vec4f(0.0),
    vec2f(0.0),
    0.0,
    1u,
  );

  // Scan one: six robust norm scales (five components plus their sum) and
  // the maximum update. Nonfinite raw storage lanes clear validity before
  // max/min arithmetic can hide them.
  for (
    var vertex = lane;
    vertex < params.counts.x;
    vertex += WORKGROUP_SIZE
  ) {
    var totalGradient = vec3f(0.0);
    var totalGradientValid = true;
    for (
      var component = 0u;
      component < CONVERGENCE_COMPONENT_STRIDE;
      component += 1u
    ) {
      let rawValue = convergenceComponent(vertex, component);
      let valueValid = jgs2_globalization_finite_vec3(rawValue);
      let value = select(vec3f(0.0), rawValue, valueValid);
      let valueScale = max(
        abs(value.x),
        max(abs(value.y), abs(value.z)),
      );
      if (component < 4u) {
        scaleReduction.firstFourChannels[component] = max(
          scaleReduction.firstFourChannels[component],
          valueScale,
        );
      } else {
        scaleReduction.finalChannels.x = max(
          scaleReduction.finalChannels.x,
          valueScale,
        );
      }
      totalGradient += value;
      totalGradientValid = totalGradientValid && valueValid;
    }
    totalGradientValid = totalGradientValid &&
      jgs2_globalization_finite_vec3(totalGradient);
    if (totalGradientValid) {
      let totalScale = max(
        abs(totalGradient.x),
        max(abs(totalGradient.y), abs(totalGradient.z)),
      );
      scaleReduction.finalChannels.y = max(
        scaleReduction.finalChannels.y,
        totalScale,
      );
    } else {
      scaleReduction.valid = 0u;
    }

    let update = dynamicData[params.offsets2.z + vertex].x;
    let updateValid = jgs2_globalization_finite_scalar(update) &&
      update >= 0.0;
    if (updateValid) {
      scaleReduction.maximumUpdate = max(
        scaleReduction.maximumUpdate,
        update,
      );
    } else {
      scaleReduction.valid = 0u;
    }
  }

  convergenceReductionLanes[lane] = scaleReduction;
  workgroupBarrier();
  var scaleStride = WORKGROUP_SIZE / 2u;
  loop {
    if (lane < scaleStride) {
      var left = convergenceReductionLanes[lane];
      let right = convergenceReductionLanes[lane + scaleStride];
      left.firstFourChannels = max(
        left.firstFourChannels,
        right.firstFourChannels,
      );
      left.finalChannels = max(left.finalChannels, right.finalChannels);
      left.maximumUpdate = max(left.maximumUpdate, right.maximumUpdate);
      left.valid &= right.valid;
      convergenceReductionLanes[lane] = left;
    }
    workgroupBarrier();
    if (scaleStride == 1u) {
      break;
    }
    scaleStride /= 2u;
  }

  let firstFourScales = convergenceReductionLanes[0].firstFourChannels;
  let finalScales = convergenceReductionLanes[0].finalChannels;
  let maximumUpdate = convergenceReductionLanes[0].maximumUpdate;
  let rawLanesValid = convergenceReductionLanes[0].valid == 1u;
  // Every lane must copy the broadcast values before lane zero reuses its
  // shared slot for normalized sums.
  workgroupBarrier();

  var sumReduction = ConvergenceReductionLane(
    vec4f(0.0),
    vec2f(0.0),
    0.0,
    1u,
  );
  // Scan two: accumulate normalized squares with the six global scales.
  for (
    var vertex = lane;
    vertex < params.counts.x;
    vertex += WORKGROUP_SIZE
  ) {
    var totalGradient = vec3f(0.0);
    var totalGradientValid = true;
    for (
      var component = 0u;
      component < CONVERGENCE_COMPONENT_STRIDE;
      component += 1u
    ) {
      let rawValue = convergenceComponent(vertex, component);
      let valueValid = jgs2_globalization_finite_vec3(rawValue);
      let value = select(vec3f(0.0), rawValue, valueValid);
      var scale = finalScales.x;
      if (component < 4u) {
        scale = firstFourScales[component];
      }
      var square = 0.0;
      if (valueValid && scale > 0.0) {
        let normalized = value / scale;
        square = dot(normalized, normalized);
      }
      let squareValid = jgs2_globalization_finite_scalar(square) &&
        square >= 0.0;
      if (component < 4u) {
        let sum = sumReduction.firstFourChannels[component] + square;
        if (squareValid && jgs2_globalization_finite_scalar(sum)) {
          sumReduction.firstFourChannels[component] = sum;
        } else {
          sumReduction.firstFourChannels[component] = 0.0;
          sumReduction.valid = 0u;
        }
      } else {
        let sum = sumReduction.finalChannels.x + square;
        if (squareValid && jgs2_globalization_finite_scalar(sum)) {
          sumReduction.finalChannels.x = sum;
        } else {
          sumReduction.finalChannels.x = 0.0;
          sumReduction.valid = 0u;
        }
      }
      totalGradient += value;
      totalGradientValid = totalGradientValid && valueValid;
    }
    totalGradientValid = totalGradientValid &&
      jgs2_globalization_finite_vec3(totalGradient);
    var totalSquare = 0.0;
    if (totalGradientValid && finalScales.y > 0.0) {
      let normalizedTotal = totalGradient / finalScales.y;
      totalSquare = dot(normalizedTotal, normalizedTotal);
    }
    let totalSquareSum = sumReduction.finalChannels.y + totalSquare;
    if (
      totalGradientValid &&
      jgs2_globalization_finite_scalar(totalSquare) &&
      totalSquare >= 0.0 &&
      jgs2_globalization_finite_scalar(totalSquareSum)
    ) {
      sumReduction.finalChannels.y = totalSquareSum;
    } else {
      sumReduction.finalChannels.y = 0.0;
      sumReduction.valid = 0u;
    }
  }

  convergenceReductionLanes[lane] = sumReduction;
  workgroupBarrier();
  var sumStride = WORKGROUP_SIZE / 2u;
  loop {
    if (lane < sumStride) {
      var left = convergenceReductionLanes[lane];
      let right = convergenceReductionLanes[lane + sumStride];
      let firstFourSums = left.firstFourChannels +
        right.firstFourChannels;
      let finalSums = left.finalChannels + right.finalChannels;
      for (var component = 0u; component < 4u; component += 1u) {
        if (jgs2_globalization_finite_scalar(firstFourSums[component])) {
          left.firstFourChannels[component] = firstFourSums[component];
        } else {
          left.firstFourChannels[component] = 0.0;
          left.valid = 0u;
        }
      }
      for (var component = 0u; component < 2u; component += 1u) {
        if (jgs2_globalization_finite_scalar(finalSums[component])) {
          left.finalChannels[component] = finalSums[component];
        } else {
          left.finalChannels[component] = 0.0;
          left.valid = 0u;
        }
      }
      left.valid &= right.valid;
      convergenceReductionLanes[lane] = left;
    }
    workgroupBarrier();
    if (sumStride == 1u) {
      break;
    }
    sumStride /= 2u;
  }

  if (lane == 0u) {
    let normalizedSums = convergenceReductionLanes[0];
    var componentNorms: array<f32, 5>;
    componentNorms[0] = firstFourScales.x *
      sqrt(normalizedSums.firstFourChannels.x);
    componentNorms[1] = firstFourScales.y *
      sqrt(normalizedSums.firstFourChannels.y);
    componentNorms[2] = firstFourScales.z *
      sqrt(normalizedSums.firstFourChannels.z);
    componentNorms[3] = firstFourScales.w *
      sqrt(normalizedSums.firstFourChannels.w);
    componentNorms[4] = finalScales.x *
      sqrt(normalizedSums.finalChannels.x);
    let gradientNorm = finalScales.y *
      sqrt(normalizedSums.finalChannels.y);
    var componentNumericsValid = rawLanesValid &&
      normalizedSums.valid == 1u;
    for (
      var component = 0u;
      component < CONVERGENCE_COMPONENT_STRIDE;
      component += 1u
    ) {
      componentNumericsValid = componentNumericsValid &&
        jgs2_globalization_finite_scalar(componentNorms[component]) &&
        componentNorms[component] >= 0.0;
    }
    let residualDenominator = max(
      1.0,
      componentNorms[0] + componentNorms[1] + componentNorms[2] +
        componentNorms[3] + componentNorms[4],
    );
    let relativeResidual = gradientNorm / residualDenominator;
    let residualNumericsValid = componentNumericsValid &&
      jgs2_globalization_finite_scalar(gradientNorm) &&
      gradientNorm >= 0.0 &&
      jgs2_globalization_finite_scalar(residualDenominator) &&
      residualDenominator >= 1.0 &&
      jgs2_globalization_finite_scalar(relativeResidual) &&
      relativeResidual >= 0.0;
    let sceneScaleValid =
      jgs2_globalization_finite_scalar(params.convergence.z) &&
      params.convergence.z >= 0.0;
    let normalizedMaximumUpdate = maximumUpdate /
      max(select(1.0e-12, params.convergence.z, sceneScaleValid), 1.0e-12);
    let updateNumericsValid = rawLanesValid && sceneScaleValid &&
      jgs2_globalization_finite_scalar(maximumUpdate) &&
      maximumUpdate >= 0.0 &&
      jgs2_globalization_finite_scalar(normalizedMaximumUpdate) &&
      normalizedMaximumUpdate >= 0.0;

    let control = params.offsets4.x;
    let geometry = dynamicData[control];
    let controlValidityBits = globalizationValidityBits(geometry.w);
    let sourceGeometryValid =
      (controlValidityBits & SOURCE_GEOMETRY_VALID_BIT) != 0u &&
      jgs2_globalization_finite_scalar(geometry.x);
    let candidateGeometryValid =
      (controlValidityBits & CANDIDATE_GEOMETRY_VALID_BIT) != 0u &&
      jgs2_globalization_finite_scalar(geometry.y);
    let acceptedMinimumValid =
      (controlValidityBits & ACCEPTED_MINIMUM_VALID_BIT) != 0u &&
      jgs2_globalization_finite_scalar(geometry.z);
    let localNumericsValid =
      (controlValidityBits & LOCAL_NUMERICS_VALID_BIT) != 0u;

    let acceptance = dynamicData[control + 1u];
    let assembledAcceptedValid = finiteNonnegativeInteger(
      acceptance.x,
      1.0,
    );
    let assembledRevertedValid = finiteNonnegativeInteger(
      acceptance.y,
      1.0,
    );
    let localFailureCountValid = finiteNonnegativeInteger(
      acceptance.z,
      f32(params.counts.x),
    );
    let historyIndexValid = finiteNonnegativeInteger(
      acceptance.w,
      f32(params.offsets4.z),
    );
    let assembledAccepted = assembledAcceptedValid && acceptance.x == 1.0;
    let assembledReverted = assembledRevertedValid && acceptance.y == 1.0;
    var localFailureCount = 0u;
    var historyIndex = 0u;
    if (localFailureCountValid) {
      localFailureCount = u32(acceptance.z);
    }
    if (historyIndexValid) {
      historyIndex = u32(acceptance.w);
    }

    let controlMetrics = dynamicData[control + 3u];
    let controlMetricsValid =
      jgs2_globalization_finite_scalar(controlMetrics.x) &&
      controlMetrics.x >= 0.0 &&
      finiteNonnegativeInteger(
        controlMetrics.y,
        f32(jgs2_globalization_max_backtracks),
      ) &&
      finiteNonnegativeInteger(controlMetrics.z, f32(params.counts.x) *
        f32(jgs2_globalization_trial_count)) &&
      finiteNonnegativeInteger(controlMetrics.w, f32(params.counts.x) *
        f32(jgs2_globalization_trial_count));
    var finite = residualNumericsValid && updateNumericsValid &&
      sourceGeometryValid && candidateGeometryValid &&
      acceptedMinimumValid && localNumericsValid &&
      assembledAcceptedValid && assembledRevertedValid &&
      localFailureCountValid && historyIndexValid && controlMetricsValid;
    let residualSatisfied = finite &&
      relativeResidual <= params.convergence.x;
    let updateSatisfied = finite &&
      normalizedMaximumUpdate <= params.convergence.y;
    let converged = residualSatisfied && updateSatisfied &&
      assembledAccepted && !assembledReverted && localFailureCount == 0u;

    if (historyIndexValid && historyIndex < params.offsets4.z) {
      let history = params.offsets4.y +
        historyIndex * GLOBALIZATION_HISTORY_STRIDE;
      dynamicData[history] = dynamicData[control];
      dynamicData[history + 1u] = vec4f(
        acceptance.xyz,
        select(0.0, 1.0, assembledReverted),
      );
      dynamicData[history + 2u] = dynamicData[control + 2u];
      dynamicData[history + 3u] = vec4f(
        select(0.0, componentNorms[0], componentNumericsValid),
        select(0.0, componentNorms[1], componentNumericsValid),
        select(0.0, componentNorms[2], componentNumericsValid),
        select(0.0, componentNorms[3], componentNumericsValid),
      );
      dynamicData[history + 4u] = vec4f(
        select(0.0, componentNorms[4], componentNumericsValid),
        select(0.0, gradientNorm, residualNumericsValid),
        select(0.0, residualDenominator, residualNumericsValid),
        select(0.0, relativeResidual, residualNumericsValid),
      );
      dynamicData[history + 5u] = vec4f(
        select(0.0, maximumUpdate, updateNumericsValid),
        select(0.0, normalizedMaximumUpdate, updateNumericsValid),
        select(0.0, 1.0, residualSatisfied),
        select(0.0, 1.0, updateSatisfied),
      );
      dynamicData[history + 6u] = vec4f(
        select(0.0, 1.0, converged),
        select(0.0, 1.0, finite),
        f32(historyIndex),
        select(0.0, controlMetrics.x, controlMetricsValid),
      );
      dynamicData[history + 7u] = vec4f(
        select(vec3f(0.0), controlMetrics.yzw, controlMetricsValid),
        1.0,
      );
    }
    dynamicData[control + 1u].w = f32(historyIndex + 1u);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn copyPosition(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }
  dynamicData[params.offsets2.x + vertex] =
    vec4f(loadPosition(params.offsets1.w, vertex), 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn bodyHorizontalCorrection(@builtin(global_invocation_id) globalId: vec3u) {
  let body = globalId.x;
  if (body >= params.counts.w) {
    return;
  }

  var totalMass = 0.0;
  var solvedMoment = vec2f(0.0);
  var predictedMoment = vec2f(0.0);
  var hasPinnedVertex = false;
  for (var vertex = 0u; vertex < params.counts.x; vertex += 1u) {
    let item = vertexData[vertex];
    if (item.info.w != body) {
      continue;
    }
    hasPinnedVertex = hasPinnedVertex || item.info.z != 0u;
    let mass = max(item.restMass.w, 0.0);
    let solved = loadPosition(params.offsets0.x, vertex);
    let predicted = loadPosition(params.offsets0.z, vertex);
    totalMass += mass;
    solvedMoment += mass * solved.xz;
    predictedMoment += mass * predicted.xz;
  }

  var correction = vec2f(0.0);
  // Anchored bodies can exchange horizontal momentum with their constraints;
  // correcting them would fight the fixed vertices. Free bodies, however,
  // have no external horizontal force on the level frictionless penalty plane.
  if (!hasPinnedVertex && totalMass > 0.0) {
    correction = (predictedMoment - solvedMoment) / totalMass;
  }
  dynamicData[params.offsets2.y + body] =
    vec4f(correction.x, 0.0, correction.y, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn applyBodyHorizontalCorrection(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x || vertexData[vertex].info.z != 0u) {
    return;
  }
  let body = vertexData[vertex].info.w;
  if (body >= params.counts.w) {
    return;
  }
  let correction = dynamicData[params.offsets2.y + body].xz;
  var position = dynamicData[params.offsets0.x + vertex];
  position.x += correction.x;
  position.z += correction.y;
  dynamicData[params.offsets0.x + vertex] = position;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn finalize(@builtin(global_invocation_id) globalId: vec3u) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }

  if (vertexData[vertex].info.z != 0u) {
    // The stable path has already proposed the pinned rest target in
    // jgs2Solve and passed that complete pose through the assembled
    // determinant gate. Preserve either its accepted candidate or its
    // byte-identical reverted source here. Reapplying the target after that
    // gate would bypass feasibility. The legacy co-rotated path has no
    // assembled gate and retains its historical hard snap.
    if (params.offsets4.w == 0u) {
      dynamicData[params.offsets0.x + vertex] =
        vec4f(vertexData[vertex].restMass.xyz, 1.0);
    }
    dynamicData[params.offsets0.w + vertex] = vec4f(0.0);
    return;
  }
  let position = loadPosition(params.offsets0.x, vertex);
  let oldPosition = loadPosition(params.offsets1.x, vertex);
  var velocity = (position - oldPosition) * params.time.y;
  // This is a simple grounded viscous-friction model, not Coulomb friction:
  // it removes only tangential velocity and leaves the normal penalty intact.
  if (
    params.solver.x > 0.0 &&
    params.contact.x > 0.0 &&
    position.y <= params.gravityFloor.w + params.contact.y
  ) {
    let tangentialScale = exp(-params.contact.x * params.time.x);
    velocity.x *= tangentialScale;
    velocity.z *= tangentialScale;
  }
  dynamicData[params.offsets0.w + vertex] = vec4f(velocity, 0.0);
}
`;
