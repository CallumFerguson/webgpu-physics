import { JGS2_CUBATURE_RECORD_WORDS } from "./layout";

export const JGS2_WORKGROUP_SIZE = 128;

export const jgs2Shader = /* wgsl */ `
const WORKGROUP_SIZE: u32 = ${JGS2_WORKGROUP_SIZE}u;
const CUBATURE_RECORD_WORDS: u32 = ${JGS2_CUBATURE_RECORD_WORDS}u;
const EMPTY_TET: u32 = 0xffffffffu;

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
  // iteration target position, per-body horizontal correction, reserved
  offsets2: vec4u,
  // dt, inverse dt, inverse dt squared, maximum local step
  time: vec4f,
  // gravity xyz, floor height
  gravityFloor: vec4f,
  // floor stiffness, relative regularization, polar epsilon, velocity damping
  solver: vec4f,
  // grounded tangential damping rate (s^-1), contact margin, reserved
  contact: vec4f,
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
  for (var iteration = 0u; iteration < 5u; iteration += 1u) {
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

fn tetGradient(tet: u32) -> array<vec3f, 4> {
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

fn cubatureHessian(
  tet: u32,
  basis: array<mat3x3f, 4>,
) -> mat3x3f {
  let rotation = loadRotation(params.offsets1.z, tet);
  let inverseRotation = transpose(rotation);
  var reducedColumns: array<vec3f, 3>;

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
  // Starting in posB means every odd solve count lands back in posA.
  dynamicData[params.offsets0.y + vertex] = vec4f(predicted, 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn tetPolarRotation(@builtin(global_invocation_id) globalId: vec3u) {
  let tet = globalId.x;
  if (tet >= params.counts.y) {
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
  for (var adjacent = 0u; adjacent < info.y; adjacent += 1u) {
    let tet = adjacency[info.x + adjacent];
    let volumeWeight = max(abs(tetData[tet].attributes.x), params.solver.z);
    average += volumeWeight * loadRotation(params.offsets1.z, tet);
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
    return;
  }

  let position = loadPosition(params.offsets1.w, vertex);
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
      hessian += rotation * stiffnessBlock(tet, slot, slot) * transpose(rotation);
    }
  }

  // A one-sided quadratic penalty keeps the first demos deterministic and
  // avoids requiring a CPU collision pipeline.
  if (params.solver.x > 0.0 && position.y < params.gravityFloor.w) {
    gradient.y += params.solver.x * (position.y - params.gravityFloor.w);
    hessian[1][1] += params.solver.x;
  }

  // Cubature supplies the complementary gradient and Hessian in Eq. 15-17.
  let localRotation = loadRotation(params.offsets1.y, vertex);
  for (var sample = 0u; sample < params.counts.z; sample += 1u) {
    let record = vertex * params.counts.z + sample;
    let recordBase = record * CUBATURE_RECORD_WORDS;
    let tet = cubatureWords[recordBase];
    let weight = cubatureFloat(recordBase + 1u);
    if (tet == EMPTY_TET || tet >= params.counts.y || !(weight > 0.0)) {
      continue;
    }
    let indices = tetData[tet].indices;

    var basis: array<mat3x3f, 4>;
    basis[0] = loadRotation(params.offsets1.y, indices.x) *
      cubatureBlock(recordBase, 0u) * transpose(localRotation);
    basis[1] = loadRotation(params.offsets1.y, indices.y) *
      cubatureBlock(recordBase, 1u) * transpose(localRotation);
    basis[2] = loadRotation(params.offsets1.y, indices.z) *
      cubatureBlock(recordBase, 2u) * transpose(localRotation);
    basis[3] = loadRotation(params.offsets1.y, indices.w) *
      cubatureBlock(recordBase, 3u) * transpose(localRotation);

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
      projectedHessian -=
        elementRotation * stiffnessBlock(tet, sourceSlot, sourceSlot) *
          transpose(elementRotation) +
        distributedInertiaWeight(vertex) * identityMat3();
    }

    gradient += weight * projectedGradient;
    hessian += weight * projectedHessian;
  }

  var delta = regularizedSolve(hessian, -gradient);
  let deltaLengthSquared = dot(delta, delta);
  let maxStep = params.time.w;
  if (maxStep > 0.0 && deltaLengthSquared > maxStep * maxStep) {
    delta *= maxStep * inverseSqrt(deltaLengthSquared);
  }
  dynamicData[params.offsets2.x + vertex] = vec4f(position + delta, 1.0);
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
    dynamicData[params.offsets0.x + vertex] =
      vec4f(vertexData[vertex].restMass.xyz, 1.0);
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
