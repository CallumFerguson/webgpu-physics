import { stableNeoHookeanWgsl } from "./stable-neo-hookean-wgsl";

export const JGS2_DIAGNOSTICS_WORKGROUP_SIZE = 128;
export const JGS2_DIAGNOSTIC_TET_ROTATION_VEC4S = 3;
export const JGS2_DIAGNOSTIC_TET_METRIC_VEC4S = 1;
export const JGS2_DIAGNOSTIC_VERTEX_RECORD_VEC4S = 7;
export const JGS2_DIAGNOSTIC_SUMMARY_VEC4S = 5;
export const JGS2_DIAGNOSTICS_UNIFORM_BYTES = 4 * 16;

export interface JGS2DiagnosticsLayout {
  /** Offsets are expressed in vec4 elements, not bytes. */
  readonly tetRotation: number;
  readonly tetMetrics: number;
  readonly vertexRecords: number;
  readonly summary: number;
  readonly vec4Count: number;
  readonly byteLength: number;
}

export interface JGS2GpuOracleVertexRecord {
  readonly vertex: number;
  readonly active: boolean;
  /** Exact active-coordinate gradient for this vertex. */
  readonly gradient: readonly [number, number, number];
  /** Row-major diagonal 3x3 block of the exact frozen global Hessian. */
  readonly localHessian: Float32Array;
  readonly inertiaEnergy: number;
  readonly floorContactEnergy: number;
  /** Euclidean norm of this vertex's final nonlinear solve delta. */
  readonly updateMagnitude: number;
  readonly updateMagnitudeValid: boolean;
  readonly gradientSquaredNorm: number;
  readonly inertiaWeight: number;
  readonly finite: boolean;
}

export interface JGS2GpuOracleTetrahedronRecord {
  readonly tetrahedron: number;
  readonly elasticityEnergy: number;
  readonly deformationDeterminant: number;
  readonly finite: boolean;
}

export interface JGS2GpuOracleDiagnostics {
  /** Sum of all three exact implicit-Euler energy components. */
  readonly energy: number;
  readonly components: {
    readonly inertia: number;
    readonly elasticity: number;
    readonly floorContact: number;
  };
  /** Euclidean norm of the complete active-coordinate gradient. */
  readonly gradientNorm: number;
  /** Numerator used by residual normalizations; equal to gradientNorm. */
  readonly residualNumerator: number;
  /**
   * Force-balance residual: ||g|| / max(1, ||g_i|| + ||g_e|| + ||g_f||).
   */
  readonly relativeResidual: number;
  readonly relativeResidualValid: boolean;
  readonly residualDenominator: number;
  readonly gradientSquaredNorm: number;
  readonly gradientValid: boolean;
  /** Largest active-vertex ||delta|| from the final nonlinear iteration. */
  readonly maximumUpdate: number;
  readonly maximumUpdateValid: boolean;
  readonly minimumDeformationDeterminant: number;
  readonly minimumDeformationDeterminantValid: boolean;
  /** Minimum signed y distance to the analytic floor over active vertices. */
  readonly minimumContactDistance: number;
  readonly minimumContactDistanceValid: boolean;
  /** Active analytic-floor penalty count (signed distance < 0). */
  readonly activeContactCount: number;
  readonly activeContactCountValid: boolean;
  /** False sentinel: no bounded general candidate buffer exists yet. */
  readonly candidateBufferOverflow: false;
  readonly candidateBufferOverflowValid: false;
  readonly activeVertexCount: number;
  /** False if any source value, element result, record, or reduction was non-finite. */
  readonly finite: boolean;
  readonly vertices: readonly JGS2GpuOracleVertexRecord[];
  readonly tetrahedra: readonly JGS2GpuOracleTetrahedronRecord[];
}

export interface JGS2GpuDiagnosticsSourceBuffers {
  readonly dynamic: GPUBuffer;
  readonly vertices: GPUBuffer;
  readonly tets: GPUBuffer;
  readonly stiffness: GPUBuffer;
  readonly adjacency: GPUBuffer;
}

export interface JGS2GpuDiagnosticsEvaluationSettings {
  /** Dynamic-buffer offsets are expressed in vec4 elements. */
  readonly currentPositionOffset: number;
  readonly predictedPositionOffset: number;
  readonly finalUpdateOffset: number;
  readonly timestep: number;
  readonly floorHeight: number;
  readonly floorStiffness: number;
  readonly rotationEpsilon: number;
}

function assertCount(name: string, value: number, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be an integer >= ${minimum}; got ${value}.`);
  }
}

export function computeJGS2DiagnosticsLayout(
  vertexCount: number,
  tetCount: number,
): JGS2DiagnosticsLayout {
  assertCount("vertexCount", vertexCount, 1);
  // Supporting zero tets makes the finite sentinel/validity contract explicit,
  // although the current tetrahedral solver itself requires at least one.
  assertCount("tetCount", tetCount, 0);

  const tetRotation = 0;
  const tetMetrics =
    tetRotation + tetCount * JGS2_DIAGNOSTIC_TET_ROTATION_VEC4S;
  const vertexRecords =
    tetMetrics + tetCount * JGS2_DIAGNOSTIC_TET_METRIC_VEC4S;
  const summary =
    vertexRecords + vertexCount * JGS2_DIAGNOSTIC_VERTEX_RECORD_VEC4S;
  const vec4Count = summary + JGS2_DIAGNOSTIC_SUMMARY_VEC4S;
  return {
    tetRotation,
    tetMetrics,
    vertexRecords,
    summary,
    vec4Count,
    byteLength: vec4Count * 16,
  };
}

function finiteFlag(value: number): boolean {
  return Number.isFinite(value) && value >= 0.5;
}

export function decodeJGS2GpuOracleDiagnostics(
  packed: Float32Array,
  vertexCount: number,
  tetCount: number,
  layout = computeJGS2DiagnosticsLayout(vertexCount, tetCount),
): JGS2GpuOracleDiagnostics {
  if (packed.length !== layout.vec4Count * 4) {
    throw new RangeError(
      `GPU diagnostics contain ${packed.length} floats; expected ${layout.vec4Count * 4}.`,
    );
  }

  const tetrahedra: JGS2GpuOracleTetrahedronRecord[] = [];
  for (let tetrahedron = 0; tetrahedron < tetCount; tetrahedron += 1) {
    const base = (layout.tetMetrics + tetrahedron) * 4;
    tetrahedra.push({
      tetrahedron,
      elasticityEnergy: packed[base]!,
      deformationDeterminant: packed[base + 1]!,
      finite: finiteFlag(packed[base + 2]!),
    });
  }

  const vertices: JGS2GpuOracleVertexRecord[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const base =
      (layout.vertexRecords + vertex * JGS2_DIAGNOSTIC_VERTEX_RECORD_VEC4S) * 4;
    vertices.push({
      vertex,
      active: finiteFlag(packed[base + 3]!),
      gradient: [packed[base]!, packed[base + 1]!, packed[base + 2]!],
      localHessian: new Float32Array([
        packed[base + 4]!,
        packed[base + 5]!,
        packed[base + 6]!,
        packed[base + 8]!,
        packed[base + 9]!,
        packed[base + 10]!,
        packed[base + 12]!,
        packed[base + 13]!,
        packed[base + 14]!,
      ]),
      inertiaEnergy: packed[base + 7]!,
      floorContactEnergy: packed[base + 11]!,
      updateMagnitude: packed[base + 15]!,
      gradientSquaredNorm: packed[base + 16]!,
      finite: finiteFlag(packed[base + 17]!),
      inertiaWeight: packed[base + 18]!,
      updateMagnitudeValid: finiteFlag(packed[base + 19]!),
    });
  }

  const summary = layout.summary * 4;
  const inertia = packed[summary]!;
  const elasticity = packed[summary + 1]!;
  const floorContact = packed[summary + 2]!;
  const gradientNorm = packed[summary + 4]!;
  const activeVertexCount = Math.max(0, Math.round(packed[summary + 9]!));
  const activeValid = finiteFlag(packed[summary + 10]!);
  return {
    energy: inertia + elasticity + floorContact,
    components: { inertia, elasticity, floorContact },
    gradientNorm,
    residualNumerator: packed[summary + 16]!,
    residualDenominator: packed[summary + 17]!,
    relativeResidual: packed[summary + 18]!,
    relativeResidualValid: finiteFlag(packed[summary + 19]!),
    maximumUpdate: packed[summary + 5]!,
    minimumDeformationDeterminant: packed[summary + 6]!,
    finite: finiteFlag(packed[summary + 7]!),
    gradientSquaredNorm: packed[summary + 8]!,
    activeVertexCount,
    gradientValid: activeValid,
    maximumUpdateValid: finiteFlag(packed[summary + 15]!),
    minimumDeformationDeterminantValid: finiteFlag(packed[summary + 11]!),
    minimumContactDistance: packed[summary + 3]!,
    minimumContactDistanceValid: finiteFlag(packed[summary + 13]!),
    activeContactCount: Math.max(0, Math.round(packed[summary + 12]!)),
    activeContactCountValid: finiteFlag(packed[summary + 14]!),
    candidateBufferOverflow: false,
    candidateBufferOverflowValid: false,
    vertices,
    tetrahedra,
  };
}

/** Roadmap relative-error definition for CPU/GPU diagnostic comparisons. */
export function jgs2DiagnosticRelativeError(
  actual: ArrayLike<number>,
  reference: ArrayLike<number>,
): number {
  if (actual.length !== reference.length) {
    throw new RangeError("Diagnostic vectors must have matching lengths.");
  }
  let differenceSquared = 0;
  let actualSquared = 0;
  let referenceSquared = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const actualValue = actual[index]!;
    const referenceValue = reference[index]!;
    differenceSquared += (actualValue - referenceValue) ** 2;
    actualSquared += actualValue ** 2;
    referenceSquared += referenceValue ** 2;
  }
  return (
    Math.sqrt(differenceSquared) /
    Math.max(1, Math.sqrt(actualSquared), Math.sqrt(referenceSquared))
  );
}

export function packJGS2DiagnosticsUniforms(
  vertexCount: number,
  tetCount: number,
  layout: JGS2DiagnosticsLayout,
  settings: JGS2GpuDiagnosticsEvaluationSettings,
): Uint8Array {
  assertCount("currentPositionOffset", settings.currentPositionOffset, 0);
  assertCount("predictedPositionOffset", settings.predictedPositionOffset, 0);
  assertCount("finalUpdateOffset", settings.finalUpdateOffset, 0);
  const finiteValues: ReadonlyArray<readonly [string, number]> = [
    ["timestep", settings.timestep],
    ["floorHeight", settings.floorHeight],
    ["floorStiffness", settings.floorStiffness],
    ["rotationEpsilon", settings.rotationEpsilon],
  ];
  for (const [name, value] of finiteValues) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${name} must be finite.`);
    }
  }
  if (!(settings.timestep > 0)) {
    throw new RangeError("timestep must be positive.");
  }
  if (settings.floorStiffness < 0) {
    throw new RangeError("floorStiffness must be nonnegative.");
  }
  if (!(settings.rotationEpsilon > 0)) {
    throw new RangeError("rotationEpsilon must be positive.");
  }

  const buffer = new ArrayBuffer(JGS2_DIAGNOSTICS_UNIFORM_BYTES);
  const integers = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  integers.set([vertexCount, tetCount, 0, 0], 0);
  integers.set(
    [
      settings.currentPositionOffset,
      settings.predictedPositionOffset,
      layout.tetRotation,
      layout.tetMetrics,
    ],
    4,
  );
  integers.set(
    [layout.vertexRecords, layout.summary, settings.finalUpdateOffset, 0],
    8,
  );
  floats.set(
    [
      1 / (settings.timestep * settings.timestep),
      settings.floorHeight,
      settings.floorStiffness,
      settings.rotationEpsilon,
    ],
    12,
  );
  return new Uint8Array(buffer);
}

export const jgs2DiagnosticsShader = /* wgsl */ `
const WORKGROUP_SIZE: u32 = ${JGS2_DIAGNOSTICS_WORKGROUP_SIZE}u;
const VERTEX_RECORD_VEC4S: u32 = ${JGS2_DIAGNOSTIC_VERTEX_RECORD_VEC4S}u;

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

struct DiagnosticParams {
  // vertex count, tetrahedron count, reserved
  counts: vec4u,
  // current position, predicted position, tet rotation, tet metric offsets
  offsets: vec4u,
  // vertex-record offset, summary offset, final-update dynamic offset, reserved
  records: vec4u,
  // inverse dt squared, floor height, floor stiffness, polar epsilon
  physics: vec4f,
}

@group(0) @binding(0)
var<storage, read> dynamicData: array<vec4f>;
@group(0) @binding(1)
var<storage, read> vertexData: array<VertexStatic>;
@group(0) @binding(2)
var<storage, read> tetData: array<TetStatic>;
@group(0) @binding(3)
var<storage, read> restStiffness: array<f32>;
@group(0) @binding(4)
var<storage, read> adjacency: array<u32>;
@group(0) @binding(5)
var<storage, read_write> diagnosticData: array<vec4f>;
@group(0) @binding(6)
var<uniform> params: DiagnosticParams;

${stableNeoHookeanWgsl}

fn finiteScalar(value: f32) -> bool {
  return value == value && abs(value) <= 3.0e38;
}

fn finiteVec3(value: vec3f) -> bool {
  return finiteScalar(value.x) && finiteScalar(value.y) && finiteScalar(value.z);
}

fn finiteMat3(value: mat3x3f) -> bool {
  return finiteVec3(value[0]) && finiteVec3(value[1]) && finiteVec3(value[2]);
}

fn finiteNumber(value: f32) -> f32 {
  return select(0.0, value, finiteScalar(value));
}

fn finiteVector(value: vec3f) -> vec3f {
  return vec3f(
    finiteNumber(value.x),
    finiteNumber(value.y),
    finiteNumber(value.z),
  );
}

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

fn safeNormalize(value: vec3f, fallback: vec3f) -> vec3f {
  let squared = dot(value, value);
  if (squared > params.physics.w * params.physics.w && finiteScalar(squared)) {
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
  return (1.0 / determinantValue) * mat3x3f(
    cross(matrix[1], matrix[2]),
    cross(matrix[2], matrix[0]),
    cross(matrix[0], matrix[1]),
  );
}

// This intentionally matches the runtime solver's seven f32 Newton-polar steps.
fn polarRotation(matrix: mat3x3f) -> mat3x3f {
  var rotation = matrix;
  for (var iteration = 0u; iteration < 7u; iteration += 1u) {
    let determinantValue = matrixDeterminant(rotation);
    let magnitude = abs(determinantValue);
    if (!(magnitude > params.physics.w) || !finiteScalar(magnitude)) {
      break;
    }
    rotation = 0.5 * (rotation + inverseTranspose(rotation, determinantValue));
  }
  return orthonormalize(rotation);
}

fn loadPosition(offset: u32, vertex: u32) -> vec3f {
  return dynamicData[offset + vertex].xyz;
}

fn loadDiagnosticRotation(tetrahedron: u32) -> mat3x3f {
  let base = params.offsets.z + tetrahedron * 3u;
  return mat3x3f(
    diagnosticData[base].xyz,
    diagnosticData[base + 1u].xyz,
    diagnosticData[base + 2u].xyz,
  );
}

fn stiffnessBlock(
  tetrahedron: u32,
  rowVertex: u32,
  columnVertex: u32,
) -> mat3x3f {
  let base = tetrahedron * 144u;
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

fn usesStableNeoHookean(tetrahedron: u32) -> bool {
  return tetData[tetrahedron].attributes.w >= 0.5;
}

fn tetrahedronInverseDm(tetrahedron: u32) -> mat3x3f {
  let item = tetData[tetrahedron];
  return mat3x3f(item.invDm0.xyz, item.invDm1.xyz, item.invDm2.xyz);
}

fn tetrahedronDeformationGradient(tetrahedron: u32) -> mat3x3f {
  let item = tetData[tetrahedron];
  let position0 = loadPosition(params.offsets.x, item.indices.x);
  let deformedShape = mat3x3f(
    loadPosition(params.offsets.x, item.indices.y) - position0,
    loadPosition(params.offsets.x, item.indices.z) - position0,
    loadPosition(params.offsets.x, item.indices.w) - position0,
  );
  return deformedShape * tetrahedronInverseDm(tetrahedron);
}

fn tetrahedronShapeGradients(tetrahedron: u32) -> array<vec3f, 4> {
  let inverseDmRows = transpose(tetrahedronInverseDm(tetrahedron));
  var gradients: array<vec3f, 4>;
  gradients[1] = inverseDmRows[0];
  gradients[2] = inverseDmRows[1];
  gradients[3] = inverseDmRows[2];
  gradients[0] = -gradients[1] - gradients[2] - gradients[3];
  return gradients;
}

fn stableTetrahedronGradient(tetrahedron: u32) -> array<vec3f, 4> {
  let item = tetData[tetrahedron];
  let deformationGradient = tetrahedronDeformationGradient(tetrahedron);
  let firstPiola = snh_first_piola(
    deformationGradient,
    item.attributes.y,
    item.attributes.z,
  );
  let gradients = tetrahedronShapeGradients(tetrahedron);
  var result: array<vec3f, 4>;
  for (var local = 0u; local < 4u; local += 1u) {
    result[local] = item.attributes.x * firstPiola * gradients[local];
  }
  return result;
}

fn stableTetrahedronLocalHessian(
  tetrahedron: u32,
  localVertex: u32,
) -> mat3x3f {
  let item = tetData[tetrahedron];
  let deformationGradient = tetrahedronDeformationGradient(tetrahedron);
  let gradient = tetrahedronShapeGradients(tetrahedron)[localVertex];
  var columns: array<vec3f, 3>;
  for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
    var axis = vec3f(0.0);
    axis[coordinate] = 1.0;
    let direction = mat3x3f(
      axis * gradient.x,
      axis * gradient.y,
      axis * gradient.z,
    );
    let stressDirection = snh_tangent_product(
      deformationGradient,
      direction,
      item.attributes.y,
      item.attributes.z,
    );
    columns[coordinate] = item.attributes.x * stressDirection * gradient;
  }
  return mat3x3f(columns[0], columns[1], columns[2]);
}

fn tetrahedronGradient(tetrahedron: u32) -> array<vec3f, 4> {
  if (usesStableNeoHookean(tetrahedron)) {
    return stableTetrahedronGradient(tetrahedron);
  }
  let indices = tetData[tetrahedron].indices;
  let rotation = loadDiagnosticRotation(tetrahedron);
  let inverseRotation = transpose(rotation);
  var displacement: array<vec3f, 4>;
  var gradient: array<vec3f, 4>;
  var currentCenter = vec3f(0.0);
  var restCenter = vec3f(0.0);
  for (var local = 0u; local < 4u; local += 1u) {
    let vertex = indices[local];
    currentCenter += 0.25 * loadPosition(params.offsets.x, vertex);
    restCenter += 0.25 * vertexData[vertex].restMass.xyz;
  }
  for (var local = 0u; local < 4u; local += 1u) {
    let vertex = indices[local];
    displacement[local] =
      inverseRotation * (loadPosition(params.offsets.x, vertex) - currentCenter) -
      (vertexData[vertex].restMass.xyz - restCenter);
  }
  for (var row = 0u; row < 4u; row += 1u) {
    var localGradient = vec3f(0.0);
    for (var column = 0u; column < 4u; column += 1u) {
      localGradient +=
        stiffnessBlock(tetrahedron, row, column) * displacement[column];
    }
    gradient[row] = rotation * localGradient;
  }
  return gradient;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn evaluateDiagnosticTetrahedra(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let tetrahedron = globalId.x;
  if (tetrahedron >= params.counts.y) {
    return;
  }

  let item = tetData[tetrahedron];
  var current: array<vec3f, 4>;
  var rest: array<vec3f, 4>;
  for (var local = 0u; local < 4u; local += 1u) {
    let vertex = item.indices[local];
    current[local] = loadPosition(params.offsets.x, vertex);
    rest[local] = vertexData[vertex].restMass.xyz;
  }
  let deformedShape = mat3x3f(
    current[1] - current[0],
    current[2] - current[0],
    current[3] - current[0],
  );
  let inverseDm = mat3x3f(item.invDm0.xyz, item.invDm1.xyz, item.invDm2.xyz);
  let deformationGradient = deformedShape * inverseDm;
  let determinantValue = matrixDeterminant(deformationGradient);
  let rotation = polarRotation(deformationGradient);
  let inverseRotation = transpose(rotation);

  var displacement: array<vec3f, 4>;
  var localGradient: array<vec3f, 4>;
  var allFinite = finiteMat3(deformationGradient) &&
    finiteScalar(determinantValue) && finiteMat3(rotation);
  let currentCenter = 0.25 * (current[0] + current[1] + current[2] + current[3]);
  let restCenter = 0.25 * (rest[0] + rest[1] + rest[2] + rest[3]);
  for (var local = 0u; local < 4u; local += 1u) {
    displacement[local] =
      inverseRotation * (current[local] - currentCenter) -
      (rest[local] - restCenter);
    allFinite = allFinite && finiteVec3(current[local]) &&
      finiteVec3(rest[local]) && finiteVec3(displacement[local]);
  }
  for (var row = 0u; row < 4u; row += 1u) {
    localGradient[row] = vec3f(0.0);
    for (var column = 0u; column < 4u; column += 1u) {
      localGradient[row] +=
        stiffnessBlock(tetrahedron, row, column) * displacement[column];
    }
    allFinite = allFinite && finiteVec3(localGradient[row]);
  }
  var elasticityEnergy = 0.0;
  for (var local = 0u; local < 4u; local += 1u) {
    elasticityEnergy += 0.5 * dot(displacement[local], localGradient[local]);
  }
  if (usesStableNeoHookean(tetrahedron)) {
    elasticityEnergy = item.attributes.x * snh_energy_density(
      deformationGradient,
      item.attributes.y,
      item.attributes.z,
    );
    allFinite = allFinite && finiteMat3(snh_first_piola(
      deformationGradient,
      item.attributes.y,
      item.attributes.z,
    ));
  }
  allFinite = allFinite && finiteScalar(elasticityEnergy);

  let rotationBase = params.offsets.z + tetrahedron * 3u;
  diagnosticData[rotationBase] = vec4f(finiteVector(rotation[0]), 0.0);
  diagnosticData[rotationBase + 1u] = vec4f(finiteVector(rotation[1]), 0.0);
  diagnosticData[rotationBase + 2u] = vec4f(finiteVector(rotation[2]), 0.0);
  diagnosticData[params.offsets.w + tetrahedron] = vec4f(
    finiteNumber(elasticityEnergy),
    finiteNumber(determinantValue),
    select(0.0, 1.0, allFinite),
    0.0,
  );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn evaluateDiagnosticVertices(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let vertex = globalId.x;
  if (vertex >= params.counts.x) {
    return;
  }

  let item = vertexData[vertex];
  let current = loadPosition(params.offsets.x, vertex);
  let predicted = loadPosition(params.offsets.y, vertex);
  let difference = current - predicted;
  let finalUpdate = dynamicData[params.records.z + vertex];
  let signedFloorDistance = current.y - params.physics.y;
  let isActive = item.info.z == 0u;
  var gradient = vec3f(0.0);
  var inertiaGradient = vec3f(0.0);
  var elasticityGradient = vec3f(0.0);
  var floorGradient = vec3f(0.0);
  var hessian = zeroMat3();
  var inertiaEnergy = 0.0;
  var floorEnergy = 0.0;
  var updateMagnitude = 0.0;
  var updateValid = false;
  var activeFloorContact = false;
  var inertiaWeight = 0.0;
  var allFinite = finiteVec3(current) && finiteVec3(predicted) &&
    finiteVec3(item.restMass.xyz) && finiteScalar(item.restMass.w) &&
    finiteScalar(finalUpdate.x) && finiteScalar(finalUpdate.w) &&
    finiteScalar(signedFloorDistance);

  if (isActive) {
    inertiaWeight = item.restMass.w * params.physics.x;
    inertiaGradient = inertiaWeight * difference;
    gradient = inertiaGradient;
    hessian = inertiaWeight * identityMat3();
    inertiaEnergy = 0.5 * inertiaWeight * dot(difference, difference);
    updateMagnitude = finalUpdate.x;
    updateValid = finalUpdate.w >= 0.5 && updateMagnitude >= 0.0;

    for (
      var incident = 0u;
      incident < item.info.y;
      incident += 1u
    ) {
      let tetrahedron = adjacency[item.info.x + incident];
      let indices = tetData[tetrahedron].indices;
      let slot = localTetSlot(indices, vertex);
      if (slot >= 4u) {
        allFinite = false;
        continue;
      }
      let rotation = loadDiagnosticRotation(tetrahedron);
      let elementGradient = tetrahedronGradient(tetrahedron);
      elasticityGradient += elementGradient[slot];
      if (usesStableNeoHookean(tetrahedron)) {
        hessian += stableTetrahedronLocalHessian(tetrahedron, slot);
      } else {
        hessian += rotation * stiffnessBlock(tetrahedron, slot, slot) *
          transpose(rotation);
      }
    }
    gradient += elasticityGradient;

    activeFloorContact =
      params.physics.z > 0.0 && signedFloorDistance < 0.0;
    if (activeFloorContact) {
      floorEnergy =
        0.5 * params.physics.z * signedFloorDistance * signedFloorDistance;
      floorGradient.y = params.physics.z * signedFloorDistance;
      gradient += floorGradient;
      hessian[1][1] += params.physics.z;
    }
  }

  let gradientSquaredNorm = dot(gradient, gradient);
  let inertiaGradientSquaredNorm = dot(inertiaGradient, inertiaGradient);
  let elasticityGradientSquaredNorm =
    dot(elasticityGradient, elasticityGradient);
  let floorGradientSquaredNorm = dot(floorGradient, floorGradient);
  allFinite = allFinite && finiteScalar(inertiaWeight) &&
    finiteScalar(inertiaEnergy) && finiteScalar(floorEnergy) &&
    finiteScalar(updateMagnitude) && finiteVec3(gradient) &&
    finiteVec3(inertiaGradient) && finiteVec3(elasticityGradient) &&
    finiteVec3(floorGradient) && finiteMat3(hessian) &&
    finiteScalar(gradientSquaredNorm) &&
    finiteScalar(inertiaGradientSquaredNorm) &&
    finiteScalar(elasticityGradientSquaredNorm) &&
    finiteScalar(floorGradientSquaredNorm);

  let base = params.records.x + vertex * VERTEX_RECORD_VEC4S;
  diagnosticData[base] = vec4f(
    finiteVector(gradient),
    select(0.0, 1.0, isActive),
  );
  // Matrices are returned row-major; WGSL indexes matrices by column, then row.
  diagnosticData[base + 1u] = vec4f(
    finiteNumber(hessian[0][0]),
    finiteNumber(hessian[1][0]),
    finiteNumber(hessian[2][0]),
    finiteNumber(inertiaEnergy),
  );
  diagnosticData[base + 2u] = vec4f(
    finiteNumber(hessian[0][1]),
    finiteNumber(hessian[1][1]),
    finiteNumber(hessian[2][1]),
    finiteNumber(floorEnergy),
  );
  diagnosticData[base + 3u] = vec4f(
    finiteNumber(hessian[0][2]),
    finiteNumber(hessian[1][2]),
    finiteNumber(hessian[2][2]),
    finiteNumber(updateMagnitude),
  );
  diagnosticData[base + 4u] = vec4f(
    finiteNumber(gradientSquaredNorm),
    select(0.0, 1.0, allFinite),
    finiteNumber(inertiaWeight),
    select(0.0, 1.0, updateValid),
  );
  diagnosticData[base + 5u] = vec4f(
    finiteNumber(inertiaGradientSquaredNorm),
    finiteNumber(elasticityGradientSquaredNorm),
    finiteNumber(floorGradientSquaredNorm),
    finiteNumber(signedFloorDistance),
  );
  diagnosticData[base + 6u] = vec4f(
    select(0.0, 1.0, activeFloorContact),
    0.0,
    0.0,
    0.0,
  );
}

// Diagnostics are opt-in and intended for oracle-sized scenes. A serial final
// reduction gives deterministic mesh-order accumulation without f32 atomics.
@compute @workgroup_size(1)
fn reduceDiagnostics(@builtin(global_invocation_id) globalId: vec3u) {
  if (globalId.x != 0u) {
    return;
  }

  var elasticityEnergy = 0.0;
  var inertiaEnergy = 0.0;
  var floorEnergy = 0.0;
  var gradientSquaredNorm = 0.0;
  var inertiaGradientSquaredNorm = 0.0;
  var elasticityGradientSquaredNorm = 0.0;
  var floorGradientSquaredNorm = 0.0;
  var maximumUpdate = 0.0;
  var minimumDeterminant = 3.0e38;
  var minimumContactDistance = 3.0e38;
  var activeVertexCount = 0.0;
  var activeContactCount = 0.0;
  var allActiveUpdatesValid = true;
  var allFinite = true;

  for (var tetrahedron = 0u; tetrahedron < params.counts.y; tetrahedron += 1u) {
    let metric = diagnosticData[params.offsets.w + tetrahedron];
    elasticityEnergy += metric.x;
    minimumDeterminant = min(minimumDeterminant, metric.y);
    allFinite = allFinite && metric.z >= 0.5;
  }
  for (var vertex = 0u; vertex < params.counts.x; vertex += 1u) {
    let base = params.records.x + vertex * VERTEX_RECORD_VEC4S;
    let gradientAndActive = diagnosticData[base];
    let row0 = diagnosticData[base + 1u];
    let row1 = diagnosticData[base + 2u];
    let row2 = diagnosticData[base + 3u];
    let metric = diagnosticData[base + 4u];
    let componentMetrics = diagnosticData[base + 5u];
    let contactMetrics = diagnosticData[base + 6u];
    let isActive = gradientAndActive.w >= 0.5;
    inertiaEnergy += row0.w;
    floorEnergy += row1.w;
    gradientSquaredNorm += metric.x;
    inertiaGradientSquaredNorm += componentMetrics.x;
    elasticityGradientSquaredNorm += componentMetrics.y;
    floorGradientSquaredNorm += componentMetrics.z;
    if (isActive) {
      maximumUpdate = max(maximumUpdate, row2.w);
      minimumContactDistance = min(minimumContactDistance, componentMetrics.w);
      activeContactCount += select(0.0, 1.0, contactMetrics.x >= 0.5);
      allActiveUpdatesValid = allActiveUpdatesValid && metric.w >= 0.5;
    }
    activeVertexCount += select(0.0, 1.0, isActive);
    allFinite = allFinite && metric.y >= 0.5;
  }

  let determinantValid = params.counts.y > 0u;
  if (!determinantValid) {
    minimumDeterminant = 0.0;
  }
  let gradientValid = activeVertexCount > 0.0;
  let maximumUpdateValid = gradientValid && allActiveUpdatesValid;
  let floorMetricsValid = params.physics.z > 0.0 && gradientValid;
  if (!floorMetricsValid) {
    minimumContactDistance = 0.0;
    activeContactCount = 0.0;
  }
  let gradientNorm = sqrt(max(gradientSquaredNorm, 0.0));
  let inertiaGradientNorm = sqrt(max(inertiaGradientSquaredNorm, 0.0));
  let elasticityGradientNorm = sqrt(max(elasticityGradientSquaredNorm, 0.0));
  let floorGradientNorm = sqrt(max(floorGradientSquaredNorm, 0.0));
  let residualDenominator = max(
    1.0,
    inertiaGradientNorm + elasticityGradientNorm + floorGradientNorm,
  );
  let relativeResidual = gradientNorm / residualDenominator;
  allFinite = allFinite && finiteScalar(elasticityEnergy) &&
    finiteScalar(inertiaEnergy) && finiteScalar(floorEnergy) &&
    finiteScalar(gradientSquaredNorm) && finiteScalar(gradientNorm) &&
    finiteScalar(inertiaGradientSquaredNorm) &&
    finiteScalar(elasticityGradientSquaredNorm) &&
    finiteScalar(floorGradientSquaredNorm) &&
    finiteScalar(residualDenominator) && finiteScalar(relativeResidual) &&
    finiteScalar(maximumUpdate) && finiteScalar(minimumDeterminant) &&
    finiteScalar(minimumContactDistance) && finiteScalar(activeContactCount);
  let relativeResidualValid = maximumUpdateValid && allFinite;

  let summary = params.records.y;
  diagnosticData[summary] = vec4f(
    finiteNumber(inertiaEnergy),
    finiteNumber(elasticityEnergy),
    finiteNumber(floorEnergy),
    finiteNumber(minimumContactDistance),
  );
  diagnosticData[summary + 1u] = vec4f(
    finiteNumber(gradientNorm),
    finiteNumber(maximumUpdate),
    finiteNumber(minimumDeterminant),
    select(0.0, 1.0, allFinite),
  );
  diagnosticData[summary + 2u] = vec4f(
    finiteNumber(gradientSquaredNorm),
    activeVertexCount,
    select(0.0, 1.0, gradientValid),
    select(0.0, 1.0, determinantValid),
  );
  diagnosticData[summary + 3u] = vec4f(
    finiteNumber(activeContactCount),
    select(0.0, 1.0, floorMetricsValid),
    select(0.0, 1.0, floorMetricsValid),
    select(0.0, 1.0, maximumUpdateValid),
  );
  diagnosticData[summary + 4u] = vec4f(
    finiteNumber(gradientNorm),
    finiteNumber(residualDenominator),
    finiteNumber(relativeResidual),
    select(0.0, 1.0, relativeResidualValid),
  );
}
`;

function formatCompilationErrors(
  messages: readonly GPUCompilationMessage[],
): string {
  return messages
    .filter((message) => message.type === "error")
    .map(
      (message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`,
    )
    .join("\n");
}

function encodeDispatch(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  itemCount: number,
  label: string,
): void {
  if (itemCount === 0) {
    return;
  }
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(itemCount / JGS2_DIAGNOSTICS_WORKGROUP_SIZE),
  );
  pass.end();
}

/** Lazily-created, explicitly-invoked GPU exact-oracle diagnostic path. */
export class JGS2GpuOracleEvaluator {
  private destroyed = false;
  private evaluationTail: Promise<void> = Promise.resolve();

  private constructor(
    private readonly device: GPUDevice,
    private readonly vertexCount: number,
    private readonly tetCount: number,
    private readonly layout: JGS2DiagnosticsLayout,
    private readonly scratch: GPUBuffer,
    private readonly uniforms: GPUBuffer,
    private readonly bindGroup: GPUBindGroup,
    private readonly tetPipeline: GPUComputePipeline,
    private readonly vertexPipeline: GPUComputePipeline,
    private readonly reductionPipeline: GPUComputePipeline,
  ) {}

  static async create(
    device: GPUDevice,
    vertexCount: number,
    tetCount: number,
    source: JGS2GpuDiagnosticsSourceBuffers,
  ): Promise<JGS2GpuOracleEvaluator> {
    const layout = computeJGS2DiagnosticsLayout(vertexCount, tetCount);
    if (
      layout.byteLength > device.limits.maxStorageBufferBindingSize ||
      layout.byteLength > device.limits.maxBufferSize
    ) {
      throw new RangeError(
        `JGS2 diagnostics need ${layout.byteLength} bytes, exceeding this adapter's limits.`,
      );
    }

    const scratch = device.createBuffer({
      label: "jgs2-oracle-diagnostics-scratch",
      size: layout.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniforms = device.createBuffer({
      label: "jgs2-oracle-diagnostics-uniforms",
      size: JGS2_DIAGNOSTICS_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    try {
      const bindGroupLayout = device.createBindGroupLayout({
        label: "jgs2-oracle-diagnostics-bind-group-layout",
        entries: [
          ...[0, 1, 2, 3, 4].map<GPUBindGroupLayoutEntry>((binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          })),
          {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 6,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: "uniform",
              minBindingSize: JGS2_DIAGNOSTICS_UNIFORM_BYTES,
            },
          },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({
        label: "jgs2-oracle-diagnostics-pipeline-layout",
        bindGroupLayouts: [bindGroupLayout],
      });
      const shader = device.createShaderModule({
        label: "jgs2-oracle-diagnostics-shader",
        code: jgs2DiagnosticsShader,
      });
      const compilation = await shader.getCompilationInfo();
      const errors = formatCompilationErrors(compilation.messages);
      if (errors) {
        throw new Error(`JGS2 diagnostics WGSL failed to compile:\n${errors}`);
      }
      const createPipeline = (entryPoint: string, label: string) =>
        device.createComputePipelineAsync({
          label,
          layout: pipelineLayout,
          compute: { module: shader, entryPoint },
        });
      const [tetPipeline, vertexPipeline, reductionPipeline] =
        await Promise.all([
          createPipeline(
            "evaluateDiagnosticTetrahedra",
            "jgs2-oracle-diagnostics-tetrahedron-pipeline",
          ),
          createPipeline(
            "evaluateDiagnosticVertices",
            "jgs2-oracle-diagnostics-vertex-pipeline",
          ),
          createPipeline(
            "reduceDiagnostics",
            "jgs2-oracle-diagnostics-reduction-pipeline",
          ),
        ]);
      const bindGroup = device.createBindGroup({
        label: "jgs2-oracle-diagnostics-bind-group",
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: source.dynamic } },
          { binding: 1, resource: { buffer: source.vertices } },
          { binding: 2, resource: { buffer: source.tets } },
          { binding: 3, resource: { buffer: source.stiffness } },
          { binding: 4, resource: { buffer: source.adjacency } },
          { binding: 5, resource: { buffer: scratch } },
          { binding: 6, resource: { buffer: uniforms } },
        ],
      });
      return new JGS2GpuOracleEvaluator(
        device,
        vertexCount,
        tetCount,
        layout,
        scratch,
        uniforms,
        bindGroup,
        tetPipeline,
        vertexPipeline,
        reductionPipeline,
      );
    } catch (error) {
      scratch.destroy();
      uniforms.destroy();
      throw error;
    }
  }

  evaluate(
    settings: JGS2GpuDiagnosticsEvaluationSettings,
  ): Promise<JGS2GpuOracleDiagnostics> {
    this.assertUsable();
    const evaluation = this.evaluationTail.then(() =>
      this.evaluateImmediately(settings),
    );
    this.evaluationTail = evaluation.then(
      () => undefined,
      () => undefined,
    );
    return evaluation;
  }

  private async evaluateImmediately(
    settings: JGS2GpuDiagnosticsEvaluationSettings,
  ): Promise<JGS2GpuOracleDiagnostics> {
    this.assertUsable();
    this.device.queue.writeBuffer(
      this.uniforms,
      0,
      packJGS2DiagnosticsUniforms(
        this.vertexCount,
        this.tetCount,
        this.layout,
        settings,
      ),
    );
    const readback = this.device.createBuffer({
      label: "jgs2-oracle-diagnostics-readback",
      size: this.layout.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "jgs2-oracle-diagnostics-command-encoder",
    });
    encodeDispatch(
      encoder,
      this.tetPipeline,
      this.bindGroup,
      this.tetCount,
      "jgs2-oracle-diagnostics-tetrahedron-pass",
    );
    encodeDispatch(
      encoder,
      this.vertexPipeline,
      this.bindGroup,
      this.vertexCount,
      "jgs2-oracle-diagnostics-vertex-pass",
    );
    encodeDispatch(
      encoder,
      this.reductionPipeline,
      this.bindGroup,
      1,
      "jgs2-oracle-diagnostics-reduction-pass",
    );
    encoder.copyBufferToBuffer(
      this.scratch,
      0,
      readback,
      0,
      this.layout.byteLength,
    );
    this.device.queue.submit([encoder.finish()]);

    try {
      await readback.mapAsync(GPUMapMode.READ);
      const packed = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      return decodeJGS2GpuOracleDiagnostics(
        packed,
        this.vertexCount,
        this.tetCount,
        this.layout,
      );
    } finally {
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.scratch.destroy();
    this.uniforms.destroy();
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("JGS2GpuOracleEvaluator has been destroyed.");
    }
  }
}
