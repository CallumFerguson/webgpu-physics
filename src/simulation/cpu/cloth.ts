const TRIANGLE_DOF = 9;
const HINGE_DOF = 12;
const RELATIVE_DEGENERACY_TOLERANCE = 1e-12;

type Vector3 = readonly [number, number, number];

export interface StVKMembraneMaterial {
  readonly youngModulus: number;
  readonly poissonRatio: number;
  readonly thickness: number;
}

/** Rest data for one triangle, expressed in its intrinsic 2D material basis. */
export interface StVKTriangleRestData {
  readonly restArea: number;
  /** Row-major inverse of the 2 by 2 rest-edge matrix. */
  readonly inverseRestBasis: Float64Array;
}

export interface StVKTriangleMembraneEvaluation {
  readonly energy: number;
  /** Gradient with respect to the triangle's three packed xyz positions. */
  readonly gradient: Float64Array;
  /** Row-major 3 by 2 surface deformation gradient. */
  readonly deformationGradient: Float64Array;
  /** Row-major 2 by 2 Green strain. */
  readonly greenStrain: Float64Array;
}

export interface QuadraticDihedralRestData {
  readonly restAngle: number;
  readonly restEdgeLength: number;
}

export interface QuadraticDihedralBendingEvaluation {
  readonly energy: number;
  /** Gradient with respect to the hinge's four packed xyz positions. */
  readonly gradient: Float64Array;
  readonly signedDihedral: number;
  readonly angleDifference: number;
}

function assertFiniteValues(values: ArrayLike<number>, label: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${label} must contain only finite values.`);
    }
  }
}

function assertPackedPositions(
  positions: ArrayLike<number>,
  expectedLength: number,
  label: string,
): void {
  if (positions.length !== expectedLength) {
    throw new RangeError(`${label} must contain ${expectedLength / 3} xyz vertices.`);
  }
  assertFiniteValues(positions, label);
}

function vertex(positions: ArrayLike<number>, index: number): Vector3 {
  const base = index * 3;
  return [positions[base]!, positions[base + 1]!, positions[base + 2]!];
}

function add(a: Vector3, b: Vector3): Vector3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a: Vector3, b: Vector3): Vector3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(value: Vector3, scalar: number): Vector3 {
  return [value[0] * scalar, value[1] * scalar, value[2] * scalar];
}

function dot(a: Vector3, b: Vector3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vector3, b: Vector3): Vector3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(value: Vector3): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function assertNondegenerateTriangle(
  edge0: Vector3,
  edge1: Vector3,
  label: string,
): number {
  const twiceArea = norm(cross(edge0, edge1));
  const edgeScale = Math.max(norm(edge0), norm(edge1), norm(subtract(edge1, edge0)));
  if (
    !(edgeScale > 0) ||
    !(twiceArea > RELATIVE_DEGENERACY_TOLERANCE * edgeScale * edgeScale)
  ) {
    throw new RangeError(`${label} must be nondegenerate.`);
  }
  return twiceArea;
}

/**
 * Build the intrinsic 2D rest basis for one packed xyz triangle. The first
 * material axis follows rest edge (0, 1); the second lies in the rest plane.
 */
export function precomputeStVKTriangleRest(
  restPositions: ArrayLike<number>,
): StVKTriangleRestData {
  assertPackedPositions(restPositions, TRIANGLE_DOF, "Rest triangle positions");
  const x0 = vertex(restPositions, 0);
  const edge01 = subtract(vertex(restPositions, 1), x0);
  const edge02 = subtract(vertex(restPositions, 2), x0);
  const twiceArea = assertNondegenerateTriangle(edge01, edge02, "Rest triangle");
  const edgeLength = norm(edge01);
  const tangent = scale(edge01, 1 / edgeLength);
  const normal = scale(cross(edge01, edge02), 1 / twiceArea);
  const bitangent = cross(normal, tangent);
  const materialX = dot(edge02, tangent);
  const materialY = dot(edge02, bitangent);

  // Dm = [ [edgeLength, materialX], [0, materialY] ].
  return {
    restArea: 0.5 * twiceArea,
    inverseRestBasis: new Float64Array([
      1 / edgeLength,
      -materialX / (edgeLength * materialY),
      0,
      1 / materialY,
    ]),
  };
}

function validateMembraneMaterial(material: StVKMembraneMaterial): void {
  if (!(material.youngModulus > 0) || !Number.isFinite(material.youngModulus)) {
    throw new RangeError("A finite, positive membrane Young's modulus is required.");
  }
  if (
    !(material.poissonRatio >= 0 && material.poissonRatio < 0.5) ||
    !Number.isFinite(material.poissonRatio)
  ) {
    throw new RangeError("Membrane Poisson ratio must be finite and in [0, 0.5).");
  }
  if (!(material.thickness > 0) || !Number.isFinite(material.thickness)) {
    throw new RangeError("A finite, positive membrane thickness is required.");
  }
}

/**
 * Evaluate a St. Venant-Kirchhoff membrane triangle using plane-stress Lamé
 * parameters. The gradient is analytic: dW/dDs = area * thickness * P * Dm^-T.
 */
export function evaluateStVKTriangleMembrane(
  rest: StVKTriangleRestData,
  positions: ArrayLike<number>,
  material: StVKMembraneMaterial,
): StVKTriangleMembraneEvaluation {
  assertPackedPositions(positions, TRIANGLE_DOF, "Triangle positions");
  validateMembraneMaterial(material);
  if (!(rest.restArea > 0) || !Number.isFinite(rest.restArea)) {
    throw new RangeError("The membrane rest area must be finite and positive.");
  }
  if (rest.inverseRestBasis.length !== 4) {
    throw new RangeError("The inverse rest basis must be a 2 by 2 matrix.");
  }
  assertFiniteValues(rest.inverseRestBasis, "Inverse rest basis");

  const x0 = vertex(positions, 0);
  const edge01 = subtract(vertex(positions, 1), x0);
  const edge02 = subtract(vertex(positions, 2), x0);
  const inverse = rest.inverseRestBasis;
  const deformationGradient = new Float64Array(6);
  for (let row = 0; row < 3; row += 1) {
    deformationGradient[row * 2] =
      edge01[row]! * inverse[0]! + edge02[row]! * inverse[2]!;
    deformationGradient[row * 2 + 1] =
      edge01[row]! * inverse[1]! + edge02[row]! * inverse[3]!;
  }

  let c00 = 0;
  let c01 = 0;
  let c11 = 0;
  for (let row = 0; row < 3; row += 1) {
    const f0 = deformationGradient[row * 2]!;
    const f1 = deformationGradient[row * 2 + 1]!;
    c00 += f0 * f0;
    c01 += f0 * f1;
    c11 += f1 * f1;
  }
  const e00 = 0.5 * (c00 - 1);
  const e01 = 0.5 * c01;
  const e11 = 0.5 * (c11 - 1);
  const trace = e00 + e11;
  const mu = material.youngModulus / (2 * (1 + material.poissonRatio));
  const lambda =
    (material.youngModulus * material.poissonRatio) /
    (1 - material.poissonRatio * material.poissonRatio);
  const density =
    mu * (e00 * e00 + 2 * e01 * e01 + e11 * e11) +
    0.5 * lambda * trace * trace;
  const scaleFactor = rest.restArea * material.thickness;

  // Second Piola stress S = 2 mu E + lambda tr(E) I.
  const s00 = 2 * mu * e00 + lambda * trace;
  const s01 = 2 * mu * e01;
  const s11 = 2 * mu * e11 + lambda * trace;
  const gradient = new Float64Array(TRIANGLE_DOF);
  for (let row = 0; row < 3; row += 1) {
    const f0 = deformationGradient[row * 2]!;
    const f1 = deformationGradient[row * 2 + 1]!;
    const p0 = f0 * s00 + f1 * s01;
    const p1 = f0 * s01 + f1 * s11;
    const edgeGradient0 =
      scaleFactor * (p0 * inverse[0]! + p1 * inverse[1]!);
    const edgeGradient1 =
      scaleFactor * (p0 * inverse[2]! + p1 * inverse[3]!);
    gradient[row] = -edgeGradient0 - edgeGradient1;
    gradient[3 + row] = edgeGradient0;
    gradient[6 + row] = edgeGradient1;
  }

  const energy = scaleFactor * density;
  if (!Number.isFinite(energy)) {
    throw new RangeError("The membrane evaluation overflowed.");
  }
  assertFiniteValues(gradient, "Membrane gradient");
  return {
    energy,
    gradient,
    deformationGradient,
    greenStrain: new Float64Array([e00, e01, e01, e11]),
  };
}

interface DihedralEvaluation {
  readonly angle: number;
  readonly gradient: Float64Array;
  readonly edgeLength: number;
}

function normalizedDifferential(
  unit: Vector3,
  length: number,
  differential: Vector3,
): Vector3 {
  return scale(subtract(differential, scale(unit, dot(unit, differential))), 1 / length);
}

/** Analytic differential of atan2(t dot (n0 cross n1), n0 dot n1). */
function evaluateSignedDihedral(positions: ArrayLike<number>): DihedralEvaluation {
  assertPackedPositions(positions, HINGE_DOF, "Hinge positions");
  const x0 = vertex(positions, 0);
  const x1 = vertex(positions, 1);
  const x2 = vertex(positions, 2);
  const x3 = vertex(positions, 3);
  const edge = subtract(x1, x0);
  const edgeLength = norm(edge);
  if (!(edgeLength > 0)) {
    throw new RangeError("The hinge edge must have positive length.");
  }
  const tangent = scale(edge, 1 / edgeLength);
  const to2 = subtract(x2, x0);
  const reverseEdge = scale(edge, -1);
  const to3 = subtract(x3, x1);
  const rawNormal0 = cross(edge, to2);
  const rawNormal1 = cross(reverseEdge, to3);
  const normalLength0 = assertNondegenerateTriangle(edge, to2, "First hinge triangle");
  const normalLength1 = assertNondegenerateTriangle(reverseEdge, to3, "Second hinge triangle");
  const normal0 = scale(rawNormal0, 1 / normalLength0);
  const normal1 = scale(rawNormal1, 1 / normalLength1);
  const normalCross = cross(normal0, normal1);
  const sine = dot(tangent, normalCross);
  const cosine = dot(normal0, normal1);
  const denominator = sine * sine + cosine * cosine;
  const gradient = new Float64Array(HINGE_DOF);

  for (let coordinate = 0; coordinate < HINGE_DOF; coordinate += 1) {
    const differentiatedVertex = Math.floor(coordinate / 3);
    const axis = coordinate % 3;
    const deltas: Vector3[] = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const basis = [0, 0, 0] as [number, number, number];
    basis[axis] = 1;
    deltas[differentiatedVertex] = basis;
    const differentialEdge = subtract(deltas[1]!, deltas[0]!);
    const differentialTangent = normalizedDifferential(
      tangent,
      edgeLength,
      differentialEdge,
    );
    const differentialTo2 = subtract(deltas[2]!, deltas[0]!);
    const differentialTo3 = subtract(deltas[3]!, deltas[1]!);
    const differentialNormal0Raw = add(
      cross(differentialEdge, to2),
      cross(edge, differentialTo2),
    );
    const differentialNormal1Raw = add(
      cross(scale(differentialEdge, -1), to3),
      cross(reverseEdge, differentialTo3),
    );
    const differentialNormal0 = normalizedDifferential(
      normal0,
      normalLength0,
      differentialNormal0Raw,
    );
    const differentialNormal1 = normalizedDifferential(
      normal1,
      normalLength1,
      differentialNormal1Raw,
    );
    const differentialCosine =
      dot(differentialNormal0, normal1) + dot(normal0, differentialNormal1);
    const differentialSine =
      dot(differentialTangent, normalCross) +
      dot(
        tangent,
        add(
          cross(differentialNormal0, normal1),
          cross(normal0, differentialNormal1),
        ),
      );
    gradient[coordinate] =
      (cosine * differentialSine - sine * differentialCosine) / denominator;
  }

  return { angle: Math.atan2(sine, cosine), gradient, edgeLength };
}

/**
 * Precompute an oriented hinge `(0, 1, 2)` / `(1, 0, 3)`. With consistently
 * oriented coplanar triangles, the signed rest angle is zero.
 */
export function precomputeQuadraticDihedralRest(
  restPositions: ArrayLike<number>,
): QuadraticDihedralRestData {
  const rest = evaluateSignedDihedral(restPositions);
  return { restAngle: rest.angle, restEdgeLength: rest.edgeLength };
}

function wrappedAngleDifference(angle: number, reference: number): number {
  const difference = angle - reference;
  return Math.atan2(Math.sin(difference), Math.cos(difference));
}

/**
 * Evaluate 0.5 * stiffness * restEdgeLength * (theta - thetaRest)^2.
 * The rest-edge weighting keeps the reference deliberately simple and local.
 */
export function evaluateQuadraticDihedralBending(
  rest: QuadraticDihedralRestData,
  positions: ArrayLike<number>,
  stiffness: number,
): QuadraticDihedralBendingEvaluation {
  if (!(stiffness >= 0) || !Number.isFinite(stiffness)) {
    throw new RangeError("Dihedral stiffness must be finite and nonnegative.");
  }
  if (!(rest.restEdgeLength > 0) || !Number.isFinite(rest.restEdgeLength)) {
    throw new RangeError("The rest hinge edge length must be finite and positive.");
  }
  if (!Number.isFinite(rest.restAngle)) {
    throw new RangeError("The rest dihedral angle must be finite.");
  }
  const current = evaluateSignedDihedral(positions);
  const angleDifference = wrappedAngleDifference(current.angle, rest.restAngle);
  const weightedStiffness = stiffness * rest.restEdgeLength;
  const gradientScale = weightedStiffness * angleDifference;
  return {
    energy: 0.5 * weightedStiffness * angleDifference * angleDifference,
    gradient:
      gradientScale === 0
        ? new Float64Array(HINGE_DOF)
        : Float64Array.from(
            current.gradient,
            (value) => gradientScale * value,
          ),
    signedDihedral: current.angle,
    angleDifference,
  };
}
