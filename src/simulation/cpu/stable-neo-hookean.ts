import type { DifferentiableEnergyTerm } from "./energy";
import { computeTetrahedronShapeGradients } from "./fem";
import { determinant3, multiply3 } from "./math";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import type {
  LinearMaterial,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

const MATRIX_SIZE = 9;
const TETRAHEDRON_DOF = 12;

export interface StableNeoHookeanParameters {
  /** Hooke-law Lamé parameter derived directly from Young/Poisson. */
  readonly lameLambda: number;
  /** Hooke-law shear modulus derived directly from Young/Poisson. */
  readonly lameMu: number;
  /** Reparameterized lambda from Smith, de Goes, and Kim, equation 15. */
  readonly lambda: number;
  /** Reparameterized mu from Smith, de Goes, and Kim, equation 15. */
  readonly mu: number;
  /** Stress-free determinant target from the corrected equation 14. */
  readonly alpha: number;
  /** Constant subtracted so the rest energy density is exactly zero. */
  readonly restEnergyDensity: number;
}

export interface StableNeoHookeanDensityEvaluation {
  readonly energyDensity: number;
  readonly firstPiola: Float64Array;
  /** Row-major 9 by 9 derivative of row-major firstPiola by F. */
  readonly tangent: Float64Array;
  readonly deformationDeterminant: number;
  readonly firstInvariant: number;
}

export interface StableNeoHookeanTetrahedronEvaluation {
  readonly energy: number;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
  readonly deformationGradient: Float64Array;
  readonly deformationDeterminant: number;
}

export interface StableNeoHookeanMeshEvaluation {
  readonly energy: number;
  readonly gradient: Float64Array;
  readonly hessian: Float64Array;
  readonly tetrahedronEnergies: Float64Array;
  readonly deformationDeterminants: Float64Array;
}

function assertFiniteValues(values: ArrayLike<number>, label: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${label} must contain only finite values.`);
    }
  }
}

/**
 * Convert familiar Young/Poisson parameters to the reparameterized constants
 * required by the stable Neo-Hookean energy. The conversion preserves the
 * infinitesimal Hooke-law response; using the unmodified Lamé constants would
 * make the nonlinear material disagree with the requested stiffness.
 */
export function computeStableNeoHookeanParameters(
  material: Pick<LinearMaterial, "name" | "youngModulus" | "poissonRatio">,
): StableNeoHookeanParameters {
  if (!(material.youngModulus > 0) || !Number.isFinite(material.youngModulus)) {
    throw new RangeError(
      `Material ${material.name} has invalid Young's modulus.`,
    );
  }
  if (
    !(material.poissonRatio >= 0 && material.poissonRatio < 0.5) ||
    !Number.isFinite(material.poissonRatio)
  ) {
    throw new RangeError(
      `Stable Neo-Hookean material ${material.name} requires Poisson ratio in [0, 0.5).`,
    );
  }

  const lameMu =
    material.youngModulus / (2 * (1 + material.poissonRatio));
  const lameLambda =
    (material.youngModulus * material.poissonRatio) /
    ((1 + material.poissonRatio) * (1 - 2 * material.poissonRatio));
  const mu = (4 / 3) * lameMu;
  const lambda = lameLambda + (5 / 6) * lameMu;
  if (!(lambda > 0) || !Number.isFinite(lambda)) {
    throw new RangeError(
      `Stable Neo-Hookean material ${material.name} has invalid lambda.`,
    );
  }
  const alpha = 1 + mu / lambda - mu / (4 * lambda);
  const restEnergyDensity =
    0.5 * lambda * (1 - alpha) ** 2 - 0.5 * mu * Math.log(4);

  return {
    lameLambda,
    lameMu,
    lambda,
    mu,
    alpha,
    restEnergyDensity,
  };
}

/** Polynomial cofactor matrix d det(F) / dF, valid without inverting F. */
export function cofactor3(matrix: ArrayLike<number>): Float64Array {
  if (matrix.length !== MATRIX_SIZE) {
    throw new RangeError("A cofactor requires a 3 by 3 matrix.");
  }
  return new Float64Array([
    matrix[4]! * matrix[8]! - matrix[5]! * matrix[7]!,
    matrix[5]! * matrix[6]! - matrix[3]! * matrix[8]!,
    matrix[3]! * matrix[7]! - matrix[4]! * matrix[6]!,
    matrix[2]! * matrix[7]! - matrix[1]! * matrix[8]!,
    matrix[0]! * matrix[8]! - matrix[2]! * matrix[6]!,
    matrix[1]! * matrix[6]! - matrix[0]! * matrix[7]!,
    matrix[1]! * matrix[5]! - matrix[2]! * matrix[4]!,
    matrix[2]! * matrix[3]! - matrix[0]! * matrix[5]!,
    matrix[0]! * matrix[4]! - matrix[1]! * matrix[3]!,
  ]);
}

/** Directional derivative of the polynomial cofactor matrix. */
export function directionalCofactor3(
  matrix: ArrayLike<number>,
  direction: ArrayLike<number>,
): Float64Array {
  if (matrix.length !== MATRIX_SIZE || direction.length !== MATRIX_SIZE) {
    throw new RangeError("A directional cofactor requires two 3 by 3 matrices.");
  }
  const f = matrix;
  const h = direction;
  return new Float64Array([
    h[4]! * f[8]! + f[4]! * h[8]! - h[5]! * f[7]! - f[5]! * h[7]!,
    h[5]! * f[6]! + f[5]! * h[6]! - h[3]! * f[8]! - f[3]! * h[8]!,
    h[3]! * f[7]! + f[3]! * h[7]! - h[4]! * f[6]! - f[4]! * h[6]!,
    h[2]! * f[7]! + f[2]! * h[7]! - h[1]! * f[8]! - f[1]! * h[8]!,
    h[0]! * f[8]! + f[0]! * h[8]! - h[2]! * f[6]! - f[2]! * h[6]!,
    h[1]! * f[6]! + f[1]! * h[6]! - h[0]! * f[7]! - f[0]! * h[7]!,
    h[1]! * f[5]! + f[1]! * h[5]! - h[2]! * f[4]! - f[2]! * h[4]!,
    h[2]! * f[3]! + f[2]! * h[3]! - h[0]! * f[5]! - f[0]! * h[5]!,
    h[0]! * f[4]! + f[0]! * h[4]! - h[1]! * f[3]! - f[1]! * h[3]!,
  ]);
}

/** Apply the exact material tangent to a deformation-gradient direction. */
export function applyStableNeoHookeanTangent(
  deformationGradient: ArrayLike<number>,
  direction: ArrayLike<number>,
  parameters: StableNeoHookeanParameters,
): Float64Array {
  if (
    deformationGradient.length !== MATRIX_SIZE ||
    direction.length !== MATRIX_SIZE
  ) {
    throw new RangeError("Stable Neo-Hookean tangents require 3 by 3 matrices.");
  }
  const cofactor = cofactor3(deformationGradient);
  const directionalCofactor = directionalCofactor3(
    deformationGradient,
    direction,
  );
  let firstInvariant = 0;
  let deformationDotDirection = 0;
  let determinantDirection = 0;
  for (let index = 0; index < MATRIX_SIZE; index += 1) {
    firstInvariant += deformationGradient[index]! ** 2;
    deformationDotDirection +=
      deformationGradient[index]! * direction[index]!;
    determinantDirection += cofactor[index]! * direction[index]!;
  }
  const invariantDenominator = firstInvariant + 1;
  const shearScale = parameters.mu * (1 - 1 / invariantDenominator);
  const directionalShearScale =
    (2 * parameters.mu * deformationDotDirection) /
    (invariantDenominator * invariantDenominator);
  const determinant = determinant3(deformationGradient);
  const volumeScale = parameters.lambda * (determinant - parameters.alpha);
  const directionalVolumeScale =
    parameters.lambda * determinantDirection;
  const result = new Float64Array(MATRIX_SIZE);
  for (let index = 0; index < MATRIX_SIZE; index += 1) {
    result[index] =
      shearScale * direction[index]! +
      directionalShearScale * deformationGradient[index]! +
      directionalVolumeScale * cofactor[index]! +
      volumeScale * directionalCofactor[index]!;
  }
  return result;
}

/**
 * Enforce the production feasibility domain separately from the raw material
 * oracle. Smith's density is deliberately finite through collapse and
 * inversion, while accepted simulation steps must remain above their recorded
 * positive determinant floor.
 */
export function assertStableNeoHookeanFeasible(
  deformationGradient: ArrayLike<number>,
  minimumDeterminant = 0,
): number {
  if (deformationGradient.length !== MATRIX_SIZE) {
    throw new RangeError("A deformation gradient must be a 3 by 3 matrix.");
  }
  assertFiniteValues(deformationGradient, "Deformation gradient");
  if (!(minimumDeterminant >= 0) || !Number.isFinite(minimumDeterminant)) {
    throw new RangeError("The minimum deformation determinant must be finite and nonnegative.");
  }
  const determinant = determinant3(deformationGradient);
  if (!(determinant > minimumDeterminant)) {
    throw new RangeError(
      `Deformation determinant ${determinant} does not exceed the feasibility floor ${minimumDeterminant}.`,
    );
  }
  return determinant;
}

/** Evaluate corrected equation 14 and its exact first and second derivatives. */
export function evaluateStableNeoHookeanDensity(
  deformationGradient: ArrayLike<number>,
  parameters: StableNeoHookeanParameters,
): StableNeoHookeanDensityEvaluation {
  if (deformationGradient.length !== MATRIX_SIZE) {
    throw new RangeError("A deformation gradient must be a 3 by 3 matrix.");
  }
  assertFiniteValues(deformationGradient, "Deformation gradient");
  const deformationDeterminant = determinant3(deformationGradient);
  let firstInvariant = 0;
  for (let index = 0; index < MATRIX_SIZE; index += 1) {
    firstInvariant += deformationGradient[index]! ** 2;
  }
  const invariantDenominator = firstInvariant + 1;
  const cofactor = cofactor3(deformationGradient);
  const shearScale = parameters.mu * (1 - 1 / invariantDenominator);
  const volumeScale =
    parameters.lambda * (deformationDeterminant - parameters.alpha);
  const firstPiola = new Float64Array(MATRIX_SIZE);
  for (let index = 0; index < MATRIX_SIZE; index += 1) {
    firstPiola[index] =
      shearScale * deformationGradient[index]! +
      volumeScale * cofactor[index]!;
  }
  const tangent = new Float64Array(MATRIX_SIZE * MATRIX_SIZE);
  const direction = new Float64Array(MATRIX_SIZE);
  for (let column = 0; column < MATRIX_SIZE; column += 1) {
    direction[column] = 1;
    const response = applyStableNeoHookeanTangent(
      deformationGradient,
      direction,
      parameters,
    );
    direction[column] = 0;
    for (let row = 0; row < MATRIX_SIZE; row += 1) {
      tangent[row * MATRIX_SIZE + column] = response[row]!;
    }
  }
  const determinantOffset = deformationDeterminant - 1;
  const energyDensity =
    0.5 * parameters.mu * (firstInvariant - 3) -
    0.75 * parameters.mu * determinantOffset +
    0.5 * parameters.lambda * determinantOffset * determinantOffset -
    0.5 * parameters.mu * Math.log(invariantDenominator / 4);
  assertFiniteValues(firstPiola, "First Piola stress");
  assertFiniteValues(tangent, "Stable Neo-Hookean tangent");
  if (!Number.isFinite(energyDensity)) {
    throw new Error("Stable Neo-Hookean energy is non-finite.");
  }
  return {
    energyDensity,
    firstPiola,
    tangent,
    deformationDeterminant,
    firstInvariant,
  };
}

function tetrahedronDeformationGradient(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  tetrahedron: number,
  positions: Float64Array,
): Float64Array {
  const tetrahedronOffset = tetrahedron * 4;
  const a = mesh.tetrahedra[tetrahedronOffset]!;
  const b = mesh.tetrahedra[tetrahedronOffset + 1]!;
  const c = mesh.tetrahedra[tetrahedronOffset + 2]!;
  const d = mesh.tetrahedra[tetrahedronOffset + 3]!;
  const deformedShape = new Float64Array([
    positions[b * 3]! - positions[a * 3]!,
    positions[c * 3]! - positions[a * 3]!,
    positions[d * 3]! - positions[a * 3]!,
    positions[b * 3 + 1]! - positions[a * 3 + 1]!,
    positions[c * 3 + 1]! - positions[a * 3 + 1]!,
    positions[d * 3 + 1]! - positions[a * 3 + 1]!,
    positions[b * 3 + 2]! - positions[a * 3 + 2]!,
    positions[c * 3 + 2]! - positions[a * 3 + 2]!,
    positions[d * 3 + 2]! - positions[a * 3 + 2]!,
  ]);
  return multiply3(
    deformedShape,
    restData.inverseRestMatrices.subarray(
      tetrahedron * MATRIX_SIZE,
      (tetrahedron + 1) * MATRIX_SIZE,
    ),
  );
}

/** Geometry-only det(F) for one tetrahedron; no material term is evaluated. */
export function tetrahedralDeformationDeterminant(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  tetrahedron: number,
  positions: Float64Array,
): number {
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (
    !Number.isSafeInteger(tetrahedron) ||
    tetrahedron < 0 ||
    tetrahedron >= tetrahedronCount
  ) {
    throw new RangeError("Deformation-determinant tetrahedron is out of range.");
  }
  if (positions.length !== getVertexCount(mesh) * 3) {
    throw new RangeError("Deformation-determinant positions have the wrong length.");
  }
  if (restData.inverseRestMatrices.length !== tetrahedronCount * MATRIX_SIZE) {
    throw new RangeError(
      "Deformation-determinant rest matrices do not match the mesh.",
    );
  }
  assertFiniteValues(positions, "Deformation-determinant positions");
  return determinant3(
    tetrahedronDeformationGradient(
      mesh,
      restData,
      tetrahedron,
      positions,
    ),
  );
}

/** Geometry-only feasibility oracle; it intentionally evaluates no material energy. */
export function minimumTetrahedralDeformationDeterminant(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  positions: Float64Array,
): number {
  const vertexCount = getVertexCount(mesh);
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (tetrahedronCount < 1) {
    throw new RangeError("A deformation-determinant query requires a tetrahedron.");
  }
  if (positions.length !== vertexCount * 3) {
    throw new RangeError("Deformation-determinant positions have the wrong length.");
  }
  if (restData.inverseRestMatrices.length !== tetrahedronCount * MATRIX_SIZE) {
    throw new RangeError(
      "Deformation-determinant rest matrices do not match the mesh.",
    );
  }
  assertFiniteValues(positions, "Deformation-determinant positions");

  let minimum = Number.POSITIVE_INFINITY;
  for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
    const determinant = tetrahedralDeformationDeterminant(
      mesh,
      restData,
      tetrahedron,
      positions,
    );
    // Do not let Math.min hide +Infinity behind another finite tetrahedron.
    if (!Number.isFinite(determinant)) return determinant;
    minimum = Math.min(minimum, determinant);
  }
  return minimum;
}

export function evaluateStableNeoHookeanTetrahedron(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  materials: readonly LinearMaterial[],
  tetrahedron: number,
  positions: Float64Array,
): StableNeoHookeanTetrahedronEvaluation {
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (!Number.isSafeInteger(tetrahedron) || tetrahedron < 0 || tetrahedron >= tetrahedronCount) {
    throw new RangeError("Stable Neo-Hookean tetrahedron index is out of range.");
  }
  if (positions.length !== getVertexCount(mesh) * 3) {
    throw new RangeError("Stable Neo-Hookean positions have the wrong length.");
  }
  assertFiniteValues(positions, "Stable Neo-Hookean positions");
  const material = materials[mesh.materialIds[tetrahedron] ?? -1];
  if (!material) {
    throw new RangeError(
      `Tetrahedron ${tetrahedron} references an unknown material.`,
    );
  }
  const parameters = computeStableNeoHookeanParameters(material);
  const deformationGradient = tetrahedronDeformationGradient(
    mesh,
    restData,
    tetrahedron,
    positions,
  );
  const density = evaluateStableNeoHookeanDensity(
    deformationGradient,
    parameters,
  );
  const volume = restData.volumes[tetrahedron]!;
  if (!(volume > 0) || !Number.isFinite(volume)) {
    throw new RangeError(`Tetrahedron ${tetrahedron} has invalid rest volume.`);
  }
  const gradients = computeTetrahedronShapeGradients(
    restData.inverseRestMatrices.subarray(
      tetrahedron * MATRIX_SIZE,
      (tetrahedron + 1) * MATRIX_SIZE,
    ),
  );
  const gradient = new Float64Array(TETRAHEDRON_DOF);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    for (let row = 0; row < 3; row += 1) {
      let value = 0;
      for (let column = 0; column < 3; column += 1) {
        value +=
          density.firstPiola[row * 3 + column]! *
          gradients[localVertex * 3 + column]!;
      }
      gradient[localVertex * 3 + row] = volume * value;
    }
  }
  const hessian = new Float64Array(TETRAHEDRON_DOF * TETRAHEDRON_DOF);
  const deformationDirection = new Float64Array(MATRIX_SIZE);
  for (let columnVertex = 0; columnVertex < 4; columnVertex += 1) {
    for (let columnCoordinate = 0; columnCoordinate < 3; columnCoordinate += 1) {
      for (let materialCoordinate = 0; materialCoordinate < 3; materialCoordinate += 1) {
        deformationDirection[columnCoordinate * 3 + materialCoordinate] =
          gradients[columnVertex * 3 + materialCoordinate]!;
      }
      const stressDirection = applyStableNeoHookeanTangent(
        deformationGradient,
        deformationDirection,
        parameters,
      );
      deformationDirection.fill(0);
      const localColumn = columnVertex * 3 + columnCoordinate;
      for (let rowVertex = 0; rowVertex < 4; rowVertex += 1) {
        for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
          let value = 0;
          for (let materialCoordinate = 0; materialCoordinate < 3; materialCoordinate += 1) {
            value +=
              stressDirection[rowCoordinate * 3 + materialCoordinate]! *
              gradients[rowVertex * 3 + materialCoordinate]!;
          }
          const localRow = rowVertex * 3 + rowCoordinate;
          hessian[localRow * TETRAHEDRON_DOF + localColumn] = volume * value;
        }
      }
    }
  }
  return {
    energy: volume * density.energyDensity,
    gradient,
    hessian,
    deformationGradient,
    deformationDeterminant: density.deformationDeterminant,
  };
}

/** Exact all-element Float64 material evaluation over full xyz coordinates. */
export function evaluateStableNeoHookeanMesh(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  materials: readonly LinearMaterial[],
  positions: Float64Array,
): StableNeoHookeanMeshEvaluation {
  const vertexCount = getVertexCount(mesh);
  const dimension = vertexCount * 3;
  if (positions.length !== dimension) {
    throw new RangeError("Stable Neo-Hookean positions have the wrong length.");
  }
  const tetrahedronCount = getTetrahedronCount(mesh);
  const gradient = new Float64Array(dimension);
  const hessian = new Float64Array(dimension * dimension);
  const tetrahedronEnergies = new Float64Array(tetrahedronCount);
  const deformationDeterminants = new Float64Array(tetrahedronCount);
  let energy = 0;

  for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
    const evaluation = evaluateStableNeoHookeanTetrahedron(
      mesh,
      restData,
      materials,
      tetrahedron,
      positions,
    );
    tetrahedronEnergies[tetrahedron] = evaluation.energy;
    deformationDeterminants[tetrahedron] = evaluation.deformationDeterminant;
    energy += evaluation.energy;
    for (let localRowVertex = 0; localRowVertex < 4; localRowVertex += 1) {
      const rowVertex = mesh.tetrahedra[tetrahedron * 4 + localRowVertex]!;
      for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
        const localRow = localRowVertex * 3 + rowCoordinate;
        const globalRow = rowVertex * 3 + rowCoordinate;
        gradient[globalRow] += evaluation.gradient[localRow]!;
        for (let localColumnVertex = 0; localColumnVertex < 4; localColumnVertex += 1) {
          const columnVertex =
            mesh.tetrahedra[tetrahedron * 4 + localColumnVertex]!;
          for (let columnCoordinate = 0; columnCoordinate < 3; columnCoordinate += 1) {
            const localColumn = localColumnVertex * 3 + columnCoordinate;
            const globalColumn = columnVertex * 3 + columnCoordinate;
            hessian[globalRow * dimension + globalColumn] +=
              evaluation.hessian[
                localRow * TETRAHEDRON_DOF + localColumn
              ]!;
          }
        }
      }
    }
  }
  return {
    energy,
    gradient,
    hessian,
    tetrahedronEnergies,
    deformationDeterminants,
  };
}

export interface StableNeoHookeanMaterialTermOptions {
  readonly id: string;
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly materials: readonly LinearMaterial[];
}

/** Bridge the nonlinear exact material evaluator into the shared energy API. */
export function createStableNeoHookeanMaterialTerm(
  options: StableNeoHookeanMaterialTermOptions,
): DifferentiableEnergyTerm {
  const dimension = getVertexCount(options.mesh) * 3;
  return {
    id: options.id,
    kind: "material",
    dimension,
    evaluate: (coordinates) =>
      evaluateStableNeoHookeanMesh(
        options.mesh,
        options.restData,
        options.materials,
        coordinates,
      ),
  };
}
