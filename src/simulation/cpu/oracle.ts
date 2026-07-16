import {
  choleskyFactor,
  determinant3,
  invert3,
  multiply3,
  solveCholesky,
  solveDenseLinearSystem,
} from "./math";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import type {
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";
import type {
  DenseEnergyEvaluation,
  DifferentiableEnergyModel,
} from "./energy";

const POLAR_EPSILON = 1e-14;
const POLAR_ITERATIONS = 20;

export interface EnergyEvaluation extends DenseEnergyEvaluation {}

export interface CorotatedLinearElementEvaluation extends EnergyEvaluation {}

export interface ImplicitEulerEnergyComponents {
  readonly inertia: number;
  readonly elasticity: number;
  readonly floorContact: number;
}

export interface ImplicitEulerEvaluation extends EnergyEvaluation {
  readonly components: ImplicitEulerEnergyComponents;
  /** Exact, unweighted energy from every tetrahedron in mesh order. */
  readonly tetrahedronEnergies: Float64Array;
}

export interface QuadraticFloorContact {
  readonly height: number;
  readonly stiffness: number;
}

export interface CorotatedLinearImplicitEulerOptions {
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
  readonly timestep: number;
  /** Full xyz positions used as the implicit Euler inertial target. */
  readonly predictedPositions: Float64Array;
  /**
   * Full xyz pose at which element polar frames are frozen. Defaults to the
   * rest pose. Runtime JGS2 likewise freezes these frames during one solve.
   */
  readonly rotationPositions?: Float64Array;
  readonly floorContact?: QuadraticFloorContact;
}

/**
 * Exact Float64 oracle for one frozen-frame co-rotated implicit Euler solve.
 * Coordinates contain only the active degrees of freedom in restSystem order.
 */
export interface CorotatedLinearImplicitEulerOracle
  extends DifferentiableEnergyModel<ImplicitEulerEvaluation> {
  readonly rotations: Float64Array;
}

export interface ComplementaryEquilibriumBasis {
  /** Dense active-coordinate basis, with dimension rows and three columns. */
  readonly basis: Float64Array;
  /** B^T H B, equivalently the local Schur complement. */
  readonly schurComplement: Float64Array;
}

function validateFullPositions(
  mesh: TetrahedralMesh,
  positions: Float64Array,
  label: string,
): void {
  if (positions.length !== getVertexCount(mesh) * 3) {
    throw new RangeError(`${label} must contain xyz for every mesh vertex.`);
  }
  for (const value of positions) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${label} must contain only finite coordinates.`);
    }
  }
}

function validateRotation(rotation: ArrayLike<number>): void {
  if (rotation.length !== 9) {
    throw new RangeError("A tetrahedron rotation must be a 3 by 3 matrix.");
  }
  for (let index = 0; index < rotation.length; index += 1) {
    if (!Number.isFinite(rotation[index]!)) {
      throw new RangeError("A tetrahedron rotation must be finite.");
    }
  }
}

function dot3(left: ArrayLike<number>, right: ArrayLike<number>): number {
  return left[0]! * right[0]! + left[1]! * right[1]! + left[2]! * right[2]!;
}

function cross3(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
): Float64Array {
  return new Float64Array([
    left[1]! * right[2]! - left[2]! * right[1]!,
    left[2]! * right[0]! - left[0]! * right[2]!,
    left[0]! * right[1]! - left[1]! * right[0]!,
  ]);
}

function normalized3(
  vector: ArrayLike<number>,
  fallback: readonly [number, number, number],
): Float64Array {
  const norm = Math.sqrt(dot3(vector, vector));
  if (!(norm > POLAR_EPSILON) || !Number.isFinite(norm)) {
    return new Float64Array(fallback);
  }
  return new Float64Array([
    vector[0]! / norm,
    vector[1]! / norm,
    vector[2]! / norm,
  ]);
}

function perpendicularTo(axis: ArrayLike<number>): Float64Array {
  const candidate =
    Math.abs(axis[0]!) > 0.8
      ? new Float64Array([0, 1, 0])
      : new Float64Array([1, 0, 0]);
  return normalized3(cross3(axis, candidate), [0, 0, 1]);
}

/** Match the right-handed column orthonormalization used by the WGSL path. */
function orthonormalizeRotation(matrix: Float64Array): Float64Array {
  const inputColumn0 = new Float64Array([matrix[0]!, matrix[3]!, matrix[6]!]);
  const inputColumn1 = new Float64Array([matrix[1]!, matrix[4]!, matrix[7]!]);
  const inputColumn2 = new Float64Array([matrix[2]!, matrix[5]!, matrix[8]!]);
  const column0 = normalized3(inputColumn0, [1, 0, 0]);
  const projection = dot3(column0, inputColumn1);
  const column1Candidate = new Float64Array([
    inputColumn1[0]! - projection * column0[0]!,
    inputColumn1[1]! - projection * column0[1]!,
    inputColumn1[2]! - projection * column0[2]!,
  ]);
  let column1 = normalized3(column1Candidate, [
    perpendicularTo(column0)[0]!,
    perpendicularTo(column0)[1]!,
    perpendicularTo(column0)[2]!,
  ]);
  let column2 = normalized3(cross3(column0, column1), [0, 0, 1]);
  if (dot3(column2, inputColumn2) < 0) {
    column1 = new Float64Array([-column1[0]!, -column1[1]!, -column1[2]!]);
    column2 = new Float64Array([-column2[0]!, -column2[1]!, -column2[2]!]);
  }
  column1 = normalized3(cross3(column2, column0), [
    perpendicularTo(column0)[0]!,
    perpendicularTo(column0)[1]!,
    perpendicularTo(column0)[2]!,
  ]);

  return new Float64Array([
    column0[0]!,
    column1[0]!,
    column2[0]!,
    column0[1]!,
    column1[1]!,
    column2[1]!,
    column0[2]!,
    column1[2]!,
    column2[2]!,
  ]);
}

/** Float64 polar rotation with the same Newton iteration as the GPU solver. */
export function polarRotation3(matrix: ArrayLike<number>): Float64Array {
  if (matrix.length !== 9) {
    throw new RangeError("Polar decomposition requires a 3 by 3 matrix.");
  }
  let rotation = Float64Array.from(matrix);

  for (let iteration = 0; iteration < POLAR_ITERATIONS; iteration += 1) {
    const determinant = determinant3(rotation);
    if (
      !(Math.abs(determinant) > POLAR_EPSILON) ||
      !Number.isFinite(determinant)
    ) {
      break;
    }
    const inverse = invert3(rotation);
    const next = new Float64Array(9);
    let maximumChange = 0;
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        const index = row * 3 + column;
        next[index] =
          0.5 * (rotation[index]! + inverse[column * 3 + row]!);
        maximumChange = Math.max(
          maximumChange,
          Math.abs(next[index]! - rotation[index]!),
        );
      }
    }
    rotation = next;
    if (maximumChange <= 1e-14) {
      break;
    }
  }

  return orthonormalizeRotation(rotation);
}

/** Compute one row-major polar rotation per tetrahedron from a full pose. */
export function computeCorotatedLinearRotations(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  positions: Float64Array,
): Float64Array {
  validateFullPositions(mesh, positions, "Rotation positions");
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (restData.inverseRestMatrices.length !== tetrahedronCount * 9) {
    throw new RangeError("Inverse rest matrices do not match the mesh.");
  }
  const rotations = new Float64Array(tetrahedronCount * 9);

  for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
    const vertex0 = mesh.tetrahedra[tetrahedron * 4]!;
    const vertex1 = mesh.tetrahedra[tetrahedron * 4 + 1]!;
    const vertex2 = mesh.tetrahedra[tetrahedron * 4 + 2]!;
    const vertex3 = mesh.tetrahedra[tetrahedron * 4 + 3]!;
    const x0 = vertex0 * 3;
    const deformedShape = new Float64Array([
      positions[vertex1 * 3]! - positions[x0]!,
      positions[vertex2 * 3]! - positions[x0]!,
      positions[vertex3 * 3]! - positions[x0]!,
      positions[vertex1 * 3 + 1]! - positions[x0 + 1]!,
      positions[vertex2 * 3 + 1]! - positions[x0 + 1]!,
      positions[vertex3 * 3 + 1]! - positions[x0 + 1]!,
      positions[vertex1 * 3 + 2]! - positions[x0 + 2]!,
      positions[vertex2 * 3 + 2]! - positions[x0 + 2]!,
      positions[vertex3 * 3 + 2]! - positions[x0 + 2]!,
    ]);
    const inverseRest = restData.inverseRestMatrices.subarray(
      tetrahedron * 9,
      tetrahedron * 9 + 9,
    );
    rotations.set(
      polarRotation3(multiply3(deformedShape, inverseRest)),
      tetrahedron * 9,
    );
  }

  return rotations;
}

/**
 * Evaluate one exact co-rotated linear tetrahedron with a frozen polar frame.
 * The returned gradient and Hessian use the tetrahedron's 12 local xyz DOFs.
 */
export function evaluateCorotatedLinearTetrahedron(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  tetrahedron: number,
  positions: Float64Array,
  rotation: ArrayLike<number>,
): CorotatedLinearElementEvaluation {
  validateFullPositions(mesh, positions, "Element positions");
  validateRotation(rotation);
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (!Number.isInteger(tetrahedron) || tetrahedron < 0 || tetrahedron >= tetrahedronCount) {
    throw new RangeError("The tetrahedron index is outside the mesh.");
  }
  if (restData.stiffnessMatrices.length !== tetrahedronCount * 144) {
    throw new RangeError("Rest stiffness matrices do not match the mesh.");
  }

  const displacement = new Float64Array(12);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    for (let localCoordinate = 0; localCoordinate < 3; localCoordinate += 1) {
      let rotatedPosition = 0;
      for (let worldCoordinate = 0; worldCoordinate < 3; worldCoordinate += 1) {
        rotatedPosition +=
          rotation[worldCoordinate * 3 + localCoordinate]! *
          positions[vertex * 3 + worldCoordinate]!;
      }
      displacement[localVertex * 3 + localCoordinate] =
        rotatedPosition - mesh.positions[vertex * 3 + localCoordinate]!;
    }
  }

  const stiffnessOffset = tetrahedron * 144;
  const localGradient = new Float64Array(12);
  for (let row = 0; row < 12; row += 1) {
    for (let column = 0; column < 12; column += 1) {
      localGradient[row] +=
        restData.stiffnessMatrices[
          stiffnessOffset + row * 12 + column
        ]! * displacement[column]!;
    }
  }

  let energy = 0;
  for (let row = 0; row < 12; row += 1) {
    energy += 0.5 * displacement[row]! * localGradient[row]!;
  }

  const gradient = new Float64Array(12);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    for (let worldCoordinate = 0; worldCoordinate < 3; worldCoordinate += 1) {
      for (let localCoordinate = 0; localCoordinate < 3; localCoordinate += 1) {
        gradient[localVertex * 3 + worldCoordinate] +=
          rotation[worldCoordinate * 3 + localCoordinate]! *
          localGradient[localVertex * 3 + localCoordinate]!;
      }
    }
  }

  const hessian = new Float64Array(144);
  for (let rowVertex = 0; rowVertex < 4; rowVertex += 1) {
    for (let columnVertex = 0; columnVertex < 4; columnVertex += 1) {
      for (let worldRow = 0; worldRow < 3; worldRow += 1) {
        for (let worldColumn = 0; worldColumn < 3; worldColumn += 1) {
          let value = 0;
          for (let localRow = 0; localRow < 3; localRow += 1) {
            for (let localColumn = 0; localColumn < 3; localColumn += 1) {
              value +=
                rotation[worldRow * 3 + localRow]! *
                restData.stiffnessMatrices[
                  stiffnessOffset +
                    (rowVertex * 3 + localRow) * 12 +
                    columnVertex * 3 +
                    localColumn
                ]! *
                rotation[worldColumn * 3 + localColumn]!;
            }
          }
          hessian[
            (rowVertex * 3 + worldRow) * 12 +
              columnVertex * 3 +
              worldColumn
          ] = value;
        }
      }
    }
  }

  return { energy, gradient, hessian };
}

export function activeCoordinatesFromFullPositions(
  positions: Float64Array,
  system: RestLinearSystem,
): Float64Array {
  const result = new Float64Array(system.dimension);
  for (const vertex of system.activeVertices) {
    const activeBase = system.vertexToActiveDof[vertex]!;
    result.set(positions.subarray(vertex * 3, vertex * 3 + 3), activeBase);
  }
  return result;
}

export function fullPositionsFromActiveCoordinates(
  coordinates: Float64Array,
  mesh: TetrahedralMesh,
  system: RestLinearSystem,
): Float64Array {
  if (coordinates.length !== system.dimension) {
    throw new RangeError("Active coordinates do not match the linear system.");
  }
  const result = mesh.positions.slice();
  for (const vertex of system.activeVertices) {
    const activeBase = system.vertexToActiveDof[vertex]!;
    result.set(coordinates.subarray(activeBase, activeBase + 3), vertex * 3);
  }
  return result;
}

export function createCorotatedLinearImplicitEulerOracle(
  options: CorotatedLinearImplicitEulerOptions,
): CorotatedLinearImplicitEulerOracle {
  const {
    mesh,
    restData,
    lumpedMasses,
    restSystem,
    timestep,
    predictedPositions,
    floorContact,
  } = options;
  const vertexCount = getVertexCount(mesh);
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (!(timestep > 0) || !Number.isFinite(timestep)) {
    throw new RangeError("The oracle timestep must be finite and positive.");
  }
  if (lumpedMasses.length !== vertexCount) {
    throw new RangeError("The oracle requires one lumped mass per vertex.");
  }
  if (restSystem.vertexToActiveDof.length !== vertexCount) {
    throw new RangeError("The oracle linear system does not match the mesh.");
  }
  validateFullPositions(mesh, predictedPositions, "Predicted positions");
  const rotationPositions = options.rotationPositions ?? mesh.positions;
  validateFullPositions(mesh, rotationPositions, "Rotation positions");
  if (floorContact) {
    if (!Number.isFinite(floorContact.height)) {
      throw new RangeError("The floor height must be finite.");
    }
    if (!(floorContact.stiffness >= 0) || !Number.isFinite(floorContact.stiffness)) {
      throw new RangeError("The floor stiffness must be finite and nonnegative.");
    }
  }
  const rotations = computeCorotatedLinearRotations(
    mesh,
    restData,
    rotationPositions,
  );
  const inverseTimestepSquared = 1 / (timestep * timestep);

  const evaluate = (coordinates: Float64Array): ImplicitEulerEvaluation => {
    const positions = fullPositionsFromActiveCoordinates(
      coordinates,
      mesh,
      restSystem,
    );
    const gradient = new Float64Array(restSystem.dimension);
    const hessian = new Float64Array(
      restSystem.dimension * restSystem.dimension,
    );
    const tetrahedronEnergies = new Float64Array(tetrahedronCount);
    let inertiaEnergy = 0;
    let elasticityEnergy = 0;
    let floorContactEnergy = 0;

    for (const vertex of restSystem.activeVertices) {
      const activeBase = restSystem.vertexToActiveDof[vertex]!;
      const inertia = lumpedMasses[vertex]! * inverseTimestepSquared;
      if (!(inertia > 0) || !Number.isFinite(inertia)) {
        throw new Error(`Movable vertex ${vertex} has invalid inertia.`);
      }
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const activeCoordinate = activeBase + coordinate;
        const difference =
          positions[vertex * 3 + coordinate]! -
          predictedPositions[vertex * 3 + coordinate]!;
        inertiaEnergy += 0.5 * inertia * difference * difference;
        gradient[activeCoordinate] += inertia * difference;
        hessian[
          activeCoordinate * restSystem.dimension + activeCoordinate
        ] += inertia;
      }
    }

    for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
      const element = evaluateCorotatedLinearTetrahedron(
        mesh,
        restData,
        tetrahedron,
        positions,
        rotations.subarray(tetrahedron * 9, tetrahedron * 9 + 9),
      );
      tetrahedronEnergies[tetrahedron] = element.energy;
      elasticityEnergy += element.energy;
      for (let localRowVertex = 0; localRowVertex < 4; localRowVertex += 1) {
        const rowVertex = mesh.tetrahedra[tetrahedron * 4 + localRowVertex]!;
        const rowBase = restSystem.vertexToActiveDof[rowVertex]!;
        if (rowBase < 0) {
          continue;
        }
        for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
          gradient[rowBase + rowCoordinate] +=
            element.gradient[localRowVertex * 3 + rowCoordinate]!;
        }
        for (
          let localColumnVertex = 0;
          localColumnVertex < 4;
          localColumnVertex += 1
        ) {
          const columnVertex =
            mesh.tetrahedra[tetrahedron * 4 + localColumnVertex]!;
          const columnBase = restSystem.vertexToActiveDof[columnVertex]!;
          if (columnBase < 0) {
            continue;
          }
          for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
            for (
              let columnCoordinate = 0;
              columnCoordinate < 3;
              columnCoordinate += 1
            ) {
              hessian[
                (rowBase + rowCoordinate) * restSystem.dimension +
                  columnBase +
                  columnCoordinate
              ] +=
                element.hessian[
                  (localRowVertex * 3 + rowCoordinate) * 12 +
                    localColumnVertex * 3 +
                    columnCoordinate
                ]!;
            }
          }
        }
      }
    }

    if (floorContact && floorContact.stiffness > 0) {
      for (const vertex of restSystem.activeVertices) {
        const penetration = positions[vertex * 3 + 1]! - floorContact.height;
        if (penetration >= 0) {
          continue;
        }
        const activeY = restSystem.vertexToActiveDof[vertex]! + 1;
        floorContactEnergy +=
          0.5 * floorContact.stiffness * penetration * penetration;
        gradient[activeY] += floorContact.stiffness * penetration;
        hessian[activeY * restSystem.dimension + activeY] +=
          floorContact.stiffness;
      }
    }

    const components: ImplicitEulerEnergyComponents = {
      inertia: inertiaEnergy,
      elasticity: elasticityEnergy,
      floorContact: floorContactEnergy,
    };
    return {
      energy: inertiaEnergy + elasticityEnergy + floorContactEnergy,
      gradient,
      hessian,
      components,
      tetrahedronEnergies,
    };
  };

  return {
    dimension: restSystem.dimension,
    rotations,
    evaluate,
    energy: (coordinates) => evaluate(coordinates).energy,
    gradient: (coordinates) => evaluate(coordinates).gradient,
    hessian: (coordinates) => evaluate(coordinates).hessian,
  };
}

/** Exact dense Newton step, H delta = -g, for a positive-definite oracle. */
export function solveFullNewtonStep(
  evaluation: Pick<EnergyEvaluation, "gradient" | "hessian">,
): Float64Array {
  const dimension = evaluation.gradient.length;
  if (evaluation.hessian.length !== dimension * dimension) {
    throw new RangeError("The Newton Hessian dimensions do not match its gradient.");
  }
  const lower = choleskyFactor(evaluation.hessian, dimension);
  const step = solveCholesky(lower, evaluation.gradient, dimension);
  for (let coordinate = 0; coordinate < dimension; coordinate += 1) {
    step[coordinate] *= -1;
  }
  return step;
}

/**
 * Direct equations 7-13 solve after explicitly partitioning local and
 * complementary coordinates: U_C = -H_CC^-1 H_Ci and U_i = I.
 */
export function computeDirectComplementaryEquilibriumBasis(
  hessian: Float64Array,
  dimension: number,
  localBase: number,
): ComplementaryEquilibriumBasis {
  if (hessian.length !== dimension * dimension) {
    throw new RangeError("The equilibrium-basis Hessian dimensions do not match.");
  }
  if (
    !Number.isInteger(localBase) ||
    localBase < 0 ||
    localBase + 3 > dimension
  ) {
    throw new RangeError("The local equilibrium block must contain three DOFs.");
  }

  const complementary = Array.from(
    { length: dimension - 3 },
    (_unused, index) => (index < localBase ? index : index + 3),
  );
  const basis = new Float64Array(dimension * 3);
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    basis[(localBase + coordinate) * 3 + coordinate] = 1;
  }

  if (complementary.length > 0) {
    const complementaryHessian = new Float64Array(
      complementary.length * complementary.length,
    );
    for (let row = 0; row < complementary.length; row += 1) {
      for (let column = 0; column < complementary.length; column += 1) {
        complementaryHessian[row * complementary.length + column] =
          hessian[complementary[row]! * dimension + complementary[column]!]!;
      }
    }

    for (let localColumn = 0; localColumn < 3; localColumn += 1) {
      const rightHandSide = new Float64Array(complementary.length);
      for (let row = 0; row < complementary.length; row += 1) {
        rightHandSide[row] =
          -hessian[
            complementary[row]! * dimension + localBase + localColumn
          ]!;
      }
      const solution = solveDenseLinearSystem(
        complementaryHessian,
        rightHandSide,
        complementary.length,
      );
      if (!solution) {
        throw new Error("The complementary Hessian is singular.");
      }
      for (let row = 0; row < complementary.length; row += 1) {
        basis[complementary[row]! * 3 + localColumn] = solution[row]!;
      }
    }
  }

  const hessianTimesBasis = new Float64Array(dimension * 3);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let inner = 0; inner < dimension; inner += 1) {
        hessianTimesBasis[row * 3 + column] +=
          hessian[row * dimension + inner]! * basis[inner * 3 + column]!;
      }
    }
  }
  const schurComplement = new Float64Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let inner = 0; inner < dimension; inner += 1) {
        schurComplement[row * 3 + column] +=
          basis[inner * 3 + row]! *
          hessianTimesBasis[inner * 3 + column]!;
      }
    }
  }

  return { basis, schurComplement };
}

/** Solve the exact three-coordinate Newton system induced by a dense basis. */
export function solveEquilibriumBasisNewtonStep(
  evaluation: Pick<EnergyEvaluation, "gradient" | "hessian">,
  basis: Float64Array,
): Float64Array {
  const dimension = evaluation.gradient.length;
  if (
    evaluation.hessian.length !== dimension * dimension ||
    basis.length !== dimension * 3
  ) {
    throw new RangeError("The equilibrium Newton dimensions do not match.");
  }
  const reducedGradient = new Float64Array(3);
  const hessianTimesBasis = new Float64Array(dimension * 3);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      reducedGradient[column] +=
        basis[row * 3 + column]! * evaluation.gradient[row]!;
      for (let inner = 0; inner < dimension; inner += 1) {
        hessianTimesBasis[row * 3 + column] +=
          evaluation.hessian[row * dimension + inner]! *
          basis[inner * 3 + column]!;
      }
    }
  }
  const reducedHessian = new Float64Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      for (let inner = 0; inner < dimension; inner += 1) {
        reducedHessian[row * 3 + column] +=
          basis[inner * 3 + row]! *
          hessianTimesBasis[inner * 3 + column]!;
      }
    }
  }
  const lower = choleskyFactor(reducedHessian, 3);
  const step = solveCholesky(lower, reducedGradient, 3);
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    step[coordinate] *= -1;
  }
  return step;
}
