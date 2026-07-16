import { choleskyFactor, determinant3, invert3 } from "./math";
import {
  getTetrahedronCount,
  getVertexCount,
  validateTetrahedralMesh,
} from "./mesh";
import type {
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

function validateMaterial(material: LinearMaterial): void {
  if (!(material.density > 0) || !Number.isFinite(material.density)) {
    throw new RangeError(`Material ${material.name} has invalid density.`);
  }
  if (
    !(material.youngModulus > 0) ||
    !Number.isFinite(material.youngModulus)
  ) {
    throw new RangeError(
      `Material ${material.name} has invalid Young's modulus.`,
    );
  }
  if (
    !(material.poissonRatio > -1 && material.poissonRatio < 0.5) ||
    !Number.isFinite(material.poissonRatio)
  ) {
    throw new RangeError(`Material ${material.name} has invalid Poisson ratio.`);
  }
}

function restMatrix(
  positions: Float64Array,
  vertices: readonly [number, number, number, number],
): Float64Array {
  const [a, b, c, d] = vertices;
  const ax = positions[a * 3]!;
  const ay = positions[a * 3 + 1]!;
  const az = positions[a * 3 + 2]!;

  return new Float64Array([
    positions[b * 3]! - ax,
    positions[c * 3]! - ax,
    positions[d * 3]! - ax,
    positions[b * 3 + 1]! - ay,
    positions[c * 3 + 1]! - ay,
    positions[d * 3 + 1]! - ay,
    positions[b * 3 + 2]! - az,
    positions[c * 3 + 2]! - az,
    positions[d * 3 + 2]! - az,
  ]);
}

export interface TetrahedronStiffnessResult {
  readonly volume: number;
  readonly inverseRestMatrix: Float64Array;
  readonly stiffness: Float64Array;
}

/** Four row-major material-space gradients for linear tetrahedron shapes. */
export function computeTetrahedronShapeGradients(
  inverseRestMatrix: ArrayLike<number>,
): Float64Array {
  if (inverseRestMatrix.length !== 9) {
    throw new RangeError("A tetrahedron inverse rest matrix must be 3 by 3.");
  }
  const gradients = new Float64Array(12);
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    gradients[3 + coordinate] = inverseRestMatrix[coordinate]!;
    gradients[6 + coordinate] = inverseRestMatrix[3 + coordinate]!;
    gradients[9 + coordinate] = inverseRestMatrix[6 + coordinate]!;
    gradients[coordinate] =
      -gradients[3 + coordinate]! -
      gradients[6 + coordinate]! -
      gradients[9 + coordinate]!;
  }
  return gradients;
}

export function computeLinearTetrahedronStiffness(
  positions: Float64Array,
  vertices: readonly [number, number, number, number],
  material: LinearMaterial,
): TetrahedronStiffnessResult {
  validateMaterial(material);
  const matrix = restMatrix(positions, vertices);
  const signedSixVolume = determinant3(matrix);
  const volume = Math.abs(signedSixVolume) / 6;
  if (!(volume > 1e-14) || !Number.isFinite(volume)) {
    throw new Error("Cannot assemble a degenerate tetrahedron.");
  }
  const inverseRestMatrix = invert3(matrix);
  const gradients = computeTetrahedronShapeGradients(inverseRestMatrix);

  const strainDisplacement = new Float64Array(6 * 12);
  for (let vertex = 0; vertex < 4; vertex += 1) {
    const gradientX = gradients[vertex * 3]!;
    const gradientY = gradients[vertex * 3 + 1]!;
    const gradientZ = gradients[vertex * 3 + 2]!;
    const column = vertex * 3;
    strainDisplacement[column] = gradientX;
    strainDisplacement[12 + column + 1] = gradientY;
    strainDisplacement[24 + column + 2] = gradientZ;
    strainDisplacement[36 + column] = gradientY;
    strainDisplacement[36 + column + 1] = gradientX;
    strainDisplacement[48 + column + 1] = gradientZ;
    strainDisplacement[48 + column + 2] = gradientY;
    strainDisplacement[60 + column] = gradientZ;
    strainDisplacement[60 + column + 2] = gradientX;
  }

  const young = material.youngModulus;
  const poisson = material.poissonRatio;
  const lambda = (young * poisson) / ((1 + poisson) * (1 - 2 * poisson));
  const mu = young / (2 * (1 + poisson));
  const constitutive = new Float64Array(36);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      constitutive[row * 6 + column] =
        row === column ? lambda + 2 * mu : lambda;
    }
  }
  constitutive[3 * 6 + 3] = mu;
  constitutive[4 * 6 + 4] = mu;
  constitutive[5 * 6 + 5] = mu;

  const constitutiveTimesB = new Float64Array(6 * 12);
  for (let row = 0; row < 6; row += 1) {
    for (let column = 0; column < 12; column += 1) {
      let value = 0;
      for (let inner = 0; inner < 6; inner += 1) {
        value +=
          constitutive[row * 6 + inner]! *
          strainDisplacement[inner * 12 + column]!;
      }
      constitutiveTimesB[row * 12 + column] = value;
    }
  }

  const stiffness = new Float64Array(12 * 12);
  for (let row = 0; row < 12; row += 1) {
    for (let column = row; column < 12; column += 1) {
      let value = 0;
      for (let inner = 0; inner < 6; inner += 1) {
        value +=
          strainDisplacement[inner * 12 + row]! *
          constitutiveTimesB[inner * 12 + column]!;
      }
      value *= volume;
      stiffness[row * 12 + column] = value;
      stiffness[column * 12 + row] = value;
    }
  }

  return { volume, inverseRestMatrix, stiffness };
}

export function computeRestTetraData(
  mesh: TetrahedralMesh,
  materials: readonly LinearMaterial[],
): RestTetraData {
  validateTetrahedralMesh(mesh);
  if (materials.length === 0) {
    throw new RangeError("At least one material is required.");
  }
  for (const material of materials) {
    validateMaterial(material);
  }

  const tetrahedronCount = getTetrahedronCount(mesh);
  const volumes = new Float64Array(tetrahedronCount);
  const inverseRestMatrices = new Float64Array(tetrahedronCount * 9);
  const stiffnessMatrices = new Float64Array(tetrahedronCount * 144);

  for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
    const offset = tetrahedron * 4;
    const vertices: readonly [number, number, number, number] = [
      mesh.tetrahedra[offset]!,
      mesh.tetrahedra[offset + 1]!,
      mesh.tetrahedra[offset + 2]!,
      mesh.tetrahedra[offset + 3]!,
    ];
    const material = materials[mesh.materialIds[tetrahedron] ?? -1];
    if (!material) {
      throw new RangeError(
        `Tetrahedron ${tetrahedron} references an unknown material.`,
      );
    }
    const result = computeLinearTetrahedronStiffness(
      mesh.positions,
      vertices,
      material,
    );
    volumes[tetrahedron] = result.volume;
    inverseRestMatrices.set(result.inverseRestMatrix, tetrahedron * 9);
    stiffnessMatrices.set(result.stiffness, tetrahedron * 144);
  }

  return { volumes, inverseRestMatrices, stiffnessMatrices };
}

export function computeLumpedMasses(
  mesh: TetrahedralMesh,
  materials: readonly LinearMaterial[],
  restData: RestTetraData,
): Float64Array {
  const masses = new Float64Array(getVertexCount(mesh));

  for (
    let tetrahedron = 0;
    tetrahedron < getTetrahedronCount(mesh);
    tetrahedron += 1
  ) {
    const material = materials[mesh.materialIds[tetrahedron] ?? -1];
    if (!material) {
      throw new RangeError(
        `Tetrahedron ${tetrahedron} references an unknown material.`,
      );
    }
    const vertexMass =
      (material.density * restData.volumes[tetrahedron]!) / 4;
    for (let localVertex = 0; localVertex < 4; localVertex += 1) {
      const vertex = mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
      masses[vertex] += vertexMass;
    }
  }

  return masses;
}

export function assembleRestLinearSystem(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  masses: Float64Array,
  timestep: number,
): RestLinearSystem {
  if (!(timestep > 0) || !Number.isFinite(timestep)) {
    throw new RangeError("The timestep must be finite and positive.");
  }
  const vertexCount = getVertexCount(mesh);
  if (masses.length !== vertexCount) {
    throw new RangeError("Lumped masses must have one entry per vertex.");
  }

  const vertexToActiveDof = new Int32Array(vertexCount);
  vertexToActiveDof.fill(-1);
  const activeVertexList: number[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (mesh.fixed[vertex] === 0) {
      vertexToActiveDof[vertex] = activeVertexList.length * 3;
      activeVertexList.push(vertex);
    }
  }
  if (activeVertexList.length === 0) {
    throw new Error("A simulation scene must have at least one movable vertex.");
  }

  const activeVertices = Uint32Array.from(activeVertexList);
  const dimension = activeVertices.length * 3;
  const hessian = new Float64Array(dimension * dimension);

  for (
    let tetrahedron = 0;
    tetrahedron < getTetrahedronCount(mesh);
    tetrahedron += 1
  ) {
    const stiffnessOffset = tetrahedron * 144;
    for (let localRowVertex = 0; localRowVertex < 4; localRowVertex += 1) {
      const rowVertex = mesh.tetrahedra[tetrahedron * 4 + localRowVertex]!;
      const rowBase = vertexToActiveDof[rowVertex]!;
      if (rowBase < 0) {
        continue;
      }
      for (
        let localColumnVertex = 0;
        localColumnVertex < 4;
        localColumnVertex += 1
      ) {
        const columnVertex =
          mesh.tetrahedra[tetrahedron * 4 + localColumnVertex]!;
        const columnBase = vertexToActiveDof[columnVertex]!;
        if (columnBase < 0) {
          continue;
        }
        for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
          for (
            let columnCoordinate = 0;
            columnCoordinate < 3;
            columnCoordinate += 1
          ) {
            const localRow = localRowVertex * 3 + rowCoordinate;
            const localColumn = localColumnVertex * 3 + columnCoordinate;
            hessian[
              (rowBase + rowCoordinate) * dimension +
                columnBase +
                columnCoordinate
            ] +=
              restData.stiffnessMatrices[
                stiffnessOffset + localRow * 12 + localColumn
              ]!;
          }
        }
      }
    }
  }

  const inverseTimestepSquared = 1 / (timestep * timestep);
  for (const vertex of activeVertices) {
    const base = vertexToActiveDof[vertex]!;
    const inertia = masses[vertex]! * inverseTimestepSquared;
    if (!(inertia > 0) || !Number.isFinite(inertia)) {
      throw new Error(`Movable vertex ${vertex} has invalid lumped mass.`);
    }
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      hessian[(base + coordinate) * dimension + base + coordinate] += inertia;
    }
  }

  return {
    dimension,
    activeVertices,
    vertexToActiveDof,
    hessian,
    choleskyLower: choleskyFactor(hessian, dimension),
  };
}
