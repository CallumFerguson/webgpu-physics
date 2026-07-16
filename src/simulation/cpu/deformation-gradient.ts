import { computeTetrahedronShapeGradients } from "./fem";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import { polarRotation3 } from "./oracle";
import type { RestTetraData, TetrahedralMesh } from "./types";

export interface VertexDeformationGradientStencil {
  readonly vertexCount: number;
  /** One more entry than vertexCount; records for v are starts[v]..starts[v+1]. */
  readonly starts: Uint32Array;
  /** Source vertex for each sparse coefficient record. */
  readonly vertices: Uint32Array;
  /** Three material-coordinate coefficients per sparse record. */
  readonly coefficients: Float64Array;
}

/**
 * Build F_v = sum_q x_q c_vq^T from a rest-volume-weighted average of the
 * deformation gradients of tetrahedra incident to v. This preserves affine
 * fields exactly at interior and boundary vertices:
 * sum_q c_vq = 0 and sum_q X_q c_vq^T = I.
 */
export function buildVertexDeformationGradientStencil(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
): VertexDeformationGradientStencil {
  const vertexCount = getVertexCount(mesh);
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (
    restData.volumes.length !== tetrahedronCount ||
    restData.inverseRestMatrices.length !== tetrahedronCount * 9
  ) {
    throw new RangeError(
      "Rest tetrahedron data does not match the deformation-gradient mesh.",
    );
  }
  const incident: number[][] = Array.from({ length: vertexCount }, () => []);
  for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
    for (let localVertex = 0; localVertex < 4; localVertex += 1) {
      incident[mesh.tetrahedra[tetrahedron * 4 + localVertex]!]!.push(
        tetrahedron,
      );
    }
  }

  const starts = new Uint32Array(vertexCount + 1);
  const sourceVertices: number[] = [];
  const coefficientValues: number[] = [];
  for (let center = 0; center < vertexCount; center += 1) {
    const incidentTetrahedra = incident[center]!;
    if (incidentTetrahedra.length === 0) {
      throw new Error(
        `Vertex ${center} has no incident tetrahedron for its deformation frame.`,
      );
    }
    let totalVolume = 0;
    const coefficients = new Map<number, Float64Array>();
    for (const tetrahedron of incidentTetrahedra) {
      const volume = restData.volumes[tetrahedron]!;
      if (!(volume > 0) || !Number.isFinite(volume)) {
        throw new RangeError(`Tetrahedron ${tetrahedron} has invalid rest volume.`);
      }
      totalVolume += volume;
      const gradients = computeTetrahedronShapeGradients(
        restData.inverseRestMatrices.subarray(
          tetrahedron * 9,
          (tetrahedron + 1) * 9,
        ),
      );
      for (let localVertex = 0; localVertex < 4; localVertex += 1) {
        const source = mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
        let coefficient = coefficients.get(source);
        if (!coefficient) {
          coefficient = new Float64Array(3);
          coefficients.set(source, coefficient);
        }
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          coefficient[coordinate] +=
            volume * gradients[localVertex * 3 + coordinate]!;
        }
      }
    }
    for (const source of [...coefficients.keys()].sort((left, right) => left - right)) {
      const coefficient = coefficients.get(source)!;
      sourceVertices.push(source);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        coefficientValues.push(coefficient[coordinate]! / totalVolume);
      }
    }
    starts[center + 1] = sourceVertices.length;
  }

  return {
    vertexCount,
    starts,
    vertices: Uint32Array.from(sourceVertices),
    coefficients: Float64Array.from(coefficientValues),
  };
}

/** Evaluate one row-major F_v for every vertex from the sparse stencil. */
export function evaluateVertexDeformationGradients(
  stencil: VertexDeformationGradientStencil,
  positions: Float64Array,
): Float64Array {
  if (positions.length !== stencil.vertexCount * 3) {
    throw new RangeError("Vertex deformation-gradient positions have the wrong length.");
  }
  for (const value of positions) {
    if (!Number.isFinite(value)) {
      throw new RangeError(
        "Vertex deformation-gradient positions must be finite.",
      );
    }
  }
  if (
    stencil.starts.length !== stencil.vertexCount + 1 ||
    stencil.coefficients.length !== stencil.vertices.length * 3 ||
    stencil.starts[stencil.vertexCount] !== stencil.vertices.length
  ) {
    throw new RangeError("Vertex deformation-gradient stencil is malformed.");
  }
  const deformationGradients = new Float64Array(stencil.vertexCount * 9);
  for (let center = 0; center < stencil.vertexCount; center += 1) {
    const outputOffset = center * 9;
    for (
      let record = stencil.starts[center]!;
      record < stencil.starts[center + 1]!;
      record += 1
    ) {
      const source = stencil.vertices[record]!;
      if (source >= stencil.vertexCount) {
        throw new RangeError(
          "Vertex deformation-gradient stencil references an unknown vertex.",
        );
      }
      for (let worldCoordinate = 0; worldCoordinate < 3; worldCoordinate += 1) {
        const position = positions[source * 3 + worldCoordinate]!;
        for (
          let materialCoordinate = 0;
          materialCoordinate < 3;
          materialCoordinate += 1
        ) {
          deformationGradients[
            outputOffset + worldCoordinate * 3 + materialCoordinate
          ] += position * stencil.coefficients[record * 3 + materialCoordinate]!;
        }
      }
    }
  }
  return deformationGradients;
}

/** Polar frame for each affine-exact vertex deformation gradient. */
export function computeVertexPolarFrames(
  stencil: VertexDeformationGradientStencil,
  positions: Float64Array,
): Float64Array {
  const deformationGradients = evaluateVertexDeformationGradients(
    stencil,
    positions,
  );
  const rotations = new Float64Array(stencil.vertexCount * 9);
  for (let vertex = 0; vertex < stencil.vertexCount; vertex += 1) {
    rotations.set(
      polarRotation3(
        deformationGradients.subarray(vertex * 9, (vertex + 1) * 9),
      ),
      vertex * 9,
    );
  }
  return rotations;
}
