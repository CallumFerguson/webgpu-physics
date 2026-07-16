import { determinant3 } from "./math";
import type {
  MeshTransform,
  RegularCuboidOptions,
  SurfaceTopology,
  TetrahedralMesh,
  Vec3,
} from "./types";

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer.`);
  }
}

export function getVertexCount(mesh: TetrahedralMesh): number {
  return mesh.positions.length / 3;
}

export function getTetrahedronCount(mesh: TetrahedralMesh): number {
  return mesh.tetrahedra.length / 4;
}

export function validateTetrahedralMesh(mesh: TetrahedralMesh): void {
  if (mesh.positions.length % 3 !== 0) {
    throw new Error("Mesh positions must contain complete xyz triples.");
  }
  if (mesh.tetrahedra.length % 4 !== 0) {
    throw new Error("Mesh indices must contain complete tetrahedra.");
  }

  const vertexCount = getVertexCount(mesh);
  const tetrahedronCount = getTetrahedronCount(mesh);
  if (mesh.fixed.length !== vertexCount) {
    throw new Error("The fixed mask must have one entry per vertex.");
  }
  if (mesh.bodyIds.length !== vertexCount) {
    throw new Error("Body IDs must have one entry per vertex.");
  }
  if (mesh.materialIds.length !== tetrahedronCount) {
    throw new Error("Material IDs must have one entry per tetrahedron.");
  }

  for (const coordinate of mesh.positions) {
    if (!Number.isFinite(coordinate)) {
      throw new Error("Mesh positions must be finite.");
    }
  }
  for (const vertex of mesh.tetrahedra) {
    if (vertex >= vertexCount) {
      throw new RangeError(`Tetrahedron vertex ${vertex} is out of range.`);
    }
  }
}

function tetrahedronDeterminant(
  positions: Float64Array,
  a: number,
  b: number,
  c: number,
  d: number,
): number {
  const ax = positions[a * 3]!;
  const ay = positions[a * 3 + 1]!;
  const az = positions[a * 3 + 2]!;

  return determinant3([
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

function orientTetrahedra(
  positions: Float64Array,
  tetrahedra: Uint32Array,
): void {
  for (let offset = 0; offset < tetrahedra.length; offset += 4) {
    const a = tetrahedra[offset]!;
    const b = tetrahedra[offset + 1]!;
    const c = tetrahedra[offset + 2]!;
    const d = tetrahedra[offset + 3]!;
    const determinant = tetrahedronDeterminant(positions, a, b, c, d);

    if (Math.abs(determinant) <= 1e-14) {
      throw new Error(`Tetrahedron ${offset / 4} is degenerate.`);
    }
    if (determinant < 0) {
      tetrahedra[offset + 1] = c;
      tetrahedra[offset + 2] = b;
    }
  }
}

export function generateRegularCuboidMesh(
  options: RegularCuboidOptions,
): TetrahedralMesh {
  const [cellCountX, cellCountY, cellCountZ] = options.cells;
  assertPositiveInteger(cellCountX, "cells[0]");
  assertPositiveInteger(cellCountY, "cells[1]");
  assertPositiveInteger(cellCountZ, "cells[2]");

  const origin = options.origin ?? [0, 0, 0];
  const size = options.size ?? [1, 1, 1];
  if (size.some((value) => !(value > 0) || !Number.isFinite(value))) {
    throw new RangeError("Cuboid dimensions must be finite and positive.");
  }
  const materialId = options.materialId ?? 0;
  if (!Number.isInteger(materialId) || materialId < 0 || materialId > 65_535) {
    throw new RangeError("materialId must fit in an unsigned 16-bit integer.");
  }
  const bodyId = options.bodyId ?? 0;
  if (!Number.isInteger(bodyId) || bodyId < 0 || bodyId > 65_535) {
    throw new RangeError("bodyId must fit in an unsigned 16-bit integer.");
  }

  const vertexCount =
    (cellCountX + 1) * (cellCountY + 1) * (cellCountZ + 1);
  const positions = new Float64Array(vertexCount * 3);
  const fixed = new Uint8Array(vertexCount);
  const bodyIds = new Uint16Array(vertexCount);
  bodyIds.fill(bodyId);
  const strideY = cellCountX + 1;
  const strideZ = strideY * (cellCountY + 1);
  const vertexIndex = (x: number, y: number, z: number) =>
    x + y * strideY + z * strideZ;

  for (let z = 0; z <= cellCountZ; z += 1) {
    for (let y = 0; y <= cellCountY; y += 1) {
      for (let x = 0; x <= cellCountX; x += 1) {
        const vertex = vertexIndex(x, y, z);
        const position: Vec3 = [
          origin[0] + (size[0] * x) / cellCountX,
          origin[1] + (size[1] * y) / cellCountY,
          origin[2] + (size[2] * z) / cellCountZ,
        ];
        positions.set(position, vertex * 3);
        fixed[vertex] = options.fixed?.(position, [x, y, z]) ? 1 : 0;
      }
    }
  }

  const tetrahedronCount = cellCountX * cellCountY * cellCountZ * 6;
  const tetrahedra = new Uint32Array(tetrahedronCount * 4);
  let tetrahedronOffset = 0;

  for (let z = 0; z < cellCountZ; z += 1) {
    for (let y = 0; y < cellCountY; y += 1) {
      for (let x = 0; x < cellCountX; x += 1) {
        const v000 = vertexIndex(x, y, z);
        const v100 = vertexIndex(x + 1, y, z);
        const v010 = vertexIndex(x, y + 1, z);
        const v110 = vertexIndex(x + 1, y + 1, z);
        const v001 = vertexIndex(x, y, z + 1);
        const v101 = vertexIndex(x + 1, y, z + 1);
        const v011 = vertexIndex(x, y + 1, z + 1);
        const v111 = vertexIndex(x + 1, y + 1, z + 1);
        const cellTetrahedra = [
          v000,
          v100,
          v110,
          v111,
          v000,
          v110,
          v010,
          v111,
          v000,
          v010,
          v011,
          v111,
          v000,
          v011,
          v001,
          v111,
          v000,
          v001,
          v101,
          v111,
          v000,
          v101,
          v100,
          v111,
        ];
        tetrahedra.set(cellTetrahedra, tetrahedronOffset);
        tetrahedronOffset += cellTetrahedra.length;
      }
    }
  }

  orientTetrahedra(positions, tetrahedra);
  const materialIds = new Uint16Array(tetrahedronCount);
  materialIds.fill(materialId);

  return { positions, tetrahedra, materialIds, fixed, bodyIds };
}

function resolveScale(scale: number | Vec3 | undefined): Vec3 {
  if (scale === undefined) {
    return [1, 1, 1];
  }
  return typeof scale === "number" ? [scale, scale, scale] : scale;
}

function rotateEulerXyz(position: Vec3, rotation: Vec3): Vec3 {
  const [rx, ry, rz] = rotation;
  const cx = Math.cos(rx);
  const sx = Math.sin(rx);
  const cy = Math.cos(ry);
  const sy = Math.sin(ry);
  const cz = Math.cos(rz);
  const sz = Math.sin(rz);

  const afterX: Vec3 = [
    position[0],
    cx * position[1] - sx * position[2],
    sx * position[1] + cx * position[2],
  ];
  const afterY: Vec3 = [
    cy * afterX[0] + sy * afterX[2],
    afterX[1],
    -sy * afterX[0] + cy * afterX[2],
  ];

  return [
    cz * afterY[0] - sz * afterY[1],
    sz * afterY[0] + cz * afterY[1],
    afterY[2],
  ];
}

export function transformTetrahedralMesh(
  mesh: TetrahedralMesh,
  transform: MeshTransform,
): TetrahedralMesh {
  validateTetrahedralMesh(mesh);
  const scale = resolveScale(transform.scale);
  const rotation = transform.rotationEuler ?? [0, 0, 0];
  const translation = transform.translation ?? [0, 0, 0];
  if (
    [...scale, ...rotation, ...translation].some(
      (value) => !Number.isFinite(value),
    ) ||
    scale.some((value) => value === 0)
  ) {
    throw new RangeError("Mesh transforms must be finite and non-singular.");
  }

  const positions = new Float64Array(mesh.positions.length);
  for (let vertex = 0; vertex < getVertexCount(mesh); vertex += 1) {
    const scaled: Vec3 = [
      mesh.positions[vertex * 3]! * scale[0],
      mesh.positions[vertex * 3 + 1]! * scale[1],
      mesh.positions[vertex * 3 + 2]! * scale[2],
    ];
    const rotated = rotateEulerXyz(scaled, rotation);
    positions.set(
      [
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
      ],
      vertex * 3,
    );
  }

  const tetrahedra = mesh.tetrahedra.slice();
  orientTetrahedra(positions, tetrahedra);

  return {
    positions,
    tetrahedra,
    materialIds: mesh.materialIds.slice(),
    fixed: mesh.fixed.slice(),
    bodyIds: mesh.bodyIds.slice(),
  };
}

export function appendTetrahedralMeshes(
  meshes: readonly TetrahedralMesh[],
): TetrahedralMesh {
  if (meshes.length === 0) {
    throw new RangeError("At least one mesh is required.");
  }
  for (const mesh of meshes) {
    validateTetrahedralMesh(mesh);
  }

  const vertexCount = meshes.reduce(
    (sum, mesh) => sum + getVertexCount(mesh),
    0,
  );
  const tetrahedronCount = meshes.reduce(
    (sum, mesh) => sum + getTetrahedronCount(mesh),
    0,
  );
  const positions = new Float64Array(vertexCount * 3);
  const tetrahedra = new Uint32Array(tetrahedronCount * 4);
  const materialIds = new Uint16Array(tetrahedronCount);
  const fixed = new Uint8Array(vertexCount);
  const bodyIds = new Uint16Array(vertexCount);
  let vertexOffset = 0;
  let tetrahedronOffset = 0;

  for (const mesh of meshes) {
    const meshVertexCount = getVertexCount(mesh);
    const meshTetrahedronCount = getTetrahedronCount(mesh);
    positions.set(mesh.positions, vertexOffset * 3);
    fixed.set(mesh.fixed, vertexOffset);
    bodyIds.set(mesh.bodyIds, vertexOffset);
    materialIds.set(mesh.materialIds, tetrahedronOffset);
    for (let index = 0; index < mesh.tetrahedra.length; index += 1) {
      tetrahedra[tetrahedronOffset * 4 + index] =
        mesh.tetrahedra[index]! + vertexOffset;
    }
    vertexOffset += meshVertexCount;
    tetrahedronOffset += meshTetrahedronCount;
  }

  return { positions, tetrahedra, materialIds, fixed, bodyIds };
}

interface FaceRecord {
  count: number;
  readonly key: readonly [number, number, number];
  readonly face: readonly [number, number, number];
}

function compareTriples(
  left: readonly [number, number, number],
  right: readonly [number, number, number],
): number {
  return left[0] - right[0] || left[1] - right[1] || left[2] - right[2];
}

function outwardFace(
  positions: Float64Array,
  face: readonly [number, number, number],
  opposite: number,
): readonly [number, number, number] {
  const [a, b, c] = face;
  const abx = positions[b * 3]! - positions[a * 3]!;
  const aby = positions[b * 3 + 1]! - positions[a * 3 + 1]!;
  const abz = positions[b * 3 + 2]! - positions[a * 3 + 2]!;
  const acx = positions[c * 3]! - positions[a * 3]!;
  const acy = positions[c * 3 + 1]! - positions[a * 3 + 1]!;
  const acz = positions[c * 3 + 2]! - positions[a * 3 + 2]!;
  const normalX = aby * acz - abz * acy;
  const normalY = abz * acx - abx * acz;
  const normalZ = abx * acy - aby * acx;
  const towardOpposite =
    normalX * (positions[opposite * 3]! - positions[a * 3]!) +
    normalY * (positions[opposite * 3 + 1]! - positions[a * 3 + 1]!) +
    normalZ * (positions[opposite * 3 + 2]! - positions[a * 3 + 2]!);

  return towardOpposite > 0 ? [a, c, b] : face;
}

export function extractBoundarySurface(
  mesh: TetrahedralMesh,
): SurfaceTopology {
  validateTetrahedralMesh(mesh);
  const faces = new Map<string, FaceRecord>();

  for (let offset = 0; offset < mesh.tetrahedra.length; offset += 4) {
    const a = mesh.tetrahedra[offset]!;
    const b = mesh.tetrahedra[offset + 1]!;
    const c = mesh.tetrahedra[offset + 2]!;
    const d = mesh.tetrahedra[offset + 3]!;
    const localFaces: ReadonlyArray<
      readonly [number, number, number, number]
    > = [
      [b, c, d, a],
      [a, d, c, b],
      [a, b, d, c],
      [a, c, b, d],
    ];

    for (const [first, second, third, opposite] of localFaces) {
      const sorted = [first, second, third].sort(
        (left, right) => left - right,
      ) as [number, number, number];
      const key = sorted.join(",");
      const existing = faces.get(key);
      if (existing) {
        existing.count += 1;
        if (existing.count > 2) {
          throw new Error(`Non-manifold face ${key} belongs to over two tets.`);
        }
      } else {
        faces.set(key, {
          count: 1,
          key: sorted,
          face: outwardFace(
            mesh.positions,
            [first, second, third],
            opposite,
          ),
        });
      }
    }
  }

  const boundaryFaces = [...faces.values()]
    .filter((record) => record.count === 1)
    .sort((left, right) => compareTriples(left.key, right.key));
  const triangles = new Uint32Array(boundaryFaces.length * 3);
  for (let face = 0; face < boundaryFaces.length; face += 1) {
    triangles.set(boundaryFaces[face]!.face, face * 3);
  }

  const edgeMap = new Map<string, readonly [number, number]>();
  for (let offset = 0; offset < triangles.length; offset += 3) {
    const triangle = [
      triangles[offset]!,
      triangles[offset + 1]!,
      triangles[offset + 2]!,
    ];
    for (let edge = 0; edge < 3; edge += 1) {
      const first = triangle[edge]!;
      const second = triangle[(edge + 1) % 3]!;
      const pair: readonly [number, number] =
        first < second ? [first, second] : [second, first];
      edgeMap.set(`${pair[0]},${pair[1]}`, pair);
    }
  }
  const edgePairs = [...edgeMap.values()].sort(
    (left, right) => left[0] - right[0] || left[1] - right[1],
  );
  const edges = new Uint32Array(edgePairs.length * 2);
  for (let edge = 0; edge < edgePairs.length; edge += 1) {
    edges.set(edgePairs[edge]!, edge * 2);
  }

  return { triangles, edges };
}
