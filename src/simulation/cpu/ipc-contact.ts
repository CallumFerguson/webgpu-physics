import type { SurfaceTopology } from "./types";

export interface IpcBarrierEvaluation {
  /** False only when the distance is non-positive and contact is infeasible. */
  readonly valid: boolean;
  readonly active: boolean;
  readonly value: number;
  readonly firstDerivative: number;
  readonly secondDerivative: number;
}

/** Unoriented [vertex, triangle vertex 0, triangle vertex 1, triangle vertex 2]. */
export type IpcVertexTriangleCandidate = readonly [
  vertex: number,
  triangle0: number,
  triangle1: number,
  triangle2: number,
];

/** Unoriented [edge 0 vertex 0, edge 0 vertex 1, edge 1 vertex 0, edge 1 vertex 1]. */
export type IpcEdgeEdgeCandidate = readonly [
  edge0Vertex0: number,
  edge0Vertex1: number,
  edge1Vertex0: number,
  edge1Vertex1: number,
];

export interface StaticIpcContactCandidates {
  readonly vertexTriangleCandidates: readonly IpcVertexTriangleCandidate[];
  readonly edgeEdgeCandidates: readonly IpcEdgeEdgeCandidate[];
  /**
   * Four u32 indices per candidate. Vertex-triangle candidates come first,
   * followed by edge-edge candidates, matching the two counts above.
   */
  readonly packedIndices: Uint32Array;
  readonly vertexTriangleCount: number;
  readonly edgeEdgeCount: number;
}

type Triangle = readonly [number, number, number];
type Edge = readonly [number, number];

function requireFinite(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite.`);
  }
}

function requireNonnegativeFinite(value: number, label: string): void {
  requireFinite(value, label);
  if (value < 0) {
    throw new RangeError(`${label} must be non-negative.`);
  }
}

function requirePositiveFinite(value: number, label: string): void {
  requireFinite(value, label);
  if (!(value > 0)) {
    throw new RangeError(`${label} must be positive.`);
  }
}

/**
 * Evaluates the IPC barrier
 *   B(d, dHat) = -(d - dHat)^2 log(d / dHat)
 * and its analytic distance derivatives.
 *
 * A non-positive distance is an infeasible state. It is represented explicitly
 * instead of evaluating log at or below zero.
 */
export function evaluateIpcBarrier(
  distance: number,
  activationDistance: number,
): IpcBarrierEvaluation {
  requireFinite(distance, "IPC distance");
  requirePositiveFinite(activationDistance, "IPC activation distance");

  if (distance <= 0) {
    return {
      valid: false,
      active: true,
      value: Number.POSITIVE_INFINITY,
      firstDerivative: Number.NEGATIVE_INFINITY,
      secondDerivative: Number.POSITIVE_INFINITY,
    };
  }

  if (distance >= activationDistance) {
    return {
      valid: true,
      active: false,
      value: 0,
      firstDerivative: 0,
      secondDerivative: 0,
    };
  }

  const offset = distance - activationDistance;
  const logRatio = Math.log1p(offset / activationDistance);
  const offsetOverDistance = offset / distance;

  return {
    valid: true,
    active: true,
    value: -offset * offset * logRatio,
    firstDerivative:
      -2 * offset * logRatio - (offset * offset) / distance,
    secondDerivative:
      -2 * logRatio -
      4 * offsetOverDistance +
      offsetOverDistance * offsetOverDistance,
  };
}

/** IPC's C1-smoothed friction magnitude f1. */
export function ipcFrictionF1(
  slidingMagnitude: number,
  smoothingScale: number,
): number {
  requireNonnegativeFinite(slidingMagnitude, "Sliding magnitude");
  requirePositiveFinite(smoothingScale, "Friction smoothing scale");

  if (slidingMagnitude >= smoothingScale) {
    return 1;
  }

  const normalized = slidingMagnitude / smoothingScale;
  return -normalized * normalized + 2 * normalized;
}

/** IPC's friction potential f0, whose derivative is f1. */
export function ipcFrictionF0(
  slidingMagnitude: number,
  smoothingScale: number,
): number {
  requireNonnegativeFinite(slidingMagnitude, "Sliding magnitude");
  requirePositiveFinite(smoothingScale, "Friction smoothing scale");

  if (slidingMagnitude >= smoothingScale) {
    return slidingMagnitude;
  }

  const squaredScale = smoothingScale * smoothingScale;
  return (
    -(slidingMagnitude * slidingMagnitude * slidingMagnitude) /
      (3 * squaredScale) +
    (slidingMagnitude * slidingMagnitude) / smoothingScale +
    smoothingScale / 3
  );
}

function compareTuples(left: readonly number[], right: readonly number[]): number {
  for (let index = 0; index < left.length; index += 1) {
    const difference = left[index]! - right[index]!;
    if (difference !== 0) {
      return difference;
    }
  }
  return 0;
}

function validateVertexIndex(index: number, vertexCount: number): void {
  if (!Number.isInteger(index) || index < 0 || index >= vertexCount) {
    throw new RangeError(
      `Surface vertex index ${index} is outside [0, ${vertexCount}).`,
    );
  }
}

function canonicalTriangles(
  indices: ArrayLike<number>,
  vertexCount: number,
): Triangle[] {
  if (indices.length % 3 !== 0) {
    throw new RangeError("Surface triangle indices must be packed in triples.");
  }

  const unique = new Map<string, Triangle>();
  for (let offset = 0; offset < indices.length; offset += 3) {
    const vertices = [
      indices[offset]!,
      indices[offset + 1]!,
      indices[offset + 2]!,
    ].sort((left, right) => left - right) as [number, number, number];
    for (const vertex of vertices) {
      validateVertexIndex(vertex, vertexCount);
    }
    if (vertices[0] === vertices[1] || vertices[1] === vertices[2]) {
      throw new RangeError("Surface triangles must have three distinct vertices.");
    }
    unique.set(vertices.join(","), vertices);
  }

  return [...unique.values()].sort(compareTuples);
}

function addCanonicalEdge(
  unique: Map<string, Edge>,
  first: number,
  second: number,
  vertexCount: number,
): void {
  validateVertexIndex(first, vertexCount);
  validateVertexIndex(second, vertexCount);
  if (first === second) {
    throw new RangeError("Surface edges must have two distinct vertices.");
  }
  const edge: Edge = first < second ? [first, second] : [second, first];
  unique.set(`${edge[0]},${edge[1]}`, edge);
}

function canonicalEdges(
  indices: ArrayLike<number>,
  triangles: readonly Triangle[],
  vertexCount: number,
): Edge[] {
  if (indices.length % 2 !== 0) {
    throw new RangeError("Surface edge indices must be packed in pairs.");
  }

  const unique = new Map<string, Edge>();
  for (let offset = 0; offset < indices.length; offset += 2) {
    addCanonicalEdge(
      unique,
      indices[offset]!,
      indices[offset + 1]!,
      vertexCount,
    );
  }
  // Deriving triangle edges makes the graph complete even for a triangle-only
  // caller while deduplicating the explicit SurfaceTopology edge list.
  for (const [first, second, third] of triangles) {
    addCanonicalEdge(unique, first, second, vertexCount);
    addCanonicalEdge(unique, second, third, vertexCount);
    addCanonicalEdge(unique, third, first, vertexCount);
  }

  return [...unique.values()].sort(compareTuples);
}

function buildOneRing(
  vertexCount: number,
  edges: readonly Edge[],
): readonly ReadonlySet<number>[] {
  const adjacency = Array.from(
    { length: vertexCount },
    () => new Set<number>(),
  );
  for (const [first, second] of edges) {
    adjacency[first]!.add(second);
    adjacency[second]!.add(first);
  }
  return adjacency;
}

function isInClosedOneRing(
  first: number,
  second: number,
  adjacency: readonly ReadonlySet<number>[],
): boolean {
  return first === second || adjacency[first]!.has(second);
}

/**
 * Enumerates the static IPC broad-phase superset for a small surface mesh.
 * Features are canonicalized so the result is independent of triangle/edge
 * input order. Primitive pairs sharing a vertex or joined through one surface
 * edge are omitted; disconnected components remain eligible for contact.
 */
export function buildStaticIpcContactCandidates(
  vertexCount: number,
  surface: SurfaceTopology,
): StaticIpcContactCandidates {
  if (!Number.isSafeInteger(vertexCount) || vertexCount < 0) {
    throw new RangeError("Surface vertex count must be a non-negative integer.");
  }

  const triangles = canonicalTriangles(surface.triangles, vertexCount);
  const edges = canonicalEdges(surface.edges, triangles, vertexCount);
  const adjacency = buildOneRing(vertexCount, edges);

  const vertexTriangleCandidates: IpcVertexTriangleCandidate[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (const triangle of triangles) {
      if (
        triangle.some((other) =>
          isInClosedOneRing(vertex, other, adjacency),
        )
      ) {
        continue;
      }
      vertexTriangleCandidates.push([vertex, ...triangle]);
    }
  }

  const edgeEdgeCandidates: IpcEdgeEdgeCandidate[] = [];
  for (let firstIndex = 0; firstIndex < edges.length; firstIndex += 1) {
    const first = edges[firstIndex]!;
    for (
      let secondIndex = firstIndex + 1;
      secondIndex < edges.length;
      secondIndex += 1
    ) {
      const second = edges[secondIndex]!;
      const topologicallyAdjacent = first.some((firstVertex) =>
        second.some((secondVertex) =>
          isInClosedOneRing(firstVertex, secondVertex, adjacency),
        ),
      );
      if (!topologicallyAdjacent) {
        edgeEdgeCandidates.push([...first, ...second]);
      }
    }
  }

  const packedIndices = new Uint32Array(
    (vertexTriangleCandidates.length + edgeEdgeCandidates.length) * 4,
  );
  let packedOffset = 0;
  for (const candidate of [
    ...vertexTriangleCandidates,
    ...edgeEdgeCandidates,
  ]) {
    packedIndices.set(candidate, packedOffset);
    packedOffset += 4;
  }

  return {
    vertexTriangleCandidates,
    edgeEdgeCandidates,
    packedIndices,
    vertexTriangleCount: vertexTriangleCandidates.length,
    edgeEdgeCount: edgeEdgeCandidates.length,
  };
}
