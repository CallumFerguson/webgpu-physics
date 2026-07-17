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

/** Number of scalar values used to store each packed position. */
export type IpcPositionStride = 3 | 4;

type IpcVec3 = readonly [number, number, number];

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

const IPC_LENGTH_SQUARED_FLOOR = 1e-20;
const IPC_PARALLEL_RELATIVE_EPSILON = 1e-6;

function dot3(left: IpcVec3, right: IpcVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function subtract3(left: IpcVec3, right: IpcVec3): IpcVec3 {
  return [
    left[0] - right[0],
    left[1] - right[1],
    left[2] - right[2],
  ];
}

function cross3(left: IpcVec3, right: IpcVec3): IpcVec3 {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function clamp01(value: number): number {
  if (Number.isNaN(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function finiteDistance(first: IpcVec3, second: IpcVec3): number {
  const distance = Math.hypot(
    first[0] - second[0],
    first[1] - second[1],
    first[2] - second[2],
  );
  return Number.isFinite(distance) ? distance : Number.MAX_VALUE;
}

function interpolate3(start: IpcVec3, end: IpcVec3, parameter: number): IpcVec3 {
  const point: IpcVec3 = [
    start[0] + parameter * (end[0] - start[0]),
    start[1] + parameter * (end[1] - start[1]),
    start[2] + parameter * (end[2] - start[2]),
  ];
  if (point.every(Number.isFinite)) {
    return point;
  }
  // Arithmetic can overflow for finite values near Number.MAX_VALUE. An
  // endpoint fallback preserves a finite deterministic result in that case.
  return parameter <= 0.5 ? start : end;
}

interface ClosestSegmentPoint {
  readonly point: IpcVec3;
  readonly parameter: number;
  readonly distance: number;
}

function closestPointOnSegment(
  query: IpcVec3,
  start: IpcVec3,
  end: IpcVec3,
): ClosestSegmentPoint {
  const edge = subtract3(end, start);
  const edgeLengthSquared = dot3(edge, edge);
  let parameter = 0;
  if (
    Number.isFinite(edgeLengthSquared) &&
    edgeLengthSquared > IPC_LENGTH_SQUARED_FLOOR
  ) {
    parameter = clamp01(dot3(subtract3(query, start), edge) / edgeLengthSquared);
  }
  const point = interpolate3(start, end, parameter);
  return {
    point,
    parameter,
    distance: finiteDistance(query, point),
  };
}

function validatePositionStride(positionStride: number): asserts positionStride is IpcPositionStride {
  if (positionStride !== 3 && positionStride !== 4) {
    throw new RangeError("IPC position stride must be 3 or 4.");
  }
}

function readPackedPosition(
  positions: ArrayLike<number>,
  vertex: number,
  positionStride: IpcPositionStride,
): IpcVec3 {
  if (!Number.isSafeInteger(vertex) || vertex < 0) {
    throw new RangeError(`IPC candidate vertex ${vertex} must be a non-negative integer.`);
  }
  if (positions.length % positionStride !== 0) {
    throw new RangeError(
      `Packed IPC positions length must be divisible by stride ${positionStride}.`,
    );
  }
  const vertexCount = positions.length / positionStride;
  if (vertex >= vertexCount) {
    throw new RangeError(
      `IPC candidate vertex ${vertex} is outside [0, ${vertexCount}).`,
    );
  }
  const offset = vertex * positionStride;
  const position: IpcVec3 = [
    positions[offset]!,
    positions[offset + 1]!,
    positions[offset + 2]!,
  ];
  for (const coordinate of position) {
    requireFinite(coordinate, `IPC vertex ${vertex} coordinate`);
  }
  return position;
}

/**
 * Robust unsigned point-triangle distance for one static IPC candidate.
 * Degenerate triangles fall back to the closest of their three segments,
 * matching the GPU contact feature selection (including deterministic ties).
 */
export function evaluateIpcVertexTriangleDistance(
  positions: ArrayLike<number>,
  candidate: IpcVertexTriangleCandidate,
  positionStride: IpcPositionStride = 4,
): number {
  validatePositionStride(positionStride);
  const point = readPackedPosition(positions, candidate[0], positionStride);
  const triangle0 = readPackedPosition(positions, candidate[1], positionStride);
  const triangle1 = readPackedPosition(positions, candidate[2], positionStride);
  const triangle2 = readPackedPosition(positions, candidate[3], positionStride);

  const edge01 = subtract3(triangle1, triangle0);
  const edge02 = subtract3(triangle2, triangle0);
  const edge12 = subtract3(triangle2, triangle1);
  const edge01Squared = dot3(edge01, edge01);
  const edge02Squared = dot3(edge02, edge02);
  const edge12Squared = dot3(edge12, edge12);
  const largestEdgeSquared = Math.max(
    edge01Squared,
    edge02Squared,
    edge12Squared,
  );
  const areaNormal = cross3(edge01, edge02);
  const areaSquared = dot3(areaNormal, areaNormal);
  const areaThreshold = Math.max(
    IPC_LENGTH_SQUARED_FLOOR * IPC_LENGTH_SQUARED_FLOOR,
    IPC_PARALLEL_RELATIVE_EPSILON *
      largestEdgeSquared *
      largestEdgeSquared,
  );

  if (Number.isFinite(areaSquared) && areaSquared > areaThreshold) {
    const dot00 = edge01Squared;
    const dot01 = dot3(edge01, edge02);
    const dot11 = edge02Squared;
    const pointOffset = subtract3(point, triangle0);
    const dot20 = dot3(pointOffset, edge01);
    const dot21 = dot3(pointOffset, edge02);
    const denominator = dot00 * dot11 - dot01 * dot01;
    const barycentric1 = (dot11 * dot20 - dot01 * dot21) / denominator;
    const barycentric2 = (dot00 * dot21 - dot01 * dot20) / denominator;
    const barycentric0 = 1 - barycentric1 - barycentric2;
    if (
      Number.isFinite(barycentric0) &&
      Number.isFinite(barycentric1) &&
      Number.isFinite(barycentric2) &&
      barycentric0 >= 0 &&
      barycentric1 >= 0 &&
      barycentric2 >= 0
    ) {
      const closest: IpcVec3 = [
        barycentric0 * triangle0[0] +
          barycentric1 * triangle1[0] +
          barycentric2 * triangle2[0],
        barycentric0 * triangle0[1] +
          barycentric1 * triangle1[1] +
          barycentric2 * triangle2[1],
        barycentric0 * triangle0[2] +
          barycentric1 * triangle1[2] +
          barycentric2 * triangle2[2],
      ];
      if (closest.every(Number.isFinite)) {
        return finiteDistance(point, closest);
      }
    }
  }

  let closest = closestPointOnSegment(point, triangle0, triangle1);
  const candidate12 = closestPointOnSegment(point, triangle1, triangle2);
  const candidate20 = closestPointOnSegment(point, triangle2, triangle0);
  if (candidate12.distance < closest.distance) {
    closest = candidate12;
  }
  if (candidate20.distance < closest.distance) {
    closest = candidate20;
  }
  return closest.distance;
}

/**
 * Robust unsigned distance between two finite segments for one static IPC
 * candidate, including point-segment, point-point, and parallel degeneracies.
 */
export function evaluateIpcEdgeEdgeDistance(
  positions: ArrayLike<number>,
  candidate: IpcEdgeEdgeCandidate,
  positionStride: IpcPositionStride = 4,
): number {
  validatePositionStride(positionStride);
  const edgeA0 = readPackedPosition(positions, candidate[0], positionStride);
  const edgeA1 = readPackedPosition(positions, candidate[1], positionStride);
  const edgeB0 = readPackedPosition(positions, candidate[2], positionStride);
  const edgeB1 = readPackedPosition(positions, candidate[3], positionStride);

  const directionA = subtract3(edgeA1, edgeA0);
  const directionB = subtract3(edgeB1, edgeB0);
  const offset = subtract3(edgeA0, edgeB0);
  const lengthASquared = dot3(directionA, directionA);
  const lengthBSquared = dot3(directionB, directionB);
  const lengthScale = Math.max(lengthASquared, lengthBSquared);
  const degenerateThreshold = Math.max(
    IPC_LENGTH_SQUARED_FLOOR,
    IPC_PARALLEL_RELATIVE_EPSILON * lengthScale,
  );
  const aDegenerate = !(lengthASquared > degenerateThreshold);
  const bDegenerate = !(lengthBSquared > degenerateThreshold);
  let parameterA = 0;
  let parameterB = 0;

  if (aDegenerate && !bDegenerate) {
    parameterB = clamp01(dot3(directionB, offset) / lengthBSquared);
  } else if (!aDegenerate) {
    const projectionA = dot3(directionA, offset);
    if (bDegenerate) {
      parameterA = clamp01(-projectionA / lengthASquared);
    } else {
      const coupling = dot3(directionA, directionB);
      const projectionB = dot3(directionB, offset);
      const denominator =
        lengthASquared * lengthBSquared - coupling * coupling;
      const parallelThreshold =
        IPC_PARALLEL_RELATIVE_EPSILON * lengthASquared * lengthBSquared;
      if (denominator > parallelThreshold) {
        parameterA = clamp01(
          (coupling * projectionB - projectionA * lengthBSquared) /
            denominator,
        );
      }
      const parameterBUnclamped =
        (coupling * parameterA + projectionB) / lengthBSquared;
      if (parameterBUnclamped < 0) {
        parameterB = 0;
        parameterA = clamp01(-projectionA / lengthASquared);
      } else if (parameterBUnclamped > 1) {
        parameterB = 1;
        parameterA = clamp01(
          (coupling - projectionA) / lengthASquared,
        );
      } else {
        parameterB = clamp01(parameterBUnclamped);
      }
    }
  }

  const pointA = interpolate3(edgeA0, edgeA1, parameterA);
  const pointB = interpolate3(edgeB0, edgeB1, parameterB);
  return finiteDistance(pointA, pointB);
}

/**
 * Returns the minimum unsigned distance over a static packed VT/EE superset.
 * The GPU's candidate order is authoritative: VT records come first, followed
 * by EE records. An empty superset returns the finite no-contact sentinel
 * `Number.MAX_VALUE`.
 */
export function minimumStaticIpcContactDistance(
  positions: ArrayLike<number>,
  candidates: StaticIpcContactCandidates,
  positionStride: IpcPositionStride = 4,
): number {
  validatePositionStride(positionStride);
  if (
    !Number.isSafeInteger(candidates.vertexTriangleCount) ||
    candidates.vertexTriangleCount < 0 ||
    !Number.isSafeInteger(candidates.edgeEdgeCount) ||
    candidates.edgeEdgeCount < 0
  ) {
    throw new RangeError("IPC candidate counts must be non-negative integers.");
  }
  const candidateCount =
    candidates.vertexTriangleCount + candidates.edgeEdgeCount;
  if (candidates.packedIndices.length !== candidateCount * 4) {
    throw new RangeError(
      "Packed IPC candidate indices must contain four indices per candidate.",
    );
  }

  let minimumDistance = Number.MAX_VALUE;
  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex += 1) {
    const offset = candidateIndex * 4;
    const candidate = [
      candidates.packedIndices[offset]!,
      candidates.packedIndices[offset + 1]!,
      candidates.packedIndices[offset + 2]!,
      candidates.packedIndices[offset + 3]!,
    ] as const;
    const distance =
      candidateIndex < candidates.vertexTriangleCount
        ? evaluateIpcVertexTriangleDistance(
            positions,
            candidate,
            positionStride,
          )
        : evaluateIpcEdgeEdgeDistance(positions, candidate, positionStride);
    minimumDistance = Math.min(minimumDistance, distance);
  }
  return minimumDistance;
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
