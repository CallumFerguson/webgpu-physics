export interface GreedyVertexColoring {
  /** The color assigned to each vertex, indexed by vertex id. */
  colors: Uint32Array;
  colorCount: number;
  /** Number of vertices assigned to each color. */
  colorCounts: Uint32Array;
  /** Prefix offsets into `verticesByColor`, with `colorCount + 1` entries. */
  colorOffsets: Uint32Array;
  /** Vertices grouped by color and ordered by ascending vertex id per group. */
  verticesByColor: Uint32Array;
}

/**
 * Deterministically colors the vertex conflict graph induced by element groups.
 *
 * Every pair of distinct vertices in a group is treated as a conflict. Vertices
 * are visited in ascending id order and receive the smallest available color.
 * Reordering the input groups therefore does not change the result.
 */
export function buildGreedyVertexColoring(
  vertexCount: number,
  conflictGroups: readonly ArrayLike<number>[],
): GreedyVertexColoring {
  if (!Number.isSafeInteger(vertexCount) || vertexCount < 0) {
    throw new Error("vertexCount must be a non-negative safe integer");
  }

  const adjacency = Array.from(
    { length: vertexCount },
    () => new Set<number>(),
  );

  for (let groupIndex = 0; groupIndex < conflictGroups.length; groupIndex += 1) {
    const group = conflictGroups[groupIndex];
    const uniqueVertices: number[] = [];
    const seen = new Set<number>();

    for (let localIndex = 0; localIndex < group.length; localIndex += 1) {
      const vertex = group[localIndex];
      if (!Number.isSafeInteger(vertex) || vertex < 0 || vertex >= vertexCount) {
        throw new Error(
          `conflictGroups[${groupIndex}][${localIndex}] must be a valid vertex id`,
        );
      }
      if (!seen.has(vertex)) {
        seen.add(vertex);
        uniqueVertices.push(vertex);
      }
    }

    for (let a = 0; a < uniqueVertices.length; a += 1) {
      const vertexA = uniqueVertices[a];
      for (let b = a + 1; b < uniqueVertices.length; b += 1) {
        const vertexB = uniqueVertices[b];
        adjacency[vertexA].add(vertexB);
        adjacency[vertexB].add(vertexA);
      }
    }
  }

  const uncolored = 0xffffffff;
  const colors = new Uint32Array(vertexCount);
  colors.fill(uncolored);
  const usedColorStamp = new Uint32Array(vertexCount);
  let stamp = 0;
  let colorCount = 0;

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    stamp += 1;
    for (const neighbor of adjacency[vertex]) {
      const neighborColor = colors[neighbor];
      if (neighborColor !== uncolored) {
        usedColorStamp[neighborColor] = stamp;
      }
    }

    let color = 0;
    while (color < colorCount && usedColorStamp[color] === stamp) {
      color += 1;
    }
    colors[vertex] = color;
    colorCount = Math.max(colorCount, color + 1);
  }

  const colorCounts = new Uint32Array(colorCount);
  for (const color of colors) {
    colorCounts[color] += 1;
  }

  const colorOffsets = new Uint32Array(colorCount + 1);
  for (let color = 0; color < colorCount; color += 1) {
    colorOffsets[color + 1] = colorOffsets[color] + colorCounts[color];
  }

  const verticesByColor = new Uint32Array(vertexCount);
  const writeOffsets = colorOffsets.slice(0, colorCount);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const color = colors[vertex];
    verticesByColor[writeOffsets[color]] = vertex;
    writeOffsets[color] += 1;
  }

  return {
    colors,
    colorCount,
    colorCounts,
    colorOffsets,
    verticesByColor,
  };
}
