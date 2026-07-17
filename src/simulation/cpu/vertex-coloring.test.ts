import { describe, expect, it } from "vitest";

import { buildGreedyVertexColoring } from "./vertex-coloring";

describe("greedy vertex coloring", () => {
  it("deterministically separates tetrahedron, cloth, and IPC conflicts", () => {
    const groups = [
      new Uint32Array([0, 1, 2, 3]),
      new Uint32Array([0, 4, 5]),
      new Uint32Array([1, 4, 6, 7]),
      new Uint32Array([2, 5, 6, 8]),
    ];

    const coloring = buildGreedyVertexColoring(9, groups);
    const reordered = buildGreedyVertexColoring(9, [...groups].reverse());

    expect([...coloring.colors]).toEqual([0, 1, 2, 3, 2, 1, 0, 3, 3]);
    expect(coloring.colorCount).toBe(4);
    expect([...coloring.colorCounts]).toEqual([2, 2, 2, 3]);
    expect([...coloring.colorOffsets]).toEqual([0, 2, 4, 6, 9]);
    expect([...coloring.verticesByColor]).toEqual([0, 6, 1, 5, 2, 4, 3, 7, 8]);
    expect(reordered).toEqual(coloring);

    for (const group of groups) {
      expect(new Set([...group].map((vertex) => coloring.colors[vertex])).size)
        .toBe(group.length);
    }
  });
});
