import { describe, expect, it } from "vitest";

import {
  computeJGS2DynamicOffsets,
  inferJGS2BodyCount,
  jgs2TimestepsMatch,
  validateJGS2ContactParameters,
} from "./layout";

describe("JGS2 timestep compatibility", () => {
  it("accepts values represented by the same GPU f32", () => {
    const timestep = 1 / 60;
    expect(jgs2TimestepsMatch(timestep, timestep)).toBe(true);
    expect(jgs2TimestepsMatch(timestep, Math.fround(timestep))).toBe(true);
  });

  it("rejects a changed or invalid timestep", () => {
    expect(jgs2TimestepsMatch(1 / 60, 1 / 50)).toBe(false);
    expect(jgs2TimestepsMatch(1 / 60, Number.NaN)).toBe(false);
    expect(jgs2TimestepsMatch(1 / 60, 0)).toBe(false);
  });
});

describe("JGS2 per-body bookkeeping", () => {
  it("allocates body corrections after all per-tetrahedron rotations", () => {
    const offsets = computeJGS2DynamicOffsets(2, 1, 3);

    expect(offsets).toEqual({
      posA: 0,
      posB: 2,
      predicted: 4,
      velocity: 6,
      old: 8,
      vertexRotation: 10,
      tetRotation: 16,
      bodyCorrection: 19,
      vec4Count: 22,
    });
  });

  it("infers body count from packed vertexInfo body ids", () => {
    const vertexInfo = new Uint32Array([
      0, 0, 0, 0,
      0, 0, 0, 1,
      0, 0, 0, 1,
    ]);

    expect(inferJGS2BodyCount(vertexInfo, 3)).toBe(2);
  });

  it("rejects body ids that would allocate more bodies than vertices", () => {
    expect(() =>
      inferJGS2BodyCount(new Uint32Array([0, 0, 0, 1]), 1),
    ).toThrow(/maximum is smaller than vertexCount/);
  });
});

describe("JGS2 contact parameters", () => {
  it("accepts a nonnegative damping rate and contact margin", () => {
    expect(() => validateJGS2ContactParameters(8, 0.01)).not.toThrow();
    expect(() => validateJGS2ContactParameters(0, 0)).not.toThrow();
  });

  it("rejects invalid damping rates or margins", () => {
    expect(() => validateJGS2ContactParameters(-0.01, 0)).toThrow();
    expect(() => validateJGS2ContactParameters(Number.NaN, 0)).toThrow();
    expect(() =>
      validateJGS2ContactParameters(Number.POSITIVE_INFINITY, 0),
    ).toThrow();
    expect(() => validateJGS2ContactParameters(8, -0.01)).toThrow();
  });
});
