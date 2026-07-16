import { describe, expect, it } from "vitest";

import { jgs2TimestepsMatch } from "./layout";

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
