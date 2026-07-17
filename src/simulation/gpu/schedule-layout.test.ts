import { describe, expect, it } from "vitest";

import {
  JGS2_SCHEDULE_HEADER_WORDS,
  JGS2_SCHEDULE_MAGIC,
  packJGS2ScheduleArena,
} from "./schedule-layout";

describe("JGS2 colored schedule arena", () => {
  it("packs an aligned, self-describing color map", () => {
    const packed = packJGS2ScheduleArena(
      new Uint32Array([0, 1, 2, 0, 1]),
      3,
    );

    expect(packed.integers.length % 4).toBe(0);
    expect(Array.from(packed.integers.slice(0, 4))).toEqual([
      JGS2_SCHEDULE_MAGIC,
      5,
      3,
      5,
    ]);
    expect(
      Array.from(
        packed.integers.slice(JGS2_SCHEDULE_HEADER_WORDS, JGS2_SCHEDULE_HEADER_WORDS + 5),
      ),
    ).toEqual([0, 1, 2, 0, 1]);
  });

  it("rejects colors outside the declared compact range", () => {
    expect(() =>
      packJGS2ScheduleArena(new Uint32Array([0, 2]), 2),
    ).toThrow(/outside colorCount/);
  });
});
