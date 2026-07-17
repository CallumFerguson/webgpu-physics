export const JGS2_SCHEDULE_MAGIC = 0x3247_534a;
export const JGS2_SCHEDULE_HEADER_WORDS = 4;

export interface PackedJGS2ScheduleArena {
  readonly integers: Uint32Array;
  readonly byteLength: number;
  readonly vertexCount: number;
  readonly colorCount: number;
}

function alignTo4Words(wordCount: number): number {
  return Math.ceil(wordCount / 4) * 4;
}

/** Pack the small-scene CPU coloring after the optional cloth arena. */
export function packJGS2ScheduleArena(
  colors: Uint32Array,
  colorCount: number,
): PackedJGS2ScheduleArena {
  if (!(colors instanceof Uint32Array)) {
    throw new TypeError("Schedule colors must use Uint32Array.");
  }
  if (
    !Number.isSafeInteger(colorCount) ||
    colorCount < 1 ||
    colorCount > colors.length
  ) {
    throw new RangeError(
      `Schedule colorCount must be an integer from 1 through ${colors.length}; got ${colorCount}.`,
    );
  }
  for (let vertex = 0; vertex < colors.length; vertex += 1) {
    if (colors[vertex]! >= colorCount) {
      throw new RangeError(
        `Schedule color ${colors[vertex]} for vertex ${vertex} is outside colorCount ${colorCount}.`,
      );
    }
  }

  const integers = new Uint32Array(
    alignTo4Words(JGS2_SCHEDULE_HEADER_WORDS + colors.length),
  );
  integers.set(
    [JGS2_SCHEDULE_MAGIC, colors.length, colorCount, colors.length],
    0,
  );
  integers.set(colors, JGS2_SCHEDULE_HEADER_WORDS);
  return {
    integers,
    byteLength: integers.byteLength,
    vertexCount: colors.length,
    colorCount,
  };
}
