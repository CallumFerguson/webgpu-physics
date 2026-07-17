import { describe, expect, it } from "vitest";

import { encodeSceneTimestampResolve } from "./scene-renderer";

describe("scene timestamp query resolve", () => {
  it("encodes resolve and copy into the existing render command encoder", () => {
    const calls: unknown[][] = [];
    const encoder = {
      resolveQuerySet: (...args: unknown[]) => calls.push(["resolve", ...args]),
      copyBufferToBuffer: (...args: unknown[]) => calls.push(["copy", ...args]),
    } as unknown as GPUCommandEncoder;
    const querySet = {} as GPUQuerySet;
    const resolveBuffer = {} as GPUBuffer;
    const readbackBuffer = {} as GPUBuffer;

    encodeSceneTimestampResolve(encoder, {
      querySet,
      queryCount: 8,
      resolveBuffer,
      readbackBuffer,
      byteLength: 64,
    });

    expect(calls).toEqual([
      ["resolve", querySet, 0, 8, resolveBuffer, 0],
      ["copy", resolveBuffer, 0, readbackBuffer, 0, 64],
    ]);
  });

  it("rejects mismatched query counts and byte lengths", () => {
    const encoder = {} as GPUCommandEncoder;
    expect(() =>
      encodeSceneTimestampResolve(encoder, {
        querySet: {} as GPUQuerySet,
        queryCount: 8,
        resolveBuffer: {} as GPUBuffer,
        readbackBuffer: {} as GPUBuffer,
        byteLength: 8,
      }),
    ).toThrow(/exact byte length/);
  });
});
