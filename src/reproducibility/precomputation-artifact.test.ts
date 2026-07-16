import { describe, expect, it } from "vitest";

import {
  PRECOMPUTATION_ARTIFACT_SCHEMA,
  PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION,
  PRECOMPUTATION_FINGERPRINT_COMPONENTS,
  PRECOMPUTATION_PAYLOAD_FORMAT,
  PRECOMPUTATION_PAYLOAD_FORMAT_VERSION,
  assertPrecomputationArtifactCompatible,
  fingerprintCanonicalValue,
  validatePrecomputationArtifactHeader,
  type PrecomputationArtifactHeader,
  type PrecomputationFingerprints,
  type Sha256Fingerprint,
} from "./precomputation-artifact";

function fingerprint(character: string): Sha256Fingerprint {
  return `sha256:${character.repeat(64)}`;
}

function fingerprints(): PrecomputationFingerprints {
  return {
    topology: fingerprint("1"),
    materials: fingerprint("2"),
    timestep: fingerprint("3"),
    basis: fingerprint("4"),
    cubature: fingerprint("5"),
    solverSchedule: fingerprint("6"),
  };
}

function header(): PrecomputationArtifactHeader {
  return {
    schema: PRECOMPUTATION_ARTIFACT_SCHEMA,
    schemaVersion: PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION,
    createdBy: { name: "fixture-preprocessor", version: "0.1.0" },
    fingerprints: fingerprints(),
    payload: {
      format: PRECOMPUTATION_PAYLOAD_FORMAT,
      formatVersion: PRECOMPUTATION_PAYLOAD_FORMAT_VERSION,
      byteLength: 4096,
      fingerprint: fingerprint("a"),
    },
  };
}

describe("versioned precomputation artifacts", () => {
  it("P0-ARTIFACT-001 fingerprints canonical inputs deterministically", async () => {
    const left = await fingerprintCanonicalValue({
      timestep: 1 / 60,
      topology: new Uint32Array([0, 1, 2, 3]),
      nested: { schedule: "jacobi", iterations: 12 },
    });
    const reordered = await fingerprintCanonicalValue({
      nested: { iterations: 12, schedule: "jacobi" },
      topology: new Uint32Array([0, 1, 2, 3]),
      timestep: 1 / 60,
    });
    const changed = await fingerprintCanonicalValue({
      timestep: 1 / 60,
      topology: new Uint32Array([0, 1, 3, 2]),
      nested: { schedule: "jacobi", iterations: 12 },
    });

    expect(left).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(reordered).toBe(left);
    expect(changed).not.toBe(left);
  });

  it("P0-ARTIFACT-002 distinguishes typed data and rejects ambiguous values", async () => {
    expect(
      await fingerprintCanonicalValue(new Uint32Array([1, 2, 3])),
    ).not.toBe(await fingerprintCanonicalValue(new Float32Array([1, 2, 3])));
    await expect(fingerprintCanonicalValue(Number.NaN)).rejects.toThrow(
      /non-finite/,
    );

    const cyclic: { self?: unknown } = {};
    cyclic.self = cyclic;
    await expect(fingerprintCanonicalValue(cyclic)).rejects.toThrow(/cyclic/);
  });

  it("P0-ARTIFACT-003 validates every v1 schema field", () => {
    expect(validatePrecomputationArtifactHeader(header())).toEqual(header());

    expect(() =>
      validatePrecomputationArtifactHeader({
        ...header(),
        schemaVersion: 2,
      }),
    ).toThrow(/schema version 2/);
    expect(() =>
      validatePrecomputationArtifactHeader({
        ...header(),
        payload: { ...header().payload, byteLength: -1 },
      }),
    ).toThrow(/byteLength/);
    expect(() =>
      validatePrecomputationArtifactHeader({
        ...header(),
        fingerprints: { ...fingerprints(), topology: "sha256:ABC" },
      }),
    ).toThrow(/topology/);
  });

  it("P0-ARTIFACT-004 accepts matching inputs and names every stale component", () => {
    expect(
      assertPrecomputationArtifactCompatible(header(), fingerprints()),
    ).toEqual(header());

    const stale = {
      ...fingerprints(),
      topology: fingerprint("b"),
      timestep: fingerprint("c"),
      solverSchedule: fingerprint("d"),
    };
    expect(() =>
      assertPrecomputationArtifactCompatible(header(), stale),
    ).toThrow(/topology, timestep, solverSchedule/);
  });

  it("P0-ARTIFACT-005 keeps all required compatibility dimensions explicit", () => {
    expect(PRECOMPUTATION_FINGERPRINT_COMPONENTS).toEqual([
      "topology",
      "materials",
      "timestep",
      "basis",
      "cubature",
      "solverSchedule",
    ]);
  });
});
