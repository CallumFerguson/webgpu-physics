/**
 * Metadata stored next to an offline precomputation payload.
 *
 * Fingerprints detect stale or incompatible artifacts; they are not a trust or
 * signature mechanism. Artifact producers should still distribute payloads
 * through a trusted channel.
 */

export const PRECOMPUTATION_ARTIFACT_SCHEMA =
  "org.jgs2.precomputation-artifact" as const;
export const PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION = 1 as const;
export const PRECOMPUTATION_PAYLOAD_FORMAT =
  "jgs2-precomputation-binary-le" as const;
export const PRECOMPUTATION_PAYLOAD_FORMAT_VERSION = 1 as const;

export const PRECOMPUTATION_FINGERPRINT_COMPONENTS = [
  "topology",
  "materials",
  "timestep",
  "basis",
  "cubature",
  "solverSchedule",
] as const;

export type PrecomputationFingerprintComponent =
  (typeof PRECOMPUTATION_FINGERPRINT_COMPONENTS)[number];
export type Sha256Fingerprint = `sha256:${string}`;

export type PrecomputationFingerprints = Readonly<
  Record<PrecomputationFingerprintComponent, Sha256Fingerprint>
>;

export interface PrecomputationArtifactHeader {
  readonly schema: typeof PRECOMPUTATION_ARTIFACT_SCHEMA;
  readonly schemaVersion: typeof PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION;
  readonly createdBy: {
    readonly name: string;
    readonly version: string;
  };
  readonly fingerprints: PrecomputationFingerprints;
  readonly payload: {
    readonly format: typeof PRECOMPUTATION_PAYLOAD_FORMAT;
    readonly formatVersion: typeof PRECOMPUTATION_PAYLOAD_FORMAT_VERSION;
    readonly byteLength: number;
    readonly fingerprint: Sha256Fingerprint;
  };
}

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

/**
 * Produce a platform-independent fingerprint for JSON-like input, typed
 * arrays, or ArrayBuffers. Object keys are sorted and typed-array kinds are
 * included, so construction order cannot change a fingerprint while a change
 * from (for example) Uint32 topology to Float32 data cannot go unnoticed.
 */
export async function fingerprintCanonicalValue(
  value: unknown,
): Promise<Sha256Fingerprint> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("SHA-256 fingerprinting requires Web Crypto.");
  }
  const canonical = canonicalize(value, new WeakSet<object>());
  const encoded = new TextEncoder().encode(canonical);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", encoded);
  const hex = [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  return `sha256:${hex}`;
}

/** Validate untrusted JSON metadata and return the narrowed v1 header. */
export function validatePrecomputationArtifactHeader(
  value: unknown,
): PrecomputationArtifactHeader {
  const root = requireRecord(value, "precomputation artifact");
  if (root.schema !== PRECOMPUTATION_ARTIFACT_SCHEMA) {
    throw new Error(
      `Unsupported precomputation schema ${describe(root.schema)}; expected ` +
        `${PRECOMPUTATION_ARTIFACT_SCHEMA}.`,
    );
  }
  if (root.schemaVersion !== PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported precomputation schema version ${describe(root.schemaVersion)}; ` +
        `expected ${PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION}.`,
    );
  }

  const createdBy = requireRecord(root.createdBy, "createdBy");
  const name = requireNonemptyString(createdBy.name, "createdBy.name");
  const version = requireNonemptyString(
    createdBy.version,
    "createdBy.version",
  );
  const fingerprints = validatePrecomputationFingerprints(root.fingerprints);
  const payload = requireRecord(root.payload, "payload");
  if (payload.format !== PRECOMPUTATION_PAYLOAD_FORMAT) {
    throw new Error(
      `Unsupported payload format ${describe(payload.format)}; expected ` +
        `${PRECOMPUTATION_PAYLOAD_FORMAT}.`,
    );
  }
  if (payload.formatVersion !== PRECOMPUTATION_PAYLOAD_FORMAT_VERSION) {
    throw new Error(
      `Unsupported payload format version ${describe(payload.formatVersion)}; ` +
        `expected ${PRECOMPUTATION_PAYLOAD_FORMAT_VERSION}.`,
    );
  }
  const byteLength = requireNonnegativeSafeInteger(
    payload.byteLength,
    "payload.byteLength",
  );
  const payloadFingerprint = requireFingerprint(
    payload.fingerprint,
    "payload.fingerprint",
  );

  return {
    schema: PRECOMPUTATION_ARTIFACT_SCHEMA,
    schemaVersion: PRECOMPUTATION_ARTIFACT_SCHEMA_VERSION,
    createdBy: { name, version },
    fingerprints,
    payload: {
      format: PRECOMPUTATION_PAYLOAD_FORMAT,
      formatVersion: PRECOMPUTATION_PAYLOAD_FORMAT_VERSION,
      byteLength,
      fingerprint: payloadFingerprint,
    },
  };
}

/**
 * Reject a structurally invalid or stale artifact and name every incompatible
 * input component in one error so preprocessing can be rerun once.
 */
export function assertPrecomputationArtifactCompatible(
  artifactValue: unknown,
  expectedValue: unknown,
): PrecomputationArtifactHeader {
  const artifact = validatePrecomputationArtifactHeader(artifactValue);
  const expected = validatePrecomputationFingerprints(expectedValue);
  const mismatches = PRECOMPUTATION_FINGERPRINT_COMPONENTS.filter(
    (component) => artifact.fingerprints[component] !== expected[component],
  );
  if (mismatches.length > 0) {
    throw new Error(
      `Precomputation artifact is incompatible with: ${mismatches.join(", ")}.`,
    );
  }
  return artifact;
}

export function validatePrecomputationFingerprints(
  value: unknown,
): PrecomputationFingerprints {
  const record = requireRecord(value, "fingerprints");
  const result = {} as Record<
    PrecomputationFingerprintComponent,
    Sha256Fingerprint
  >;
  for (const component of PRECOMPUTATION_FINGERPRINT_COMPONENTS) {
    result[component] = requireFingerprint(
      record[component],
      `fingerprints.${component}`,
    );
  }
  return result;
}

function canonicalize(value: unknown, active: WeakSet<object>): string {
  if (value === null) {
    return "null";
  }
  switch (typeof value) {
    case "boolean":
      return value ? "boolean:true" : "boolean:false";
    case "number": {
      if (!Number.isFinite(value)) {
        throw new Error("Cannot fingerprint a non-finite number.");
      }
      const normalized = Object.is(value, -0) ? 0 : value;
      return `number:${normalized.toString()}`;
    }
    case "string":
      return `string:${JSON.stringify(value)}`;
    case "bigint":
      return `bigint:${value.toString()}`;
    case "undefined":
    case "function":
    case "symbol":
      throw new Error(`Cannot fingerprint a value of type ${typeof value}.`);
    case "object":
      break;
  }

  if (active.has(value)) {
    throw new Error("Cannot fingerprint a cyclic value.");
  }
  active.add(value);
  try {
    if (Array.isArray(value)) {
      return `array:${value.length}:[${value
        .map((entry) => canonicalize(entry, active))
        .join(",")}]`;
    }
    if (value instanceof ArrayBuffer) {
      return canonicalizeByteBuffer("ArrayBuffer", new Uint8Array(value));
    }
    if (ArrayBuffer.isView(value)) {
      if (value instanceof DataView) {
        return canonicalizeByteBuffer(
          "DataView",
          new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
        );
      }
      const typed = value as unknown as {
        readonly length: number;
        readonly [index: number]: number | bigint;
      };
      const kind = value.constructor.name;
      const entries: string[] = [];
      for (let index = 0; index < typed.length; index += 1) {
        entries.push(canonicalize(typed[index], active));
      }
      return `typed:${kind}:${typed.length}:[${entries.join(",")}]`;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error(
        `Cannot fingerprint non-plain object ${value.constructor?.name ?? "unknown"}.`,
      );
    }
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    return `object:${keys.length}:{${keys
      .map(
        (key) =>
          `${JSON.stringify(key)}=${canonicalize(record[key], active)}`,
      )
      .join(",")}}`;
  } finally {
    active.delete(value);
  }
}

function canonicalizeByteBuffer(kind: string, bytes: Uint8Array): string {
  const hex = [...bytes]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  return `bytes:${kind}:${bytes.byteLength}:${hex}`;
}

function requireRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function requireNonemptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a nonempty string.`);
  }
  return value;
}

function requireNonnegativeSafeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${label} must be a nonnegative safe integer.`);
  }
  return value as number;
}

function requireFingerprint(
  value: unknown,
  label: string,
): Sha256Fingerprint {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256 fingerprint.`);
  }
  return value as Sha256Fingerprint;
}

function describe(value: unknown): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}
