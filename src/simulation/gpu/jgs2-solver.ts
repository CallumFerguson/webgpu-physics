import {
  JGS2_UNIFORM_BYTES,
  JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT,
  JGS2_OBJECTIVE_TARGET_ACTIVE_BIT,
  JGS2_VERTEX_OBJECTIVE_BYTES,
  JGS2_VERTEX_OBJECTIVE_WORDS,
  computeJGS2DynamicOffsets,
  inferJGS2BodyCount,
  inferJGS2MaterialMode,
  jgs2TimestepsMatch,
  minimumJGS2InputDeformationDeterminant,
  normalizeOddIterationCount,
  packJGS2Cubature,
  packJGS2InitialDynamic,
  packJGS2TetStatic,
  packJGS2VertexObjectives,
  packJGS2VertexStatic,
  validateJGS2ContactParameters,
  validateJGS2GpuInput,
  type JGS2DynamicOffsets,
  type JGS2GpuInput,
} from "./layout";
import {
  decodeJGS2GlobalizationHistoryRecord,
  decodeJGS2GlobalizationLocalDiagnostic,
  JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT,
  JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT,
  JGS2_GLOBALIZATION_HISTORY_CAPACITY,
  JGS2_GLOBALIZATION_MAX_RUNTIME_TOLERANCE,
  JGS2_GLOBALIZATION_DETERMINANT_FLOOR,
  JGS2_GLOBALIZATION_HISTORY_VEC4S,
  JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S,
  JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT,
  JGS2_GLOBALIZATION_VALIDITY_BITS_MASK,
  type JGS2GlobalizationHistoryRecord,
  type JGS2GlobalizationLocalDiagnostic,
} from "./jgs2-globalization";
import {
  JGS2GpuOracleEvaluator,
  type JGS2GpuOracleDiagnostics,
} from "./jgs2-diagnostics";
import {
  GpuTimestampFrameTimer,
  type GpuTimestampIntervalWrites,
  type GpuTimestampMeasurement,
} from "./gpu-timestamp";
import {
  packIpcContactBuffer,
  packIpcIncidentCandidateAdjacency,
} from "./ipc-contact-layout";
import {
  JGS2_CLOTH_GLOBAL_WORDS,
  packJGS2GpuClothArena,
} from "./cloth-layout";
import { packJGS2ScheduleArena } from "./schedule-layout";
import { JGS2_WORKGROUP_SIZE, jgs2Shader } from "./jgs2-shader";
import {
  buildGreedyVertexColoring,
  minimumStaticIpcContactDistance,
} from "../cpu";

export type JGS2Schedule = "jacobi" | "graph-colored-gauss-seidel";

export interface JGS2StepSettings {
  /** Parallel Jacobi or one CPU-colored Gauss-Seidel sweep per iteration. */
  readonly schedule: JGS2Schedule;
  readonly timestep: number;
  readonly gravity: readonly [number, number, number];
  readonly iterations: number;
  readonly floorHeight: number;
  readonly floorStiffness: number;
  readonly velocityDamping: number;
  readonly regularization: number;
  readonly rotationEpsilon: number;
  readonly maxStep: number;
  /** Component-aware relative residual tolerance for GPU convergence flags. */
  readonly residualTolerance: number;
  /** Maximum accepted update divided by the initial scene AABB diagonal. */
  readonly normalizedUpdateTolerance: number;
  /** Exponential x/z damping rate in s^-1 for grounded vertices. */
  readonly contactTangentialDamping: number;
  /** Distance above the floor at which tangential damping becomes active. */
  readonly contactMargin: number;
  /** IPC barrier activation distance; the barrier is zero at and above it. */
  readonly ipcActivationDistance: number;
  /** Strict collision-safe separation maintained by the global step cap. */
  readonly ipcMinimumDistance: number;
  /** Nonnegative multiplier applied to the IPC barrier potential. */
  readonly ipcBarrierStiffness: number;
  /** Coulomb coefficient used by the frame-lagged dissipative potential. */
  readonly ipcFrictionCoefficient: number;
  /** Velocity magnitude below which friction is C1-smoothed. */
  readonly ipcFrictionVelocityEpsilon: number;
  /** Fraction of the conservative collision-safe step bound in [0, 1]. */
  readonly ipcStepSafety: number;
  /** Project-specific correction that restores each free body's predicted x/z COM. */
  readonly horizontalBodyCorrection: boolean;
  /** Disable project-specific damping and momentum-altering corrections. */
  readonly parityMode: boolean;
}

export interface JGS2PositionBufferView {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
  readonly stride: 16;
}

export interface JGS2ObjectiveActivity {
  readonly flags: number;
  readonly externalForces: boolean;
  readonly quadraticTargets: boolean;
  readonly externalForceVertexCount: number;
  readonly quadraticTargetVertexCount: number;
}

export interface JGS2GlobalizationDiagnostics {
  readonly local: readonly JGS2GlobalizationLocalDiagnostic[];
  /** Executed records; production stops at convergence, exact APIs do not. */
  readonly history: readonly JGS2GlobalizationHistoryRecord[];
  readonly historyCount: number;
}

export interface JGS2AssembledCandidateTestResult {
  readonly positions: Float32Array;
  readonly globalization: JGS2GlobalizationDiagnostics;
}

export interface JGS2ConvergenceReductionTestInput {
  /** Five xyz arrays in inertia/material/external/target/contact order. */
  readonly gradientComponents: readonly Float32Array[];
  /** Accepted xyz update per vertex. */
  readonly acceptedUpdates: Float32Array;
  readonly assembledAccepted: boolean;
  readonly assembledReverted: boolean;
  readonly localFailureCount: number;
  /** Test-only logical prefix used to exercise reduction-size boundaries. */
  readonly reductionVertexCount?: number;
}

/**
 * Practical command-buffer bound for one solver submission. Callers needing a
 * longer run should chunk it so browsers do not accumulate unbounded commands.
 */
export const JGS2_MAX_BATCH_FRAMES = 240;

/**
 * Maximum total stable-material nonlinear iterations encoded into one command
 * buffer. Each globalized iteration expands to eight compute passes, so this
 * caps the iteration body at roughly 16k passes while preserving the app's
 * normal 120-frame, nine-iteration test batches. The legacy co-rotated path is
 * intentionally governed only by {@link JGS2_MAX_BATCH_FRAMES}.
 */
export const JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION = 2048;

/**
 * Finite normalization scale used when the dynamic-vertex AABB has no extent
 * (including an entirely pinned input). Keeping it small and positive avoids
 * division by zero without letting distant fixed anchors relax convergence.
 */
export const JGS2_DEGENERATE_DYNAMIC_SCENE_SCALE = 1e-12;
const F32_UNIT_ROUNDOFF = 2 ** -24;

export const DEFAULT_JGS2_STEP_SETTINGS: JGS2StepSettings = {
  schedule: "jacobi",
  timestep: 1 / 60,
  gravity: [0, -9.81, 0],
  iterations: 9,
  floorHeight: 0,
  floorStiffness: 20_000,
  velocityDamping: 0.999,
  regularization: 1e-6,
  rotationEpsilon: 1e-7,
  maxStep: 0.1,
  residualTolerance: 1e-3,
  normalizedUpdateTolerance: 1e-3,
  contactTangentialDamping: 8,
  contactMargin: 0.01,
  ipcActivationDistance: 0.05,
  ipcMinimumDistance: 0.001,
  ipcBarrierStiffness: 0,
  ipcFrictionCoefficient: 0,
  ipcFrictionVelocityEpsilon: 0.01,
  ipcStepSafety: 0.9,
  horizontalBodyCorrection: true,
  parityMode: false,
};

const EMPTY_IPC_CONTACT_CANDIDATES = Object.freeze({
  vertexTriangleCandidates: Object.freeze([]),
  edgeEdgeCandidates: Object.freeze([]),
  packedIndices: new Uint32Array(0),
  vertexTriangleCount: 0,
  edgeEdgeCount: 0,
});

interface JGS2GpuBuffers {
  readonly dynamic: GPUBuffer;
  readonly vertices: GPUBuffer;
  readonly tets: GPUBuffer;
  readonly stiffness: GPUBuffer;
  readonly adjacency: GPUBuffer;
  readonly cubature: GPUBuffer;
  readonly objectives: GPUBuffer;
}

interface JGS2FrameTimestampWrites {
  readonly querySet: GPUQuerySet;
  readonly startWriteIndex?: number;
  readonly endWriteIndex?: number;
}

interface JGS2Pipelines {
  readonly predict: GPUComputePipeline;
  readonly tetPolarRotation: GPUComputePipeline;
  readonly vertexPolarRotation: GPUComputePipeline;
  readonly solve: GPUComputePipeline;
  /** Shared by the explicit assembled-candidate test path. */
  readonly assembledVertexEnergy?: GPUComputePipeline;
  readonly candidateTetrahedron?: GPUComputePipeline;
  readonly reduceCandidate?: GPUComputePipeline;
  readonly applyCandidate?: GPUComputePipeline;
  readonly convergenceGradient?: GPUComputePipeline;
  readonly reduceConvergence?: GPUComputePipeline;
  readonly lagContact?: GPUComputePipeline;
  readonly buildActiveContactRows?: GPUComputePipeline;
  readonly promoteAcceptedContactCache?: GPUComputePipeline;
  readonly candidateContactStep?: GPUComputePipeline;
  readonly validateContactStep?: GPUComputePipeline;
  readonly reduceContactStep?: GPUComputePipeline;
  readonly applyContactStep?: GPUComputePipeline;
  readonly copyPosition: GPUComputePipeline;
  readonly bodyHorizontalCorrection: GPUComputePipeline;
  readonly applyBodyHorizontalCorrection: GPUComputePipeline;
  readonly finalize: GPUComputePipeline;
}

interface JGS2UniformState {
  readonly base: GPUBuffer;
  readonly fromBToA: GPUBuffer;
  readonly fromAToB: GPUBuffer;
  readonly baseBindGroup: GPUBindGroup;
  readonly fromBToABindGroup: GPUBindGroup;
  readonly fromAToBBindGroup: GPUBindGroup;
  /** One immutable-address uniform/bind group per direction and active color. */
  readonly coloredAToA: readonly GPUBuffer[];
  readonly coloredBToB: readonly GPUBuffer[];
  readonly coloredAToABindGroups: readonly GPUBindGroup[];
  readonly coloredBToBBindGroups: readonly GPUBindGroup[];
}

const JGS2_COLORED_SCHEDULE_FLAG = 0x8000_0000;
const JGS2_COLORED_SCHEDULE_BODY_MASK = 0x0000_ffff;
/** This is intentionally a bounded small-scene scheduler, not a general one. */
const JGS2_COLORED_SCHEDULE_MAX_COLOR = 255;

function alignTo4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

interface PackedObjectiveActivity {
  readonly flags: number;
  readonly externalForceVertexCount: number;
  readonly quadraticTargetVertexCount: number;
}

function computePackedObjectiveActivity(
  packed: Float32Array,
  vertexCount: number,
): PackedObjectiveActivity {
  let externalForceVertexCount = 0;
  let quadraticTargetVertexCount = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const base = vertex * JGS2_VERTEX_OBJECTIVE_WORDS;
    if (
      packed[base] !== 0 ||
      packed[base + 1] !== 0 ||
      packed[base + 2] !== 0
    ) {
      externalForceVertexCount += 1;
    }
    if (packed[base + 7]! > 0) {
      quadraticTargetVertexCount += 1;
    }
  }
  return {
    flags:
      (externalForceVertexCount > 0
        ? JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT
        : 0) |
      (quadraticTargetVertexCount > 0
        ? JGS2_OBJECTIVE_TARGET_ACTIVE_BIT
        : 0),
    externalForceVertexCount,
    quadraticTargetVertexCount,
  };
}

function packPinnedVertexMask(input: JGS2GpuInput): Uint8Array {
  const pinned = new Uint8Array(input.vertexCount);
  for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
    pinned[vertex] = Number(input.vertexInfo[vertex * 4 + 2] !== 0);
  }
  return pinned;
}

/** Deterministic small-scene conflicts used by the graph-colored schedule. */
export function buildJGS2InputVertexColoring(input: JGS2GpuInput) {
  const conflictGroups: Array<ArrayLike<number>> = [];
  for (let tet = 0; tet < input.tetCount; tet += 1) {
    conflictGroups.push(input.tetIndices.subarray(tet * 4, tet * 4 + 4));
  }
  // A trained record may reference a nonincident tet. Couple its source
  // vertex to every position read by that tet so one in-place color dispatch
  // never reads a position another lane in the same dispatch is writing.
  for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
    for (let sample = 0; sample < input.cubatureK; sample += 1) {
      const record = vertex * input.cubatureK + sample;
      const tet = input.cubatureTetIds[record]!;
      if (tet === 0xffff_ffff || input.cubatureWeights[record]! <= 0) {
        continue;
      }
      conflictGroups.push([
        vertex,
        ...input.tetIndices.subarray(tet * 4, tet * 4 + 4),
      ]);
    }
  }
  if (input.cloth) {
    for (
      let triangle = 0;
      triangle < input.cloth.triangleIndices.length / 3;
      triangle += 1
    ) {
      conflictGroups.push(
        input.cloth.triangleIndices.subarray(
          triangle * 3,
          triangle * 3 + 3,
        ),
      );
    }
    for (
      let hinge = 0;
      hinge < input.cloth.hingeIndices.length / 4;
      hinge += 1
    ) {
      conflictGroups.push(
        input.cloth.hingeIndices.subarray(hinge * 4, hinge * 4 + 4),
      );
    }
  }
  if (input.contactCandidates) {
    for (
      let candidate = 0;
      candidate < input.contactCandidates.packedIndices.length / 4;
      candidate += 1
    ) {
      conflictGroups.push(
        input.contactCandidates.packedIndices.subarray(
          candidate * 4,
          candidate * 4 + 4,
        ),
      );
    }
  }
  return buildGreedyVertexColoring(input.vertexCount, conflictGroups);
}

function finiteF32(value: number, label: string): number {
  const rounded = Math.fround(value);
  if (!Number.isFinite(value) || !Number.isFinite(rounded)) {
    throw new RangeError(`${label} must be finite after f32 conversion.`);
  }
  return rounded;
}

function validateObjectiveArrayLength(
  values: Float32Array,
  expectedLength: number,
  label: string,
): void {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `${label} must contain ${expectedLength} values; got ${values.length}.`,
    );
  }
}

function createInitializedBuffer(
  device: GPUDevice,
  label: string,
  usage: GPUBufferUsageFlags,
  data: ArrayBufferView,
  createdBuffers: GPUBuffer[],
): GPUBuffer {
  const size = Math.max(4, alignTo4(data.byteLength));
  const buffer = device.createBuffer({
    label,
    size,
    usage,
    mappedAtCreation: true,
  });
  createdBuffers.push(buffer);
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function assertStorageBufferFits(
  device: GPUDevice,
  label: string,
  byteLength: number,
): void {
  if (byteLength > device.limits.maxStorageBufferBindingSize) {
    throw new RangeError(
      `${label} needs ${byteLength} bytes, exceeding this adapter's ` +
        `maxStorageBufferBindingSize of ${device.limits.maxStorageBufferBindingSize}.`,
    );
  }
  if (byteLength > device.limits.maxBufferSize) {
    throw new RangeError(
      `${label} needs ${byteLength} bytes, exceeding this adapter's ` +
        `maxBufferSize of ${device.limits.maxBufferSize}.`,
    );
  }
}

function validateStepSettings(settings: JGS2StepSettings): void {
  if (
    settings.schedule !== "jacobi" &&
    settings.schedule !== "graph-colored-gauss-seidel"
  ) {
    throw new RangeError(`Unknown JGS2 schedule: ${String(settings.schedule)}.`);
  }
  const finiteValues: ReadonlyArray<readonly [string, number]> = [
    ["timestep", settings.timestep],
    ["iterations", settings.iterations],
    ["floorHeight", settings.floorHeight],
    ["floorStiffness", settings.floorStiffness],
    ["velocityDamping", settings.velocityDamping],
    ["regularization", settings.regularization],
    ["rotationEpsilon", settings.rotationEpsilon],
    ["maxStep", settings.maxStep],
    ["residualTolerance", settings.residualTolerance],
    ["normalizedUpdateTolerance", settings.normalizedUpdateTolerance],
    ["contactTangentialDamping", settings.contactTangentialDamping],
    ["contactMargin", settings.contactMargin],
    ["ipcActivationDistance", settings.ipcActivationDistance],
    ["ipcMinimumDistance", settings.ipcMinimumDistance],
    ["ipcBarrierStiffness", settings.ipcBarrierStiffness],
    ["ipcFrictionCoefficient", settings.ipcFrictionCoefficient],
    ["ipcFrictionVelocityEpsilon", settings.ipcFrictionVelocityEpsilon],
    ["ipcStepSafety", settings.ipcStepSafety],
    ["gravity.x", settings.gravity[0]],
    ["gravity.y", settings.gravity[1]],
    ["gravity.z", settings.gravity[2]],
  ];
  for (const [name, value] of finiteValues) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${name} must be finite; got ${value}.`);
    }
  }
  if (!(settings.timestep > 0)) {
    throw new RangeError("timestep must be positive.");
  }
  if (settings.floorStiffness < 0) {
    throw new RangeError("floorStiffness must be nonnegative.");
  }
  if (settings.velocityDamping < 0 || settings.velocityDamping > 1) {
    throw new RangeError("velocityDamping must be between zero and one.");
  }
  if (!(settings.regularization > 0)) {
    throw new RangeError("regularization must be positive.");
  }
  if (!(settings.rotationEpsilon > 0)) {
    throw new RangeError("rotationEpsilon must be positive.");
  }
  if (settings.maxStep < 0) {
    throw new RangeError("maxStep must be nonnegative.");
  }
  for (const [name, value] of [
    ["residualTolerance", settings.residualTolerance],
    ["normalizedUpdateTolerance", settings.normalizedUpdateTolerance],
  ] as const) {
    if (
      !(value > 0) ||
      value > JGS2_GLOBALIZATION_MAX_RUNTIME_TOLERANCE
    ) {
      throw new RangeError(
        `${name} must be positive and no greater than ` +
          `${JGS2_GLOBALIZATION_MAX_RUNTIME_TOLERANCE}.`,
      );
    }
  }
  validateJGS2ContactParameters(
    settings.contactTangentialDamping,
    settings.contactMargin,
  );
  if (!(settings.ipcActivationDistance > 0)) {
    throw new RangeError("ipcActivationDistance must be positive.");
  }
  if (
    settings.ipcMinimumDistance < 0 ||
    settings.ipcMinimumDistance >= settings.ipcActivationDistance
  ) {
    throw new RangeError(
      "ipcMinimumDistance must be nonnegative and below ipcActivationDistance.",
    );
  }
  if (settings.ipcBarrierStiffness < 0) {
    throw new RangeError("ipcBarrierStiffness must be nonnegative.");
  }
  if (settings.ipcFrictionCoefficient < 0) {
    throw new RangeError("ipcFrictionCoefficient must be nonnegative.");
  }
  if (!(settings.ipcFrictionVelocityEpsilon > 0)) {
    throw new RangeError("ipcFrictionVelocityEpsilon must be positive.");
  }
  if (!(settings.ipcStepSafety > 0) || !(settings.ipcStepSafety < 1)) {
    throw new RangeError("ipcStepSafety must be strictly between zero and one.");
  }
  if (typeof settings.horizontalBodyCorrection !== "boolean") {
    throw new TypeError("horizontalBodyCorrection must be a boolean.");
  }
  if (typeof settings.parityMode !== "boolean") {
    throw new TypeError("parityMode must be a boolean.");
  }
}

export function resolveJGS2StepSettings(
  defaults: JGS2StepSettings,
  overrides: Partial<JGS2StepSettings>,
): JGS2StepSettings {
  let merged: JGS2StepSettings = {
    ...defaults,
    ...overrides,
    gravity: overrides.gravity ?? defaults.gravity,
  };
  if (merged.parityMode) {
    merged = {
      ...merged,
      velocityDamping: 1,
      contactTangentialDamping: 0,
      horizontalBodyCorrection: false,
    };
  }
  validateStepSettings(merged);
  return merged;
}

function validateExactIterationCount(iterations: number): void {
  if (!Number.isSafeInteger(iterations) || iterations < 1) {
    throw new RangeError(
      `iterations must be a positive safe integer; got ${iterations}.`,
    );
  }
}

function validateFrameBatchCount(frameCount: number): void {
  if (
    !Number.isSafeInteger(frameCount) ||
    frameCount < 1 ||
    frameCount > JGS2_MAX_BATCH_FRAMES
  ) {
    throw new RangeError(
      `frameCount must be a positive safe integer no greater than ` +
        `${JGS2_MAX_BATCH_FRAMES}; got ${frameCount}.`,
    );
  }
}

function validateGlobalizedSubmissionWork(
  frameCount: number,
  iterations: number,
): void {
  if (
    iterations >
    Math.floor(JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION / frameCount)
  ) {
    throw new RangeError(
      `Stable JGS2 may encode at most ` +
        `${JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION} total nonlinear ` +
        `iterations per submission; got ${frameCount} frames * ` +
        `${iterations} iterations.`,
    );
  }
}

function validateTimestampWrites(writes: GpuTimestampIntervalWrites): void {
  if (
    !Number.isSafeInteger(writes.startWriteIndex) ||
    !Number.isSafeInteger(writes.endWriteIndex) ||
    writes.startWriteIndex < 0 ||
    writes.endWriteIndex < 0 ||
    writes.startWriteIndex === writes.endWriteIndex
  ) {
    throw new RangeError(
      "GPU timestamp write indices must be distinct nonnegative safe integers.",
    );
  }
}

function packUniforms(
  input: Pick<JGS2GpuInput, "vertexCount" | "tetCount" | "cubatureK"> & {
    readonly bodyCount: number;
  },
  offsets: JGS2DynamicOffsets,
  settings: JGS2StepSettings,
  sourcePosition: number,
  targetPosition: number,
  globalizationEnabled: boolean,
  stopAfterConvergence: boolean,
  sceneScale: number,
  objectiveFlags: number,
  activeColor: number | null = null,
): Uint8Array {
  const buffer = new ArrayBuffer(JGS2_UNIFORM_BYTES);
  const integers = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);

  let packedBodyCount = input.bodyCount;
  if (activeColor !== null) {
    if (input.bodyCount > JGS2_COLORED_SCHEDULE_BODY_MASK) {
      throw new RangeError(
        "Colored Gauss-Seidel supports at most 65,535 small-scene bodies.",
      );
    }
    if (
      !Number.isSafeInteger(activeColor) ||
      activeColor < 0 ||
      activeColor > JGS2_COLORED_SCHEDULE_MAX_COLOR
    ) {
      throw new RangeError(
        `Colored Gauss-Seidel active color must be from 0 through ${JGS2_COLORED_SCHEDULE_MAX_COLOR}.`,
      );
    }
    packedBodyCount =
      (JGS2_COLORED_SCHEDULE_FLAG |
        (activeColor << 16) |
        input.bodyCount) >>> 0;
  }
  integers.set(
    [input.vertexCount, input.tetCount, input.cubatureK, packedBodyCount],
    0,
  );
  integers.set(
    [offsets.posA, offsets.posB, offsets.predicted, offsets.velocity],
    4,
  );
  integers.set(
    [offsets.old, offsets.vertexRotation, offsets.tetRotation, sourcePosition],
    8,
  );
  integers.set(
    [
      targetPosition,
      offsets.bodyCorrection,
      offsets.finalUpdate,
      objectiveFlags,
    ],
    12,
  );

  const inverseTimestep = 1 / settings.timestep;
  floats.set(
    [
      settings.timestep,
      inverseTimestep,
      inverseTimestep * inverseTimestep,
      settings.maxStep,
    ],
    16,
  );
  floats.set(
    [
      settings.gravity[0],
      settings.gravity[1],
      settings.gravity[2],
      settings.floorHeight,
    ],
    20,
  );
  floats.set(
    [
      settings.floorStiffness,
      settings.regularization,
      settings.rotationEpsilon,
      settings.velocityDamping,
    ],
    24,
  );
  floats.set(
    [
      settings.contactTangentialDamping,
      settings.contactMargin,
      settings.ipcActivationDistance,
      settings.ipcBarrierStiffness,
    ],
    28,
  );
  integers.set(
    [
      offsets.localGlobalization,
      offsets.tetGlobalization,
      offsets.assembledVertexEnergy,
      offsets.convergenceGradient,
    ],
    32,
  );
  integers.set(
    [
      offsets.globalizationControl,
      offsets.globalizationHistory,
      JGS2_GLOBALIZATION_HISTORY_CAPACITY,
      Number(globalizationEnabled),
    ],
    36,
  );
  floats.set(
    [
      settings.residualTolerance,
      settings.normalizedUpdateTolerance,
      sceneScale,
      Number(stopAfterConvergence),
    ],
    40,
  );
  return new Uint8Array(buffer);
}

export function computeJGS2InitialDynamicSceneScale(
  input: Pick<JGS2GpuInput, "vertexCount" | "positions" | "vertexInfo">,
): number {
  let minimumX = Number.POSITIVE_INFINITY;
  let minimumY = Number.POSITIVE_INFINITY;
  let minimumZ = Number.POSITIVE_INFINITY;
  let maximumX = Number.NEGATIVE_INFINITY;
  let maximumY = Number.NEGATIVE_INFINITY;
  let maximumZ = Number.NEGATIVE_INFINITY;
  let dynamicVertexCount = 0;
  for (let vertex = 0; vertex < input.vertexCount; vertex += 1) {
    const base = vertex * 4;
    if (input.vertexInfo[base + 2] !== 0) {
      continue;
    }
    dynamicVertexCount += 1;
    const x = input.positions[base]!;
    const y = input.positions[base + 1]!;
    const z = input.positions[base + 2]!;
    minimumX = Math.min(minimumX, x);
    minimumY = Math.min(minimumY, y);
    minimumZ = Math.min(minimumZ, z);
    maximumX = Math.max(maximumX, x);
    maximumY = Math.max(maximumY, y);
    maximumZ = Math.max(maximumZ, z);
  }
  const diagonal = Math.hypot(
    maximumX - minimumX,
    maximumY - minimumY,
    maximumZ - minimumZ,
  );
  return dynamicVertexCount > 0 && Number.isFinite(diagonal) && diagonal > 0
    ? diagonal
    : JGS2_DEGENERATE_DYNAMIC_SCENE_SCALE;
}

function createUniformBuffer(device: GPUDevice, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: JGS2_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function createBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  buffers: JGS2GpuBuffers,
  uniforms: GPUBuffer,
  label: string,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout,
    entries: [
      { binding: 0, resource: { buffer: buffers.dynamic } },
      { binding: 1, resource: { buffer: buffers.vertices } },
      { binding: 2, resource: { buffer: buffers.tets } },
      { binding: 3, resource: { buffer: buffers.stiffness } },
      { binding: 4, resource: { buffer: buffers.adjacency } },
      { binding: 5, resource: { buffer: buffers.cubature } },
      { binding: 6, resource: { buffer: buffers.objectives } },
      { binding: 7, resource: { buffer: uniforms } },
    ],
  });
}

function encodeDispatch(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  itemCount: number,
  label: string,
  timestampWrites?: GPUComputePassTimestampWrites,
): void {
  if (itemCount === 0) {
    return;
  }
  const pass = encoder.beginComputePass({
    label,
    ...(timestampWrites ? { timestampWrites } : {}),
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(itemCount / JGS2_WORKGROUP_SIZE));
  pass.end();
}

async function validateGpuInitialStableSource(
  device: GPUDevice,
  buffers: JGS2GpuBuffers,
  pipelines: Pick<
    Required<JGS2Pipelines>,
    "candidateTetrahedron" | "reduceCandidate"
  >,
  uniforms: JGS2UniformState,
  offsets: JGS2DynamicOffsets,
  globalizationElementCount: number,
): Promise<void> {
  const readback = device.createBuffer({
    label: "jgs2-initial-stable-source-readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({
      label: "jgs2-initial-stable-source-command-encoder",
    });
    encodeDispatch(
      encoder,
      pipelines.candidateTetrahedron,
      uniforms.baseBindGroup,
      globalizationElementCount,
      "jgs2-initial-stable-source-element-pass",
    );
    encodeDispatch(
      encoder,
      pipelines.reduceCandidate,
      uniforms.baseBindGroup,
      1,
      "jgs2-initial-stable-source-reduction-pass",
    );
    encoder.copyBufferToBuffer(
      buffers.dynamic,
      offsets.globalizationControl * 16,
      readback,
      0,
      16,
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const control = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    const sourceMinimum = control[0]!;
    const candidateMinimum = control[1]!;
    const acceptedMinimum = control[2]!;
    const requiredValidityBits =
      JGS2_GLOBALIZATION_SOURCE_GEOMETRY_VALID_BIT |
      JGS2_GLOBALIZATION_CANDIDATE_GEOMETRY_VALID_BIT |
      JGS2_GLOBALIZATION_ACCEPTED_MINIMUM_VALID_BIT;
    const validityBits = control[3]!;
    if (
      ![sourceMinimum, candidateMinimum, acceptedMinimum].every(
        Number.isFinite,
      ) ||
      !(sourceMinimum > JGS2_GLOBALIZATION_DETERMINANT_FLOOR) ||
      !(candidateMinimum > JGS2_GLOBALIZATION_DETERMINANT_FLOOR) ||
      !(acceptedMinimum > JGS2_GLOBALIZATION_DETERMINANT_FLOOR) ||
      !Number.isSafeInteger(validityBits) ||
      (validityBits & requiredValidityBits) !== requiredValidityBits
    ) {
      throw new RangeError(
        "The uploaded stable Neo-Hookean source pose is not feasible under " +
          "the production GPU f32 determinant calculation; " +
          `source J=${sourceMinimum}, candidate J=${candidateMinimum}, ` +
          `accepted J=${acceptedMinimum}.`,
      );
    }
  } finally {
    readback.destroy();
  }
}

function formatCompilationErrors(messages: readonly GPUCompilationMessage[]): string {
  return messages
    .filter((message) => message.type === "error")
    .map(
      (message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`,
    )
    .join("\n");
}

export class JGS2GpuSolver {
  readonly vertexCount: number;
  readonly tetCount: number;
  readonly cubatureK: number;
  readonly bodyCount: number;
  readonly contactCandidateCount: number;
  readonly clothTriangleCount: number;
  readonly scheduleColorCount: number;
  readonly globalizationElementCount: number;
  readonly dynamicOffsets: JGS2DynamicOffsets;
  /** True for the stable-material production path governed by Phase 1 policy. */
  readonly globalizationEnabled: boolean;

  private destroyed = false;
  private submittedIterations = 0;
  private submittedSettings: JGS2StepSettings;
  private oracleEvaluator?: JGS2GpuOracleEvaluator;
  private oracleEvaluatorCreation?: Promise<JGS2GpuOracleEvaluator>;
  private readonly timestampTimer: GpuTimestampFrameTimer;
  private diagnosticReadbackCount = 0;
  private globalizationRecordsAvailable = false;
  private objectiveFlagsValue: number;
  private activeExternalForceCount: number;
  private activeTargetCount: number;
  private objectiveRevisionValue = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly inputShape: Pick<
      JGS2GpuInput,
      "vertexCount" | "tetCount" | "cubatureK"
    > & { readonly bodyCount: number },
    private readonly buffers: JGS2GpuBuffers,
    private readonly pipelines: JGS2Pipelines,
    private readonly uniforms: JGS2UniformState,
    private readonly defaultSettings: JGS2StepSettings,
    private readonly preprocessingTimestep: number,
    private readonly sceneScale: number,
    private readonly objectiveData: Float32Array,
    private readonly pinnedVertices: Uint8Array,
    initialObjectiveActivity: PackedObjectiveActivity,
    contactCandidateCount: number,
    clothTriangleCount: number,
    scheduleColorCount: number,
    globalizationEnabled: boolean,
    offsets: JGS2DynamicOffsets,
  ) {
    this.vertexCount = inputShape.vertexCount;
    this.tetCount = inputShape.tetCount;
    this.cubatureK = inputShape.cubatureK;
    this.bodyCount = inputShape.bodyCount;
    this.contactCandidateCount = contactCandidateCount;
    this.clothTriangleCount = clothTriangleCount;
    this.scheduleColorCount = scheduleColorCount;
    this.globalizationElementCount = this.tetCount + clothTriangleCount;
    this.dynamicOffsets = offsets;
    this.globalizationEnabled = globalizationEnabled;
    this.objectiveFlagsValue = initialObjectiveActivity.flags;
    this.activeExternalForceCount =
      initialObjectiveActivity.externalForceVertexCount;
    this.activeTargetCount =
      initialObjectiveActivity.quadraticTargetVertexCount;
    this.submittedSettings = defaultSettings;
    this.timestampTimer = new GpuTimestampFrameTimer(device, "jgs2-step");
  }

  static async create(
    device: GPUDevice,
    input: JGS2GpuInput,
    settings: Partial<JGS2StepSettings> = {},
  ): Promise<JGS2GpuSolver> {
    validateJGS2GpuInput(input);
    if (device.limits.maxStorageBuffersPerShaderStage < 7) {
      throw new Error(
        "JGS2 requires seven storage buffers " +
          "in the compute stage, but this " +
          `adapter supports ${device.limits.maxStorageBuffersPerShaderStage}.`,
      );
    }

    const materialMode = inferJGS2MaterialMode(input);
    const globalizationEnabled = materialMode === "stable-neo-hookean";
    const vertexColoring = buildJGS2InputVertexColoring(input);
    if (vertexColoring.colorCount > JGS2_COLORED_SCHEDULE_MAX_COLOR + 1) {
      throw new RangeError(
        `Colored Gauss-Seidel supports at most ${JGS2_COLORED_SCHEDULE_MAX_COLOR + 1} colors; ` +
          `the small-scene conflict graph needs ${vertexColoring.colorCount}.`,
      );
    }
    const clothTriangleCount = input.cloth
      ? input.cloth.triangleIndices.length / 3
      : 0;
    const globalizationElementCount = input.tetCount + clothTriangleCount;
    if (clothTriangleCount > 0 && !globalizationEnabled) {
      throw new RangeError(
        "Triangle cloth requires the stable Neo-Hookean production path.",
      );
    }
    const contactCandidates =
      input.contactCandidates ?? EMPTY_IPC_CONTACT_CANDIDATES;
    const packedContacts = packIpcContactBuffer(contactCandidates);
    const packedContactIncidence = packIpcIncidentCandidateAdjacency(
      input.vertexCount,
      contactCandidates,
    );
    const packedAdjacency = new Uint32Array(
      input.adjacency.length + packedContactIncidence.words.length,
    );
    packedAdjacency.set(input.adjacency);
    packedAdjacency.set(packedContactIncidence.words, input.adjacency.length);
    if (!globalizationEnabled && packedContacts.candidateCount !== 0) {
      throw new RangeError(
        "IPC contact candidates require the stable Neo-Hookean production path.",
      );
    }
    const packedObjectives = packJGS2VertexObjectives(input);
    const initialObjectiveActivity = computePackedObjectiveActivity(
      packedObjectives,
      input.vertexCount,
    );
    if (!globalizationEnabled && initialObjectiveActivity.flags !== 0) {
      throw new RangeError(
        "Active external forces and quadratic targets require the stable " +
          "Neo-Hookean production path.",
      );
    }
    const pinnedVertices = packPinnedVertexMask(input);
    // Legacy regression shaders still bind the objective slot, but inactive
    // scenes should not pay for a full per-vertex allocation they never read.
    const uploadedObjectives = globalizationEnabled
      ? packedObjectives
      : new Float32Array(JGS2_VERTEX_OBJECTIVE_WORDS);
    let resolvedSettings = resolveJGS2StepSettings(
      DEFAULT_JGS2_STEP_SETTINGS,
      settings,
    );
    if (packedContacts.candidateCount > 0) {
      const minimumContactDistance = minimumStaticIpcContactDistance(
        input.positions,
        input.contactCandidates!,
        4,
      );
      const contactSourceMargin =
        8 *
        F32_UNIT_ROUNDOFF *
        Math.max(
          1,
          resolvedSettings.ipcActivationDistance,
          Math.abs(minimumContactDistance),
        );
      const requiredContactDistance =
        resolvedSettings.ipcMinimumDistance + contactSourceMargin;
      if (
        !Number.isFinite(minimumContactDistance) ||
        !(minimumContactDistance > requiredContactDistance)
      ) {
        throw new RangeError(
          "IPC candidates require an initially feasible source pose with " +
            `distance greater than ${requiredContactDistance} (minimum plus ` +
            "an f32 safety margin); " +
            `got ${minimumContactDistance}.`,
        );
      }
    }
    if (globalizationEnabled) {
      const minimumDeterminant =
        minimumJGS2InputDeformationDeterminant(input);
      if (
        !Number.isFinite(minimumDeterminant) ||
        !(minimumDeterminant > JGS2_GLOBALIZATION_DETERMINANT_FLOOR)
      ) {
        throw new RangeError(
          "A stable Neo-Hookean production solve requires an initially " +
            `feasible pose with minimum J greater than ` +
            `${JGS2_GLOBALIZATION_DETERMINANT_FLOOR}; got ` +
            `${minimumDeterminant}.`,
        );
      }
    }
    const bodyCount = inferJGS2BodyCount(input.vertexInfo, input.vertexCount);
    const offsets = computeJGS2DynamicOffsets(
      input.vertexCount,
      input.tetCount,
      bodyCount,
      globalizationEnabled,
      packedContacts.byteLength / 16,
      globalizationElementCount,
    );
    const dynamic = packJGS2InitialDynamic(input, offsets);
    new Uint32Array(dynamic.buffer).set(
      packedContacts.integers,
      offsets.ipcContact * 4,
    );
    new Float32Array(dynamic.buffer).set(
      [
        resolvedSettings.ipcMinimumDistance,
        resolvedSettings.ipcFrictionCoefficient,
        resolvedSettings.ipcFrictionVelocityEpsilon,
        resolvedSettings.ipcStepSafety,
      ],
      (offsets.ipcContact + 1) * 4,
    );
    const vertices = packJGS2VertexStatic(input);
    const tets = packJGS2TetStatic(input);
    const cubaturePrefix = packJGS2Cubature(input);
    const packedCloth = input.cloth
      ? packJGS2GpuClothArena(input.cloth)
      : undefined;
    const clothArena = packedCloth?.integers ??
      new Uint32Array(JGS2_CLOTH_GLOBAL_WORDS);
    const packedSchedule = packJGS2ScheduleArena(
      vertexColoring.colors,
      vertexColoring.colorCount,
    );
    const cubature = new Uint32Array(
      cubaturePrefix.length + clothArena.length + packedSchedule.integers.length,
    );
    cubature.set(cubaturePrefix);
    cubature.set(clothArena, cubaturePrefix.length);
    cubature.set(
      packedSchedule.integers,
      cubaturePrefix.length + clothArena.length,
    );

    const storageSizes: ReadonlyArray<readonly [string, number]> = [
      ["JGS2 dynamic buffer", dynamic.byteLength],
      ["JGS2 vertex buffer", vertices.byteLength],
      ["JGS2 tetrahedron buffer", tets.byteLength],
      ["JGS2 stiffness buffer", input.tetRestStiffness.byteLength],
      ["JGS2 adjacency buffer", Math.max(4, packedAdjacency.byteLength)],
      ["JGS2 cubature buffer", Math.max(4, cubature.byteLength)],
      ["JGS2 objective buffer", uploadedObjectives.byteLength],
    ];
    for (const [label, byteLength] of storageSizes) {
      assertStorageBufferFits(device, label, byteLength);
    }

    const storageBuffers: GPUBuffer[] = [];
    const uniformBuffers: GPUBuffer[] = [];
    try {
      const buffers: JGS2GpuBuffers = {
        dynamic: createInitializedBuffer(
          device,
          "jgs2-dynamic",
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.VERTEX,
          dynamic,
          storageBuffers,
        ),
        vertices: createInitializedBuffer(
          device,
          "jgs2-vertices",
          GPUBufferUsage.STORAGE,
          vertices,
          storageBuffers,
        ),
        tets: createInitializedBuffer(
          device,
          "jgs2-tetrahedra",
          GPUBufferUsage.STORAGE,
          tets,
          storageBuffers,
        ),
        stiffness: createInitializedBuffer(
          device,
          "jgs2-rest-stiffness",
          GPUBufferUsage.STORAGE,
          input.tetRestStiffness,
          storageBuffers,
        ),
        adjacency: createInitializedBuffer(
          device,
          "jgs2-adjacency",
          GPUBufferUsage.STORAGE,
          packedAdjacency,
          storageBuffers,
        ),
        cubature: createInitializedBuffer(
          device,
          "jgs2-cubature",
          GPUBufferUsage.STORAGE,
          cubature,
          storageBuffers,
        ),
        objectives: createInitializedBuffer(
          device,
          "jgs2-objectives",
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          uploadedObjectives,
          storageBuffers,
        ),
      };
      const bindGroupLayout = device.createBindGroupLayout({
        label: "jgs2-bind-group-layout",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          ...[1, 2, 3].map<GPUBindGroupLayoutEntry>((binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          })),
          {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          ...[5, 6].map<GPUBindGroupLayoutEntry>((binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          })),
          {
            binding: 7,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform", minBindingSize: JGS2_UNIFORM_BYTES },
          },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({
        label: "jgs2-pipeline-layout",
        bindGroupLayouts: [bindGroupLayout],
      });
      const shader = device.createShaderModule({
        label: "jgs2-compute-shader",
        code: jgs2Shader,
      });
      const compilation = await shader.getCompilationInfo();
      const compilationErrors = formatCompilationErrors(compilation.messages);
      if (compilationErrors) {
        throw new Error(`JGS2 WGSL failed to compile:\n${compilationErrors}`);
      }

      const createPipeline = (entryPoint: string, label: string) =>
        device.createComputePipelineAsync({
          label,
          layout: pipelineLayout,
          compute: { module: shader, entryPoint },
        });
      const [
        predict,
        tetPolarRotation,
        vertexPolarRotation,
        solve,
        copyPosition,
        bodyHorizontalCorrection,
        applyBodyHorizontalCorrection,
        finalize,
      ] = await Promise.all([
        createPipeline("predict", "jgs2-predict-pipeline"),
        createPipeline("tetPolarRotation", "jgs2-tet-polar-pipeline"),
        createPipeline("vertexPolarRotation", "jgs2-vertex-polar-pipeline"),
        createPipeline("jgs2Solve", "jgs2-solve-pipeline"),
        createPipeline("copyPosition", "jgs2-copy-position-pipeline"),
        createPipeline(
          "bodyHorizontalCorrection",
          "jgs2-body-horizontal-correction-pipeline",
        ),
        createPipeline(
          "applyBodyHorizontalCorrection",
          "jgs2-apply-body-horizontal-correction-pipeline",
        ),
        createPipeline("finalize", "jgs2-finalize-pipeline"),
      ]);
      const globalizationPipelines = globalizationEnabled
        ? await Promise.all([
            createPipeline(
              "assembledVertexEnergy",
              "jgs2-assembled-vertex-energy-pipeline",
            ),
            createPipeline(
              "candidateTetrahedron",
              "jgs2-candidate-tetrahedron-pipeline",
            ),
            createPipeline(
              "reduceCandidate",
              "jgs2-reduce-candidate-pipeline",
            ),
            createPipeline(
              "applyCandidate",
              "jgs2-apply-candidate-pipeline",
            ),
            createPipeline(
              "convergenceGradient",
              "jgs2-convergence-gradient-pipeline",
            ),
            createPipeline(
              "reduceConvergence",
              "jgs2-reduce-convergence-pipeline",
            ),
          ])
        : undefined;
      const contactPipelines =
        globalizationEnabled && packedContacts.candidateCount > 0
          ? await Promise.all([
              createPipeline("lagContact", "jgs2-ipc-lag-contact-pipeline"),
              createPipeline(
                "buildActiveContactRows",
                "jgs2-ipc-build-active-contact-rows-pipeline",
              ),
              createPipeline(
                "promoteAcceptedContactCache",
                "jgs2-ipc-promote-accepted-contact-cache-pipeline",
              ),
              createPipeline(
                "candidateContactStep",
                "jgs2-ipc-candidate-step-pipeline",
              ),
              createPipeline(
                "validateContactStep",
                "jgs2-ipc-validate-step-pipeline",
              ),
              createPipeline(
                "reduceContactStep",
                "jgs2-ipc-reduce-step-pipeline",
              ),
              createPipeline(
                "applyContactStep",
                "jgs2-ipc-apply-step-pipeline",
              ),
            ])
          : undefined;
      const pipelines: JGS2Pipelines = {
        predict,
        tetPolarRotation,
        vertexPolarRotation,
        solve,
        ...(globalizationPipelines
          ? {
              assembledVertexEnergy: globalizationPipelines[0],
              candidateTetrahedron: globalizationPipelines[1],
              reduceCandidate: globalizationPipelines[2],
              applyCandidate: globalizationPipelines[3],
              convergenceGradient: globalizationPipelines[4],
              reduceConvergence: globalizationPipelines[5],
            }
          : {}),
        ...(contactPipelines
          ? {
              lagContact: contactPipelines[0],
              buildActiveContactRows: contactPipelines[1],
              promoteAcceptedContactCache: contactPipelines[2],
              candidateContactStep: contactPipelines[3],
              validateContactStep: contactPipelines[4],
              reduceContactStep: contactPipelines[5],
              applyContactStep: contactPipelines[6],
            }
          : {}),
        copyPosition,
        bodyHorizontalCorrection,
        applyBodyHorizontalCorrection,
        finalize,
      };

      const base = createUniformBuffer(device, "jgs2-uniform-base");
      uniformBuffers.push(base);
      const fromBToA = createUniformBuffer(device, "jgs2-uniform-b-to-a");
      uniformBuffers.push(fromBToA);
      const fromAToB = createUniformBuffer(device, "jgs2-uniform-a-to-b");
      uniformBuffers.push(fromAToB);
      const coloredAToA = Array.from(
        { length: vertexColoring.colorCount },
        (_, color) => {
          const buffer = createUniformBuffer(
            device,
            `jgs2-uniform-colored-a-to-a-${color}`,
          );
          uniformBuffers.push(buffer);
          return buffer;
        },
      );
      const coloredBToB = Array.from(
        { length: vertexColoring.colorCount },
        (_, color) => {
          const buffer = createUniformBuffer(
            device,
            `jgs2-uniform-colored-b-to-b-${color}`,
          );
          uniformBuffers.push(buffer);
          return buffer;
        },
      );
      const uniforms: JGS2UniformState = {
        base,
        fromBToA,
        fromAToB,
        baseBindGroup: createBindGroup(
          device,
          bindGroupLayout,
          buffers,
          base,
          "jgs2-base-bind-group",
        ),
        fromBToABindGroup: createBindGroup(
          device,
          bindGroupLayout,
          buffers,
          fromBToA,
          "jgs2-b-to-a-bind-group",
        ),
        fromAToBBindGroup: createBindGroup(
          device,
          bindGroupLayout,
          buffers,
          fromAToB,
          "jgs2-a-to-b-bind-group",
        ),
        coloredAToA,
        coloredBToB,
        coloredAToABindGroups: coloredAToA.map((uniform, color) =>
          createBindGroup(
            device,
            bindGroupLayout,
            buffers,
            uniform,
            `jgs2-colored-a-to-a-bind-group-${color}`,
          ),
        ),
        coloredBToBBindGroups: coloredBToB.map((uniform, color) =>
          createBindGroup(
            device,
            bindGroupLayout,
            buffers,
            uniform,
            `jgs2-colored-b-to-b-bind-group-${color}`,
          ),
        ),
      };

      const inputShape = {
        vertexCount: input.vertexCount,
        tetCount: input.tetCount,
        cubatureK: input.cubatureK,
        bodyCount,
      };
      const sceneScale = computeJGS2InitialDynamicSceneScale(input);
      if (globalizationEnabled) {
        device.queue.writeBuffer(
          base,
          0,
          packUniforms(
            inputShape,
            offsets,
            resolvedSettings,
            offsets.posA,
            offsets.posA,
            true,
            false,
            sceneScale,
            initialObjectiveActivity.flags,
          ),
        );
        await validateGpuInitialStableSource(
          device,
          buffers,
          {
            candidateTetrahedron: pipelines.candidateTetrahedron!,
            reduceCandidate: pipelines.reduceCandidate!,
          },
          uniforms,
          offsets,
          globalizationElementCount,
        );
      }

      return new JGS2GpuSolver(
        device,
        inputShape,
        buffers,
        pipelines,
        uniforms,
        resolvedSettings,
        resolvedSettings.timestep,
        sceneScale,
        uploadedObjectives,
        pinnedVertices,
        initialObjectiveActivity,
        packedContacts.candidateCount,
        clothTriangleCount,
        vertexColoring.colorCount,
        globalizationEnabled,
        offsets,
      );
    } catch (error) {
      for (const uniform of uniformBuffers) {
        uniform.destroy();
      }
      for (const buffer of storageBuffers) {
        buffer.destroy();
      }
      throw error;
    }
  }

  get currentPositionBuffer(): GPUBuffer {
    this.assertUsable();
    return this.buffers.dynamic;
  }

  get currentPositionByteOffset(): number {
    return this.dynamicOffsets.posA * 16;
  }

  get currentPositionByteLength(): number {
    return this.vertexCount * 16;
  }

  get velocityByteOffset(): number {
    return this.dynamicOffsets.velocity * 16;
  }

  get velocityByteLength(): number {
    return this.vertexCount * 16;
  }

  get currentPositionView(): JGS2PositionBufferView {
    return this.positionView(this.dynamicOffsets.posA);
  }

  /** Predicted inertial position for the most recently submitted frame. */
  get predictedPositionView(): JGS2PositionBufferView {
    this.assertUsable();
    return this.positionView(this.dynamicOffsets.predicted);
  }

  /** Position at the start of the most recently submitted frame. */
  get oldPositionView(): JGS2PositionBufferView {
    this.assertUsable();
    return this.positionView(this.dynamicOffsets.old);
  }

  /** Configured iteration budget, distinct from stable historyCount after a stop. */
  get lastSubmittedIterationCount(): number {
    return this.submittedIterations;
  }

  get defaultStepSettings(): JGS2StepSettings {
    return this.defaultSettings;
  }

  get lastSubmittedSettings(): JGS2StepSettings {
    return this.submittedSettings;
  }

  /** Current CPU-mirrored activity bits uploaded with the next frame. */
  get objectiveActivity(): JGS2ObjectiveActivity {
    this.assertUsable();
    const flags = this.objectiveFlagsValue;
    return Object.freeze({
      flags,
      externalForces:
        (flags & JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT) !== 0,
      quadraticTargets:
        (flags & JGS2_OBJECTIVE_TARGET_ACTIVE_BIT) !== 0,
      externalForceVertexCount: this.activeExternalForceCount,
      quadraticTargetVertexCount: this.activeTargetCount,
    });
  }

  /** Number of successful objective-buffer writes since solver creation. */
  get objectiveRevision(): number {
    this.assertUsable();
    return this.objectiveRevisionValue;
  }

  /** Replace one vertex's constant external-force vector. */
  setExternalForce(
    vertex: number,
    force: readonly [number, number, number],
  ): void {
    this.assertObjectiveUpdatesSupported();
    this.assertObjectiveVertex(vertex);
    const packed = new Float32Array(4);
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      packed[coordinate] = finiteF32(
        force[coordinate],
        `External force ${coordinate} for vertex ${vertex}`,
      );
    }
    if (
      this.pinnedVertices[vertex] !== 0 &&
      (packed[0] !== 0 || packed[1] !== 0 || packed[2] !== 0)
    ) {
      throw new RangeError(
        `Pinned vertex ${vertex} cannot have a nonzero external force.`,
      );
    }
    const base = vertex * JGS2_VERTEX_OBJECTIVE_WORDS;
    const wasActive =
      this.objectiveData[base] !== 0 ||
      this.objectiveData[base + 1] !== 0 ||
      this.objectiveData[base + 2] !== 0;
    const isActive = packed[0] !== 0 || packed[1] !== 0 || packed[2] !== 0;
    this.device.queue.writeBuffer(
      this.buffers.objectives,
      vertex * JGS2_VERTEX_OBJECTIVE_BYTES,
      packed,
    );
    this.objectiveData.set(packed, base);
    this.activeExternalForceCount += Number(isActive) - Number(wasActive);
    this.finishObjectiveWrite();
  }

  /** Replace every external force in one validated, queue-ordered write. */
  replaceExternalForces(forces: Float32Array): void {
    this.assertObjectiveUpdatesSupported();
    validateObjectiveArrayLength(
      forces,
      this.vertexCount * 3,
      "External forces",
    );
    const next = this.objectiveData.slice();
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      const source = vertex * 3;
      const destination = vertex * JGS2_VERTEX_OBJECTIVE_WORDS;
      let nonzero = false;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const value = finiteF32(
          forces[source + coordinate]!,
          `External forces[${source + coordinate}]`,
        );
        next[destination + coordinate] = value;
        nonzero ||= value !== 0;
      }
      next[destination + 3] = 0;
      if (this.pinnedVertices[vertex] !== 0 && nonzero) {
        throw new RangeError(
          `Pinned vertex ${vertex} cannot have a nonzero external force.`,
        );
      }
    }
    this.writeCompleteObjectiveState(next);
  }

  clearExternalForces(): void {
    this.replaceExternalForces(new Float32Array(this.vertexCount * 3));
  }

  /** Set one soft isotropic target; stiffness zero is an inactive target. */
  setQuadraticTarget(
    vertex: number,
    target: readonly [number, number, number],
    stiffness: number,
  ): void {
    this.assertObjectiveUpdatesSupported();
    this.assertObjectiveVertex(vertex);
    const packed = new Float32Array(4);
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      packed[coordinate] = finiteF32(
        target[coordinate],
        `Quadratic target ${coordinate} for vertex ${vertex}`,
      );
    }
    packed[3] = finiteF32(
      stiffness,
      `Quadratic-target stiffness for vertex ${vertex}`,
    );
    if (packed[3] < 0) {
      throw new RangeError(
        `Quadratic-target stiffness for vertex ${vertex} must be nonnegative.`,
      );
    }
    if (this.pinnedVertices[vertex] !== 0 && packed[3] !== 0) {
      throw new RangeError(
        `Pinned vertex ${vertex} cannot have a nonzero quadratic-target stiffness.`,
      );
    }
    const base = vertex * JGS2_VERTEX_OBJECTIVE_WORDS + 4;
    const wasActive = this.objectiveData[base + 3]! > 0;
    const isActive = packed[3]! > 0;
    this.device.queue.writeBuffer(
      this.buffers.objectives,
      vertex * JGS2_VERTEX_OBJECTIVE_BYTES + 16,
      packed,
    );
    this.objectiveData.set(packed, base);
    this.activeTargetCount += Number(isActive) - Number(wasActive);
    this.finishObjectiveWrite();
  }

  /** Replace every soft target in one validated, queue-ordered write. */
  replaceQuadraticTargets(
    targetPositions: Float32Array,
    targetStiffnesses: Float32Array,
  ): void {
    this.assertObjectiveUpdatesSupported();
    validateObjectiveArrayLength(
      targetPositions,
      this.vertexCount * 3,
      "Quadratic target positions",
    );
    validateObjectiveArrayLength(
      targetStiffnesses,
      this.vertexCount,
      "Quadratic target stiffnesses",
    );
    const next = this.objectiveData.slice();
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      const source = vertex * 3;
      const destination = vertex * JGS2_VERTEX_OBJECTIVE_WORDS + 4;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        next[destination + coordinate] = finiteF32(
          targetPositions[source + coordinate]!,
          `Quadratic target positions[${source + coordinate}]`,
        );
      }
      const stiffness = finiteF32(
        targetStiffnesses[vertex]!,
        `Quadratic target stiffnesses[${vertex}]`,
      );
      if (stiffness < 0) {
        throw new RangeError(
          `Quadratic target stiffnesses[${vertex}] must be nonnegative.`,
        );
      }
      if (this.pinnedVertices[vertex] !== 0 && stiffness !== 0) {
        throw new RangeError(
          `Pinned vertex ${vertex} cannot have a nonzero quadratic-target stiffness.`,
        );
      }
      next[destination + 3] = stiffness;
    }
    this.writeCompleteObjectiveState(next);
  }

  /** Atomically release one target without changing position or velocity. */
  releaseQuadraticTarget(vertex: number): void {
    this.assertObjectiveUpdatesSupported();
    this.assertObjectiveVertex(vertex);
    const base = vertex * JGS2_VERTEX_OBJECTIVE_WORDS + 4;
    const wasActive = this.objectiveData[base + 3]! > 0;
    const released = this.objectiveData.slice(base, base + 4);
    released[3] = 0;
    this.device.queue.writeBuffer(
      this.buffers.objectives,
      vertex * JGS2_VERTEX_OBJECTIVE_BYTES + 16,
      released,
    );
    this.objectiveData.set(released, base);
    this.activeTargetCount -= Number(wasActive);
    this.finishObjectiveWrite();
  }

  releaseAllQuadraticTargets(): void {
    this.assertObjectiveUpdatesSupported();
    const next = this.objectiveData.slice();
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      next[vertex * JGS2_VERTEX_OBJECTIVE_WORDS + 7] = 0;
    }
    this.writeCompleteObjectiveState(next);
  }

  /** Number of explicit state/oracle diagnostic readbacks requested by tests. */
  get explicitDiagnosticReadbackCount(): number {
    return this.diagnosticReadbackCount;
  }

  step(overrides: Partial<JGS2StepSettings> = {}): void {
    this.assertUsable();
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but step() requested ${overrides.timestep}. Regenerate the ` +
          "precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    const iterations = normalizeOddIterationCount(settings.iterations);
    this.submitFrame(settings, iterations, true);
  }

  /**
   * Advance several identical-settings frames in one command buffer and one
   * queue submission. At most {@link JGS2_MAX_BATCH_FRAMES} may be encoded at
   * once. Stable-material submissions must also keep frameCount * iterations
   * within {@link JGS2_MAX_GLOBALIZED_ITERATIONS_PER_SUBMISSION}; chunk longer
   * offline/test runs at the tighter boundary.
   */
  stepFrames(
    frameCount: number,
    overrides: Partial<JGS2StepSettings> = {},
  ): void {
    this.assertUsable();
    validateFrameBatchCount(frameCount);
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepFrames() requested ${overrides.timestep}. Regenerate the ` +
          "precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    const iterations = normalizeOddIterationCount(settings.iterations);
    this.submitFrames(settings, iterations, frameCount, true);
  }

  /**
   * Advance one complete simulation frame using exactly the requested number
   * of nonlinear Jacobi iterations. Unlike step(), even counts are retained
   * and a stable convergence flag does not short-circuit later iterations.
   */
  stepExactIterations(
    iterations: number,
    overrides: Partial<JGS2StepSettings> = {},
  ): void {
    this.assertUsable();
    validateExactIterationCount(iterations);
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepExactIterations() requested ${overrides.timestep}. ` +
          "Regenerate the precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    this.submitFrame(settings, iterations, false);
  }

  /**
   * Batched counterpart to stepExactIterations(). Every encoded frame retains
   * the requested nonlinear iteration count, including even counts. The same
   * per-submission work bounds as stepFrames() apply.
   */
  stepFramesExactIterations(
    frameCount: number,
    iterations: number,
    overrides: Partial<JGS2StepSettings> = {},
  ): void {
    this.assertUsable();
    validateFrameBatchCount(frameCount);
    validateExactIterationCount(iterations);
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepFramesExactIterations() requested ${overrides.timestep}. ` +
          "Regenerate the precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    this.submitFrames(settings, iterations, frameCount, false);
  }

  /**
   * Advance one complete simulation frame and explicitly measure only its GPU
   * compute command stream. This test/benchmark-only path is never used by the
   * production animation loop.
   */
  async stepWithGpuTimestamp(
    overrides: Partial<JGS2StepSettings> = {},
  ): Promise<GpuTimestampMeasurement> {
    this.assertUsable();
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepWithGpuTimestamp() requested ${overrides.timestep}. ` +
          "Regenerate the precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    const iterations = normalizeOddIterationCount(settings.iterations);
    const measurement = await this.timestampTimer.measure((encoder) => {
      this.prepareFrame(settings, iterations, true);
      this.encodeFrame(encoder, settings, iterations);
    });
    this.markGlobalizationRecordsAvailable();
    return measurement;
  }

  /**
   * Submit one production-shaped simulation frame with caller-owned timestamp
   * writes. The caller resolves/maps the shared query set after its profile.
   */
  stepWithGpuTimestampWrites(
    writes: GpuTimestampIntervalWrites,
    overrides: Partial<JGS2StepSettings> = {},
  ): void {
    this.assertUsable();
    validateTimestampWrites(writes);
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepWithGpuTimestampWrites() requested ${overrides.timestep}. ` +
          "Regenerate the precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    this.submitFrameWithGpuTimestampWrites(
      settings,
      normalizeOddIterationCount(settings.iterations),
      writes,
      true,
    );
  }

  /** Batched production steps measured as one GPU timestamp interval. */
  stepFramesWithGpuTimestampWrites(
    frameCount: number,
    writes: GpuTimestampIntervalWrites,
    overrides: Partial<JGS2StepSettings> = {},
  ): void {
    this.assertUsable();
    validateFrameBatchCount(frameCount);
    validateTimestampWrites(writes);
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepFramesWithGpuTimestampWrites() requested ${overrides.timestep}. ` +
          "Regenerate the precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    this.submitFrames(
      settings,
      normalizeOddIterationCount(settings.iterations),
      frameCount,
      true,
      writes,
    );
  }

  /** Exact-iteration variant used by parity/conservation benchmarks. */
  stepExactIterationsWithGpuTimestampWrites(
    iterations: number,
    writes: GpuTimestampIntervalWrites,
    overrides: Partial<JGS2StepSettings> = {},
  ): void {
    this.assertUsable();
    validateExactIterationCount(iterations);
    validateTimestampWrites(writes);
    if (
      overrides.timestep !== undefined &&
      !jgs2TimestepsMatch(this.preprocessingTimestep, overrides.timestep)
    ) {
      throw new RangeError(
        `JGS2 was precomputed for timestep ${this.preprocessingTimestep}, ` +
          `but stepExactIterationsWithGpuTimestampWrites() requested ${overrides.timestep}. ` +
          "Regenerate the precomputation before changing timestep.",
      );
    }
    const settings = this.resolveStepSettings(overrides);
    this.submitFrameWithGpuTimestampWrites(settings, iterations, writes, false);
  }

  private submitFrame(
    settings: JGS2StepSettings,
    iterations: number,
    stopAfterConvergence: boolean,
  ): void {
    this.submitFrames(settings, iterations, 1, stopAfterConvergence);
  }

  private resolveStepSettings(
    overrides: Partial<JGS2StepSettings>,
  ): JGS2StepSettings {
    if (
      this.contactCandidateCount > 0 &&
      overrides.ipcMinimumDistance !== undefined &&
      Math.fround(overrides.ipcMinimumDistance) !==
        Math.fround(this.defaultSettings.ipcMinimumDistance)
    ) {
      throw new RangeError(
        "ipcMinimumDistance is fixed when an IPC solver is created; recreate " +
          "the solver before changing it.",
      );
    }
    const resolved = resolveJGS2StepSettings(this.defaultSettings, overrides);
    // The free-body correction is one common x/z translation per body, so it
    // preserves every deformation gradient and determinant on the horizontal
    // debug plane. Active force/target objectives and mesh contacts are not
    // independently translation invariant per body, however, and must retain
    // their solved center-of-mass motion.
    return this.globalizationEnabled &&
      (this.objectiveFlagsValue !== 0 ||
        (this.contactCandidateCount > 0 &&
          resolved.ipcBarrierStiffness > 0)) &&
      resolved.horizontalBodyCorrection
      ? { ...resolved, horizontalBodyCorrection: false }
      : resolved;
  }

  private submitFrameWithGpuTimestampWrites(
    settings: JGS2StepSettings,
    iterations: number,
    writes: GpuTimestampIntervalWrites,
    stopAfterConvergence: boolean,
  ): void {
    this.submitFrames(
      settings,
      iterations,
      1,
      stopAfterConvergence,
      writes,
    );
  }

  private submitFrames(
    settings: JGS2StepSettings,
    iterations: number,
    frameCount: number,
    stopAfterConvergence: boolean,
    timestampWrites?: GpuTimestampIntervalWrites,
  ): void {
    if (this.globalizationEnabled) {
      const substepsPerIteration =
        settings.schedule === "graph-colored-gauss-seidel"
          ? this.scheduleColorCount
          : 1;
      validateGlobalizedSubmissionWork(
        frameCount,
        iterations * substepsPerIteration,
      );
    }
    this.prepareFrame(settings, iterations, stopAfterConvergence);
    const encoder = this.device.createCommandEncoder({
      label:
        frameCount === 1
          ? "jgs2-step-command-encoder"
          : "jgs2-batched-step-command-encoder",
    });
    for (let frame = 0; frame < frameCount; frame += 1) {
      const frameTimestampWrites: JGS2FrameTimestampWrites | undefined =
        timestampWrites
          ? {
              querySet: timestampWrites.querySet,
              ...(frame === 0
                ? { startWriteIndex: timestampWrites.startWriteIndex }
                : {}),
              ...(frame === frameCount - 1
                ? { endWriteIndex: timestampWrites.endWriteIndex }
                : {}),
            }
          : undefined;
      this.encodeFrame(
        encoder,
        settings,
        iterations,
        frameTimestampWrites,
      );
    }
    this.device.queue.submit([encoder.finish()]);
    this.markGlobalizationRecordsAvailable();
  }

  private markGlobalizationRecordsAvailable(): void {
    if (this.globalizationEnabled) {
      this.globalizationRecordsAvailable = true;
    }
  }

  private prepareFrame(
    settings: JGS2StepSettings,
    iterations: number,
    stopAfterConvergence: boolean,
  ): void {
    if (
      this.globalizationEnabled &&
      iterations > JGS2_GLOBALIZATION_HISTORY_CAPACITY
    ) {
      throw new RangeError(
        `Stable JGS2 records at most ` +
          `${JGS2_GLOBALIZATION_HISTORY_CAPACITY} nonlinear iterations per ` +
          `frame; got ${iterations}.`,
      );
    }
    this.submittedIterations = iterations;
    const previousSettings = this.submittedSettings;
    if (
      settings.ipcMinimumDistance !== previousSettings.ipcMinimumDistance ||
      settings.ipcFrictionCoefficient !==
        previousSettings.ipcFrictionCoefficient ||
      settings.ipcFrictionVelocityEpsilon !==
        previousSettings.ipcFrictionVelocityEpsilon ||
      settings.ipcStepSafety !== previousSettings.ipcStepSafety
    ) {
      this.device.queue.writeBuffer(
        this.buffers.dynamic,
        (this.dynamicOffsets.ipcContact + 1) * 16,
        new Float32Array([
          settings.ipcMinimumDistance,
          settings.ipcFrictionCoefficient,
          settings.ipcFrictionVelocityEpsilon,
          settings.ipcStepSafety,
        ]),
      );
    }
    this.submittedSettings = settings;

    this.device.queue.writeBuffer(
      this.uniforms.base,
      0,
      packUniforms(
        this.inputShape,
        this.dynamicOffsets,
        settings,
        this.dynamicOffsets.posA,
        this.dynamicOffsets.posA,
        this.globalizationEnabled,
        this.globalizationEnabled && stopAfterConvergence,
        this.sceneScale,
        this.objectiveFlagsValue,
      ),
    );
    this.device.queue.writeBuffer(
      this.uniforms.fromBToA,
      0,
      packUniforms(
        this.inputShape,
        this.dynamicOffsets,
        settings,
        this.dynamicOffsets.posB,
        this.dynamicOffsets.posA,
        this.globalizationEnabled,
        this.globalizationEnabled && stopAfterConvergence,
        this.sceneScale,
        this.objectiveFlagsValue,
      ),
    );
    this.device.queue.writeBuffer(
      this.uniforms.fromAToB,
      0,
      packUniforms(
        this.inputShape,
        this.dynamicOffsets,
        settings,
        this.dynamicOffsets.posA,
        this.dynamicOffsets.posB,
        this.globalizationEnabled,
        this.globalizationEnabled && stopAfterConvergence,
        this.sceneScale,
        this.objectiveFlagsValue,
      ),
    );
    if (settings.schedule === "graph-colored-gauss-seidel") {
      if (
        this.uniforms.coloredAToA.length !== this.scheduleColorCount ||
        this.uniforms.coloredBToB.length !== this.scheduleColorCount
      ) {
        throw new Error("Colored Gauss-Seidel uniforms are unavailable.");
      }
      for (let color = 0; color < this.scheduleColorCount; color += 1) {
        this.device.queue.writeBuffer(
          this.uniforms.coloredAToA[color]!,
          0,
          packUniforms(
            this.inputShape,
            this.dynamicOffsets,
            settings,
            this.dynamicOffsets.posA,
            this.dynamicOffsets.posA,
            this.globalizationEnabled,
            this.globalizationEnabled && stopAfterConvergence,
            this.sceneScale,
            this.objectiveFlagsValue,
            color,
          ),
        );
        this.device.queue.writeBuffer(
          this.uniforms.coloredBToB[color]!,
          0,
          packUniforms(
            this.inputShape,
            this.dynamicOffsets,
            settings,
            this.dynamicOffsets.posB,
            this.dynamicOffsets.posB,
            this.globalizationEnabled,
            this.globalizationEnabled && stopAfterConvergence,
            this.sceneScale,
            this.objectiveFlagsValue,
            color,
          ),
        );
      }
    }
  }

  private encodeFrame(
    encoder: GPUCommandEncoder,
    settings: JGS2StepSettings,
    iterations: number,
    timestampWrites?: JGS2FrameTimestampWrites,
  ): void {
    const contactEnabled =
      this.contactCandidateCount > 0 && settings.ipcBarrierStiffness > 0;
    encodeDispatch(
      encoder,
      this.pipelines.predict,
      this.uniforms.baseBindGroup,
      this.vertexCount,
      "jgs2-predict-pass",
      timestampWrites?.startWriteIndex !== undefined
        ? {
            querySet: timestampWrites.querySet,
            beginningOfPassWriteIndex: timestampWrites.startWriteIndex,
          }
        : undefined,
    );

    if (contactEnabled) {
      encodeDispatch(
        encoder,
        this.pipelines.lagContact!,
        this.uniforms.baseBindGroup,
        this.contactCandidateCount,
        "jgs2-ipc-lag-contact-pass",
      );
    }

    if (this.globalizationEnabled) {
      // The local shader consumes stable material only after this exact-f32
      // all-tetrahedron source check has populated the control record. At the
      // start of a frame posA is the accepted pose and predict copies the same
      // bits to posB, so the base bind group intentionally evaluates posA as
      // both source and candidate.
      encodeDispatch(
        encoder,
        this.pipelines.candidateTetrahedron!,
        this.uniforms.baseBindGroup,
        this.globalizationElementCount,
        "jgs2-source-feasibility-preflight-pass",
      );
      encodeDispatch(
        encoder,
        this.pipelines.reduceCandidate!,
        this.uniforms.baseBindGroup,
        1,
        "jgs2-source-feasibility-preflight-reduction-pass",
      );
    }

    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const bindGroup =
        iteration % 2 === 0
          ? this.uniforms.fromBToABindGroup
          : this.uniforms.fromAToBBindGroup;
      if (contactEnabled) {
        encodeDispatch(
          encoder,
          this.pipelines.buildActiveContactRows!,
          bindGroup,
          this.vertexCount,
          `jgs2-ipc-build-active-contact-rows-pass-${iteration}`,
        );
      }
      if (settings.schedule === "graph-colored-gauss-seidel") {
        // Preserve the complete accepted sweep source for assembled
        // globalization, then update the target in place one conflict-free
        // color at a time. Distinct per-color uniforms are required because
        // queue.writeBuffer calls are not command-stream snapshots.
        encodeDispatch(
          encoder,
          this.pipelines.copyPosition,
          bindGroup,
          this.vertexCount,
          `jgs2-colored-copy-source-pass-${iteration}`,
        );
        for (let color = 0; color < this.scheduleColorCount; color += 1) {
          const colorBindGroup =
            iteration % 2 === 0
              ? this.uniforms.coloredAToABindGroups[color]!
              : this.uniforms.coloredBToBBindGroups[color]!;
          encodeDispatch(
            encoder,
            this.pipelines.tetPolarRotation,
            colorBindGroup,
            this.tetCount,
            `jgs2-colored-tet-polar-pass-${iteration}-${color}`,
          );
          encodeDispatch(
            encoder,
            this.pipelines.vertexPolarRotation,
            colorBindGroup,
            this.vertexCount,
            `jgs2-colored-vertex-polar-pass-${iteration}-${color}`,
          );
          encodeDispatch(
            encoder,
            this.pipelines.solve,
            colorBindGroup,
            this.vertexCount,
            `jgs2-colored-solve-pass-${iteration}-${color}`,
          );
        }
        // The in-place color uniforms have source == target. Re-evaluate the
        // complete sweep against the preserved plain S/T pair after a dispatch
        // boundary makes every colored write visible.
        if (this.globalizationEnabled && !contactEnabled) {
          encodeDispatch(
            encoder,
            this.pipelines.assembledVertexEnergy!,
            bindGroup,
            this.vertexCount,
            `jgs2-colored-assembled-energy-pass-${iteration}`,
          );
        }
      } else {
        encodeDispatch(
          encoder,
          this.pipelines.tetPolarRotation,
          bindGroup,
          this.tetCount,
          `jgs2-tet-polar-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.vertexPolarRotation,
          bindGroup,
          this.vertexCount,
          `jgs2-vertex-polar-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.solve,
          bindGroup,
          this.vertexCount,
          `jgs2-solve-pass-${iteration}`,
        );
      }
      if (contactEnabled) {
        encodeDispatch(
          encoder,
          this.pipelines.candidateContactStep!,
          bindGroup,
          this.contactCandidateCount,
          `jgs2-ipc-candidate-step-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.reduceContactStep!,
          bindGroup,
          1,
          `jgs2-ipc-reduce-step-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.applyContactStep!,
          bindGroup,
          this.vertexCount,
          `jgs2-ipc-apply-step-pass-${iteration}`,
        );
        // Verify the actual f32 pose after scaling. A failed candidate writes
        // alpha zero, and the second apply pass reverts every vertex to the
        // iteration source before assembled material acceptance runs.
        encodeDispatch(
          encoder,
          this.pipelines.validateContactStep!,
          bindGroup,
          this.contactCandidateCount,
          `jgs2-ipc-validate-scaled-step-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.reduceContactStep!,
          bindGroup,
          1,
          `jgs2-ipc-reduce-scaled-step-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.applyContactStep!,
          bindGroup,
          this.vertexCount,
          `jgs2-ipc-apply-validated-step-pass-${iteration}`,
        );
        // Contact safe-step scaling happens after jgs2Solve. Evaluate the
        // final source/candidate energy pair only after validation cached the
        // scaled target pose.
        encodeDispatch(
          encoder,
          this.pipelines.assembledVertexEnergy!,
          bindGroup,
          this.vertexCount,
          `jgs2-ipc-rescaled-energy-pass-${iteration}`,
        );
      }
      if (this.globalizationEnabled) {
        encodeDispatch(
          encoder,
          this.pipelines.candidateTetrahedron!,
          bindGroup,
          this.globalizationElementCount,
          `jgs2-candidate-tetrahedron-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.reduceCandidate!,
          bindGroup,
          1,
          `jgs2-reduce-candidate-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.applyCandidate!,
          bindGroup,
          this.vertexCount,
          `jgs2-apply-candidate-pass-${iteration}`,
        );
        if (contactEnabled) {
          encodeDispatch(
            encoder,
            this.pipelines.promoteAcceptedContactCache!,
            bindGroup,
            this.contactCandidateCount,
            `jgs2-ipc-promote-accepted-contact-cache-pass-${iteration}`,
          );
        }
        encodeDispatch(
          encoder,
          this.pipelines.convergenceGradient!,
          bindGroup,
          this.vertexCount,
          `jgs2-convergence-gradient-pass-${iteration}`,
        );
        encodeDispatch(
          encoder,
          this.pipelines.reduceConvergence!,
          bindGroup,
          1,
          `jgs2-reduce-convergence-pass-${iteration}`,
        );
      }
    }

    if (iterations % 2 === 0) {
      encodeDispatch(
        encoder,
        this.pipelines.copyPosition,
        this.uniforms.fromBToABindGroup,
        this.vertexCount,
        "jgs2-copy-even-result-to-canonical-position-pass",
      );
    }

    if (settings.horizontalBodyCorrection) {
      encodeDispatch(
        encoder,
        this.pipelines.bodyHorizontalCorrection,
        this.uniforms.baseBindGroup,
        this.bodyCount,
        "jgs2-body-horizontal-correction-pass",
      );
      encodeDispatch(
        encoder,
        this.pipelines.applyBodyHorizontalCorrection,
        this.uniforms.baseBindGroup,
        this.vertexCount,
        "jgs2-apply-body-horizontal-correction-pass",
      );
    }

    encodeDispatch(
      encoder,
      this.pipelines.finalize,
      this.uniforms.baseBindGroup,
      this.vertexCount,
      "jgs2-finalize-pass",
      timestampWrites?.endWriteIndex !== undefined
        ? {
            querySet: timestampWrites.querySet,
            endOfPassWriteIndex: timestampWrites.endWriteIndex,
          }
        : undefined,
    );
  }

  async awaitIdle(): Promise<void> {
    this.assertUsable();
    await this.device.queue.onSubmittedWorkDone();
  }

  async readPositions(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.currentPositionByteOffset,
      this.currentPositionByteLength,
      "jgs2-position-readback",
    );
  }

  async readVelocities(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.velocityByteOffset,
      this.velocityByteLength,
      "jgs2-velocity-readback",
    );
  }

  /** Read the GPU's exact inertial target for CPU-oracle comparisons. */
  async readPredictedPositions(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.dynamicOffsets.predicted * 16,
      this.currentPositionByteLength,
      "jgs2-predicted-position-readback",
    );
  }

  /**
   * Explicitly read each vertex's final nonlinear update record. The x lane is
   * ||delta|| and w is one after a submitted frame (zero before the first
   * nonlinear iteration); pinned vertices report a zero magnitude. A stopped
   * production frame retains the final executed iteration's update.
   */
  async readFinalIterationUpdates(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.dynamicOffsets.finalUpdate * 16,
      this.currentPositionByteLength,
      "jgs2-final-update-readback",
    );
  }

  /**
   * Explicit test/inspection readback of the final local records and the
   * GPU-resident per-iteration history. Production stepping never calls it.
   */
  async readGlobalizationDiagnostics(): Promise<JGS2GlobalizationDiagnostics> {
    this.assertUsable();
    if (!this.globalizationEnabled) {
      throw new Error(
        "Globalization diagnostics are available only for stable " +
          "Neo-Hookean production solves.",
      );
    }
    if (!this.globalizationRecordsAvailable) {
      throw new Error(
        "Globalization diagnostics are unavailable until a stable nonlinear " +
          "iteration or diagnostic test kernel has produced records.",
      );
    }
    const first = this.dynamicOffsets.localGlobalization;
    const packed = await this.readVec4Region(
      first * 16,
      (this.dynamicOffsets.vec4Count - first) * 16,
      "jgs2-globalization-readback",
    );
    const local: JGS2GlobalizationLocalDiagnostic[] = [];
    const localFloats = JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S * 4;
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      const base = vertex * localFloats;
      local.push(
        decodeJGS2GlobalizationLocalDiagnostic(
          packed.subarray(base, base + localFloats),
        ),
      );
    }
    const controlFloatOffset =
      (this.dynamicOffsets.globalizationControl - first) * 4;
    const rawHistoryCount = packed[controlFloatOffset + 7]!;
    if (
      !Number.isSafeInteger(rawHistoryCount) ||
      rawHistoryCount < 0 ||
      rawHistoryCount > JGS2_GLOBALIZATION_HISTORY_CAPACITY
    ) {
      throw new Error(
        `GPU returned invalid globalization history count ` +
          `${rawHistoryCount}.`,
      );
    }
    const history: JGS2GlobalizationHistoryRecord[] = [];
    const historyFloats = JGS2_GLOBALIZATION_HISTORY_VEC4S * 4;
    const historyFloatOffset =
      (this.dynamicOffsets.globalizationHistory - first) * 4;
    for (let index = 0; index < rawHistoryCount; index += 1) {
      const base = historyFloatOffset + index * historyFloats;
      history.push(
        decodeJGS2GlobalizationHistoryRecord(
          packed.subarray(base, base + historyFloats),
        ),
      );
    }
    return { local, history, historyCount: rawHistoryCount };
  }

  /**
   * Test-only entry point for the production assembled determinant/revert and
   * convergence kernels. It deliberately shares the real pipelines rather
   * than duplicating their policy in a diagnostic shader.
   */
  async evaluateAssembledCandidateForTest(
    sourcePositions: Float32Array,
    candidatePositions: Float32Array,
    overrides: Partial<JGS2StepSettings> = {},
  ): Promise<JGS2AssembledCandidateTestResult> {
    this.assertUsable();
    if (!this.globalizationEnabled) {
      throw new Error("Assembled candidate tests require stable globalization.");
    }
    const expectedFloats = this.vertexCount * 4;
    if (
      sourcePositions.length !== expectedFloats ||
      candidatePositions.length !== expectedFloats
    ) {
      throw new RangeError(
        `Assembled candidate poses must each contain ${expectedFloats} floats.`,
      );
    }
    for (const [label, values] of [
      ["source", sourcePositions],
      ["candidate", candidatePositions],
    ] as const) {
      for (const value of values) {
        if (!Number.isFinite(value)) {
          throw new RangeError(`${label} candidate pose must be finite.`);
        }
      }
    }
    const settings = this.resolveStepSettings(overrides);
    this.prepareFrame(settings, 1, false);
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.posB * 16,
      sourcePositions,
    );
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.posA * 16,
      candidatePositions,
    );
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.predicted * 16,
      sourcePositions,
    );
    const local = new Float32Array(
      this.vertexCount * JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S * 4,
    );
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      local[
        vertex * JGS2_GLOBALIZATION_LOCAL_DIAGNOSTIC_VEC4S * 4 + 15
      ] = 0;
    }
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.localGlobalization * 16,
      local,
    );
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.globalizationControl * 16,
      new Float32Array(16),
    );

    const encoder = this.device.createCommandEncoder({
      label: "jgs2-test-assembled-candidate-command-encoder",
    });
    const bindGroup = this.uniforms.fromBToABindGroup;
    encodeDispatch(
      encoder,
      this.pipelines.assembledVertexEnergy!,
      bindGroup,
      this.vertexCount,
      "jgs2-test-assembled-vertex-energy-pass",
    );
    encodeDispatch(
      encoder,
      this.pipelines.candidateTetrahedron!,
      bindGroup,
      this.globalizationElementCount,
      "jgs2-test-candidate-tetrahedron-pass",
    );
    encodeDispatch(
      encoder,
      this.pipelines.reduceCandidate!,
      bindGroup,
      1,
      "jgs2-test-reduce-candidate-pass",
    );
    encodeDispatch(
      encoder,
      this.pipelines.applyCandidate!,
      bindGroup,
      this.vertexCount,
      "jgs2-test-apply-candidate-pass",
    );
    encodeDispatch(
      encoder,
      this.pipelines.convergenceGradient!,
      bindGroup,
      this.vertexCount,
      "jgs2-test-convergence-gradient-pass",
    );
    encodeDispatch(
      encoder,
      this.pipelines.reduceConvergence!,
      bindGroup,
      1,
      "jgs2-test-reduce-convergence-pass",
    );
    this.device.queue.submit([encoder.finish()]);
    this.markGlobalizationRecordsAvailable();
    const positions = await this.readPositions();
    const globalization = await this.readGlobalizationDiagnostics();
    return { positions, globalization };
  }

  /** Test-only wrapper around the exact production convergence reduction. */
  async evaluateConvergenceReductionForTest(
    input: JGS2ConvergenceReductionTestInput,
    overrides: Partial<JGS2StepSettings> = {},
  ): Promise<JGS2GlobalizationHistoryRecord> {
    this.assertUsable();
    if (!this.globalizationEnabled) {
      throw new Error("Convergence reduction tests require stable globalization.");
    }
    if (input.gradientComponents.length !== 5) {
      throw new RangeError("Convergence reduction requires five gradient components.");
    }
    const reductionVertexCount = input.reductionVertexCount ?? this.vertexCount;
    if (
      !Number.isSafeInteger(reductionVertexCount) ||
      reductionVertexCount < 1 ||
      reductionVertexCount > this.vertexCount
    ) {
      throw new RangeError(
        `Convergence reduction vertex count must be between 1 and ` +
          `${this.vertexCount}; got ${reductionVertexCount}.`,
      );
    }
    const xyzCount = this.vertexCount * 3;
    if (
      input.gradientComponents.some((component) => component.length !== xyzCount) ||
      input.acceptedUpdates.length !== xyzCount
    ) {
      throw new RangeError(
        `Convergence gradient and update arrays must contain ${xyzCount} floats.`,
      );
    }
    if (
      typeof input.assembledAccepted !== "boolean" ||
      typeof input.assembledReverted !== "boolean" ||
      !Number.isSafeInteger(input.localFailureCount) ||
      input.localFailureCount < 0
    ) {
      throw new RangeError("Invalid assembled convergence test controls.");
    }
    const packedComponents = new Float32Array(this.vertexCount * 5 * 4);
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      for (let component = 0; component < 5; component += 1) {
        const source = input.gradientComponents[component]!;
        const destination = (vertex * 5 + component) * 4;
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          const value = source[vertex * 3 + coordinate]!;
          if (!Number.isFinite(value)) {
            throw new RangeError("Convergence test gradients must be finite.");
          }
          packedComponents[destination + coordinate] = value;
        }
      }
    }
    const packedUpdates = new Float32Array(this.vertexCount * 4);
    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      const x = input.acceptedUpdates[vertex * 3]!;
      const y = input.acceptedUpdates[vertex * 3 + 1]!;
      const z = input.acceptedUpdates[vertex * 3 + 2]!;
      if (![x, y, z].every(Number.isFinite)) {
        throw new RangeError("Convergence test updates must be finite.");
      }
      packedUpdates[vertex * 4] = Math.hypot(x, y, z);
      packedUpdates[vertex * 4 + 3] = 1;
    }
    const control = new Float32Array(16);
    control.set([1, 1, 1, JGS2_GLOBALIZATION_VALIDITY_BITS_MASK], 0);
    control.set(
      [
        Number(input.assembledAccepted),
        Number(input.assembledReverted),
        input.localFailureCount,
        0,
      ],
      4,
    );
    control.set([0, 0, 0, 1], 8);
    const settings = this.resolveStepSettings(overrides);
    this.prepareFrame(settings, 1, false);
    if (reductionVertexCount !== this.vertexCount) {
      this.device.queue.writeBuffer(
        this.uniforms.base,
        0,
        packUniforms(
          { ...this.inputShape, vertexCount: reductionVertexCount },
          this.dynamicOffsets,
          settings,
          this.dynamicOffsets.posA,
          this.dynamicOffsets.posA,
          true,
          false,
          this.sceneScale,
          this.objectiveFlagsValue,
        ),
      );
    }
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.convergenceGradient * 16,
      packedComponents,
    );
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.finalUpdate * 16,
      packedUpdates,
    );
    this.device.queue.writeBuffer(
      this.buffers.dynamic,
      this.dynamicOffsets.globalizationControl * 16,
      control,
    );
    const encoder = this.device.createCommandEncoder({
      label: "jgs2-test-convergence-reduction-command-encoder",
    });
    encodeDispatch(
      encoder,
      this.pipelines.reduceConvergence!,
      this.uniforms.baseBindGroup,
      1,
      "jgs2-test-convergence-reduction-pass",
    );
    this.device.queue.submit([encoder.finish()]);
    this.markGlobalizationRecordsAvailable();
    const diagnostics = await this.readGlobalizationDiagnostics();
    const record = diagnostics.history[0];
    if (!record) {
      throw new Error("GPU convergence reduction did not write history slot zero.");
    }
    return record;
  }

  /**
   * Explicit test-only readback of the three padded columns of every
   * vertex-local polar frame. Production rendering/stepping never calls this.
   */
  async readVertexRotations(): Promise<Float32Array> {
    this.assertUsable();
    return this.readVec4Region(
      this.dynamicOffsets.vertexRotation * 16,
      this.vertexCount * 3 * 16,
      "jgs2-vertex-rotation-readback",
    );
  }

  /**
   * Explicitly evaluate the exact, all-element frozen-frame implicit-Euler
   * oracle on the GPU. The production frame loop never creates, dispatches, or
   * reads this diagnostic path unless this method is called.
   *
   * The current canonical position freezes the co-rotated element frames and
   * is evaluated against the most recently submitted predicted inertial pose
   * and step settings. Callers should pause stepping while awaiting a coherent
   * deterministic snapshot.
   */
  async readOracleDiagnostics(): Promise<JGS2GpuOracleDiagnostics> {
    this.assertUsable();
    const settings = this.submittedSettings;
    if (
      this.contactCandidateCount > 0 &&
      settings.ipcBarrierStiffness > 0
    ) {
      throw new Error(
        "Exact oracle diagnostics do not yet include IPC contact energy; " +
          "disable IPC before requesting this diagnostic.",
      );
    }
    const evaluator = await this.getOracleEvaluator();
    this.assertUsable();
    this.diagnosticReadbackCount += 1;
    return evaluator.evaluate({
      currentPositionOffset: this.dynamicOffsets.posA,
      predictedPositionOffset: this.dynamicOffsets.predicted,
      finalUpdateOffset: this.dynamicOffsets.finalUpdate,
      timestep: settings.timestep,
      floorHeight: settings.floorHeight,
      floorStiffness: settings.floorStiffness,
      rotationEpsilon: settings.rotationEpsilon,
      objectiveFlags: this.objectiveFlagsValue,
    });
  }

  private getOracleEvaluator(): Promise<JGS2GpuOracleEvaluator> {
    if (this.oracleEvaluator) {
      return Promise.resolve(this.oracleEvaluator);
    }
    if (!this.oracleEvaluatorCreation) {
      this.oracleEvaluatorCreation = JGS2GpuOracleEvaluator.create(
        this.device,
        this.vertexCount,
        this.tetCount,
        {
          dynamic: this.buffers.dynamic,
          vertices: this.buffers.vertices,
          tets: this.buffers.tets,
          stiffness: this.buffers.stiffness,
          adjacency: this.buffers.adjacency,
          objectives: this.buffers.objectives,
        },
      )
        .then((evaluator) => {
          if (this.destroyed) {
            evaluator.destroy();
            throw new Error("JGS2GpuSolver has been destroyed.");
          }
          this.oracleEvaluator = evaluator;
          return evaluator;
        })
        .catch((error: unknown) => {
          this.oracleEvaluatorCreation = undefined;
          throw error;
        });
    }
    return this.oracleEvaluatorCreation;
  }

  private assertObjectiveUpdatesSupported(): void {
    this.assertUsable();
    if (!this.globalizationEnabled) {
      throw new Error(
        "Dynamic external forces and quadratic targets require the stable " +
          "Neo-Hookean production path.",
      );
    }
  }

  private assertObjectiveVertex(vertex: number): void {
    if (
      !Number.isSafeInteger(vertex) ||
      vertex < 0 ||
      vertex >= this.vertexCount
    ) {
      throw new RangeError(
        `Objective vertex must be an integer from 0 through ` +
          `${this.vertexCount - 1}; got ${vertex}.`,
      );
    }
  }

  private writeCompleteObjectiveState(next: Float32Array): void {
    const activity = computePackedObjectiveActivity(next, this.vertexCount);
    this.device.queue.writeBuffer(this.buffers.objectives, 0, next);
    this.objectiveData.set(next);
    this.activeExternalForceCount = activity.externalForceVertexCount;
    this.activeTargetCount = activity.quadraticTargetVertexCount;
    this.objectiveFlagsValue = activity.flags;
    this.objectiveRevisionValue += 1;
  }

  private finishObjectiveWrite(): void {
    this.objectiveFlagsValue =
      (this.activeExternalForceCount > 0
        ? JGS2_OBJECTIVE_EXTERNAL_FORCE_ACTIVE_BIT
        : 0) |
      (this.activeTargetCount > 0
        ? JGS2_OBJECTIVE_TARGET_ACTIVE_BIT
        : 0);
    this.objectiveRevisionValue += 1;
  }

  private positionView(vec4Offset: number): JGS2PositionBufferView {
    return {
      buffer: this.currentPositionBuffer,
      offset: vec4Offset * 16,
      size: this.currentPositionByteLength,
      stride: 16,
    };
  }

  private async readVec4Region(
    sourceOffset: number,
    byteLength: number,
    label: string,
  ): Promise<Float32Array> {
    this.diagnosticReadbackCount += 1;
    let readback: GPUBuffer | undefined;
    try {
      readback = this.device.createBuffer({
        label,
        size: byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = this.device.createCommandEncoder({
        label: `${label}-encoder`,
      });
      encoder.copyBufferToBuffer(
        this.buffers.dynamic,
        sourceOffset,
        readback,
        0,
        byteLength,
      );
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      return result;
    } finally {
      readback?.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    for (const buffer of Object.values(this.buffers)) {
      buffer.destroy();
    }
    this.uniforms.base.destroy();
    this.uniforms.fromBToA.destroy();
    this.uniforms.fromAToB.destroy();
    for (const uniform of this.uniforms.coloredAToA) uniform.destroy();
    for (const uniform of this.uniforms.coloredBToB) uniform.destroy();
    this.oracleEvaluator?.destroy();
    this.timestampTimer.destroy();
  }

  private assertUsable(): void {
    if (this.destroyed) {
      throw new Error("JGS2GpuSolver has been destroyed.");
    }
  }
}
