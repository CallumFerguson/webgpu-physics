import { buildLowFrequencyTrainingPoses } from "./cubature";
import { multiply3 } from "./math";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import { evaluateStableNeoHookeanMesh } from "./stable-neo-hookean";
import type {
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

export const NONLINEAR_CUBATURE_CORPUS_VERSION = "phase1-nonlinear-cubature-v1";
export const NONLINEAR_CUBATURE_MODE_COUNT = 8;
export const NONLINEAR_CUBATURE_VALIDATION_ONLY_MODE_COUNT = 4;
export const NONLINEAR_CUBATURE_INVERSE_ITERATIONS = 24;
export const NONLINEAR_CUBATURE_TRAINING_AMPLITUDE = 0.12;
export const NONLINEAR_CUBATURE_VALIDATION_LOW_AMPLITUDE = 0.06;
export const NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE = 0.18;
export const NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT = 8;
export const NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE_START = 4;
export const NONLINEAR_CUBATURE_ROTATED_VALIDATION_COUNT = 4;
export const NONLINEAR_CUBATURE_VALIDATION_PREDICTED_BLEND = 0.25;
export const NONLINEAR_CUBATURE_MINIMUM_POSE_DETERMINANT = 0.5;
export const NONLINEAR_CUBATURE_DIRECTION_NORMALIZATION =
  "max(max-tet-displacement-gradient-frobenius,max-vertex-displacement-over-scene-scale)" as const;
export const NONLINEAR_CUBATURE_MIXTURE_COEFFICIENT_FORMULA =
  "sin((r+1)(k+1)0.754877666)+0.5cos((r+1)(k+3)0.569840291)" as const;

export type NonlinearCubaturePoseKind =
  | "training-mode"
  | "validation-combination"
  | "validation-rotated";

export interface NonlinearCubaturePose {
  readonly id: string;
  readonly kind: NonlinearCubaturePoseKind;
  readonly index: number;
  /** Absolute full xyz positions. */
  readonly positions: Float64Array;
  /** Absolute implicit-Euler inertial targets. */
  readonly predictedPositions: Float64Array;
  /** Full xyz displacement from the unrotated rest pose. */
  readonly displacement: Float64Array;
  readonly maximumDisplacement: number;
  /** Frozen pre-rotation perturbation amplitude. */
  readonly perturbationAmplitude: number;
  readonly minimumDeformationDeterminant: number;
}

export interface NonlinearCubaturePoseCorpus {
  readonly version: typeof NONLINEAR_CUBATURE_CORPUS_VERSION;
  readonly sceneScale: number;
  readonly training: readonly NonlinearCubaturePose[];
  readonly validation: readonly NonlinearCubaturePose[];
}

export interface NonlinearCubaturePoseCorpusOptions {
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly materials: readonly LinearMaterial[];
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
  readonly modeCount?: number;
  readonly inverseIterations?: number;
  readonly trainingAmplitude?: number;
  readonly validationLowAmplitude?: number;
  readonly validationHighAmplitude?: number;
  readonly validationCount?: number;
  readonly minimumDeterminant?: number;
}

function requirePositive(value: number, label: string): number {
  if (!(value > 0) || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite and positive.`);
  }
  return value;
}

function requirePositiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer.`);
  }
  return value;
}

function computeSceneScale(positions: Float64Array): number {
  if (positions.length < 3 || positions.length % 3 !== 0) {
    throw new RangeError("Nonlinear Cubature rest positions are malformed.");
  }
  const minimum = [
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const maximum = [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];
  for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const value = positions[vertex * 3 + coordinate]!;
      if (!Number.isFinite(value)) {
        throw new RangeError("Nonlinear Cubature rest positions must be finite.");
      }
      minimum[coordinate] = Math.min(minimum[coordinate]!, value);
      maximum[coordinate] = Math.max(maximum[coordinate]!, value);
    }
  }
  return requirePositive(
    Math.hypot(
      maximum[0]! - minimum[0]!,
      maximum[1]! - minimum[1]!,
      maximum[2]! - minimum[2]!,
    ),
    "Nonlinear Cubature scene scale",
  );
}

function maximumVertexNorm(values: Float64Array): number {
  let maximum = 0;
  for (let vertex = 0; vertex < values.length / 3; vertex += 1) {
    maximum = Math.max(
      maximum,
      Math.hypot(
        values[vertex * 3]!,
        values[vertex * 3 + 1]!,
        values[vertex * 3 + 2]!,
      ),
    );
  }
  return maximum;
}

function displacementDirectionMeasure(
  mesh: TetrahedralMesh,
  restData: RestTetraData,
  displacement: Float64Array,
  sceneScale: number,
): number {
  let maximumGradientNorm = 0;
  for (
    let tetrahedron = 0;
    tetrahedron < getTetrahedronCount(mesh);
    tetrahedron += 1
  ) {
    const a = mesh.tetrahedra[tetrahedron * 4]!;
    const displacementShape = new Float64Array(9);
    for (let column = 0; column < 3; column += 1) {
      const vertex = mesh.tetrahedra[tetrahedron * 4 + column + 1]!;
      for (let row = 0; row < 3; row += 1) {
        displacementShape[row * 3 + column] =
          displacement[vertex * 3 + row]! - displacement[a * 3 + row]!;
      }
    }
    const gradient = multiply3(
      displacementShape,
      restData.inverseRestMatrices.subarray(
        tetrahedron * 9,
        tetrahedron * 9 + 9,
      ),
    );
    let squaredFrobenius = 0;
    for (const value of gradient) squaredFrobenius += value * value;
    maximumGradientNorm = Math.max(
      maximumGradientNorm,
      Math.sqrt(squaredFrobenius),
    );
  }
  return Math.max(
    maximumGradientNorm,
    maximumVertexNorm(displacement) / sceneScale,
  );
}

function evaluatePose(
  options: NonlinearCubaturePoseCorpusOptions,
  id: string,
  kind: NonlinearCubaturePoseKind,
  index: number,
  positions: Float64Array,
  predictedPositions: Float64Array,
  perturbationAmplitude: number,
  minimumDeterminant: number,
): NonlinearCubaturePose {
  const material = evaluateStableNeoHookeanMesh(
    options.mesh,
    options.restData,
    options.materials,
    positions,
  );
  const minimumDeformationDeterminant = Math.min(
    ...material.deformationDeterminants,
  );
  if (!(minimumDeformationDeterminant >= minimumDeterminant)) {
    throw new RangeError(
      `${id} has minimum J ${minimumDeformationDeterminant}; required at least ` +
        `${minimumDeterminant}.`,
    );
  }
  const displacement = Float64Array.from(
    positions,
    (value, entry) => value - options.mesh.positions[entry]!,
  );
  return {
    id,
    kind,
    index,
    positions,
    predictedPositions,
    displacement,
    maximumDisplacement: maximumVertexNorm(displacement),
    perturbationAmplitude,
    minimumDeformationDeterminant,
  };
}

function createScaledPose(
  options: NonlinearCubaturePoseCorpusOptions,
  kind: "training-mode" | "validation-combination",
  index: number,
  rawDisplacement: Float64Array,
  requestedAmplitude: number,
  predictedBlend: number,
  sceneScale: number,
  minimumDeterminant: number,
): NonlinearCubaturePose {
  const rawMeasure = displacementDirectionMeasure(
    options.mesh,
    options.restData,
    rawDisplacement,
    sceneScale,
  );
  if (!(rawMeasure > 1e-14) || !Number.isFinite(rawMeasure)) {
    throw new Error(`Nonlinear Cubature ${kind} ${index} has zero direction.`);
  }
  const scale = requestedAmplitude / rawMeasure;
  const positions = options.mesh.positions.slice();
  const predictedPositions = options.mesh.positions.slice();
  for (let entry = 0; entry < positions.length; entry += 1) {
    const displacement = rawDisplacement[entry]! * scale;
    positions[entry] += displacement;
    predictedPositions[entry] += predictedBlend * displacement;
  }
  const split = kind === "training-mode" ? "training" : "validation";
  return evaluatePose(
    options,
    `phase1.nonlinear-cubature-${split}/${index
      .toString()
      .padStart(2, "0")}`,
    kind,
    index,
    positions,
    predictedPositions,
    requestedAmplitude,
    minimumDeterminant,
  );
}

function restCenterOfMass(
  positions: Float64Array,
  masses: Float64Array,
): readonly [number, number, number] {
  if (masses.length !== positions.length / 3) {
    throw new RangeError("Nonlinear Cubature masses have the wrong length.");
  }
  const center = [0, 0, 0];
  let totalMass = 0;
  for (let vertex = 0; vertex < masses.length; vertex += 1) {
    const mass = requirePositive(masses[vertex]!, `Vertex ${vertex} mass`);
    totalMass += mass;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      center[coordinate] += mass * positions[vertex * 3 + coordinate]!;
    }
  }
  return [center[0]! / totalMass, center[1]! / totalMass, center[2]! / totalMass];
}

function rotatePositions(
  positions: Float64Array,
  center: readonly [number, number, number],
  rotation: Float64Array,
): Float64Array {
  const result = new Float64Array(positions.length);
  for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
    const relative = [
      positions[vertex * 3]! - center[0],
      positions[vertex * 3 + 1]! - center[1],
      positions[vertex * 3 + 2]! - center[2],
    ];
    for (let row = 0; row < 3; row += 1) {
      result[vertex * 3 + row] =
        center[row]! +
        rotation[row * 3]! * relative[0]! +
        rotation[row * 3 + 1]! * relative[1]! +
        rotation[row * 3 + 2]! * relative[2]!;
    }
  }
  return result;
}

function vectorDot(left: Float64Array, right: Float64Array): number {
  let result = 0;
  for (let entry = 0; entry < left.length; entry += 1) {
    result += left[entry]! * right[entry]!;
  }
  return result;
}

function normalizeAndAppendMode(
  destination: Float64Array[],
  candidate: Float64Array,
): boolean {
  for (const existing of destination) {
    const projection = vectorDot(candidate, existing);
    for (let entry = 0; entry < candidate.length; entry += 1) {
      candidate[entry] -= projection * existing[entry]!;
    }
  }
  const norm = Math.sqrt(vectorDot(candidate, candidate));
  if (!(norm > 1e-10) || !Number.isFinite(norm)) return false;
  for (let entry = 0; entry < candidate.length; entry += 1) {
    candidate[entry] /= norm;
  }
  for (const value of candidate) {
    if (Math.abs(value) <= 1e-12) continue;
    if (value < 0) {
      for (let entry = 0; entry < candidate.length; entry += 1) {
        candidate[entry] *= -1;
      }
    }
    break;
  }
  destination.push(candidate);
  return true;
}

function buildFreeComponentRigidDisplacementModes(
  mesh: TetrahedralMesh,
  masses: Float64Array,
): readonly Float64Array[] {
  const vertexCount = getVertexCount(mesh);
  if (masses.length !== vertexCount) {
    throw new RangeError("Nonlinear Cubature masses have the wrong length.");
  }
  const parents = Uint32Array.from(
    { length: vertexCount },
    (_unused, vertex) => vertex,
  );
  const find = (vertex: number): number => {
    let root = vertex;
    while (parents[root] !== root) root = parents[root]!;
    while (parents[vertex] !== vertex) {
      const next = parents[vertex]!;
      parents[vertex] = root;
      vertex = next;
    }
    return root;
  };
  const union = (left: number, right: number): void => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot === rightRoot) return;
    const minimum = Math.min(leftRoot, rightRoot);
    const maximum = Math.max(leftRoot, rightRoot);
    parents[maximum] = minimum;
  };
  for (let tetrahedron = 0; tetrahedron < getTetrahedronCount(mesh); tetrahedron += 1) {
    const first = mesh.tetrahedra[tetrahedron * 4]!;
    for (let localVertex = 1; localVertex < 4; localVertex += 1) {
      union(first, mesh.tetrahedra[tetrahedron * 4 + localVertex]!);
    }
  }
  const components = new Map<number, number[]>();
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const root = find(vertex);
    const component = components.get(root) ?? [];
    component.push(vertex);
    components.set(root, component);
  }
  const rigidModes: Float64Array[] = [];
  for (const vertices of [...components.values()].sort(
    (left, right) => left[0]! - right[0]!,
  )) {
    if (vertices.some((vertex) => mesh.fixed[vertex] !== 0)) continue;
    let totalMass = 0;
    const center = [0, 0, 0];
    for (const vertex of vertices) {
      const mass = requirePositive(masses[vertex]!, `Vertex ${vertex} mass`);
      totalMass += mass;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        center[coordinate] +=
          mass * mesh.positions[vertex * 3 + coordinate]!;
      }
    }
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      center[coordinate] /= totalMass;
    }
    const candidates = Array.from({ length: 6 }, () =>
      new Float64Array(mesh.positions.length),
    );
    for (const vertex of vertices) {
      const x = mesh.positions[vertex * 3]! - center[0]!;
      const y = mesh.positions[vertex * 3 + 1]! - center[1]!;
      const z = mesh.positions[vertex * 3 + 2]! - center[2]!;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        candidates[coordinate]![vertex * 3 + coordinate] = 1;
      }
      candidates[3]!.set([0, -z, y], vertex * 3);
      candidates[4]!.set([z, 0, -x], vertex * 3);
      candidates[5]!.set([-y, x, 0], vertex * 3);
    }
    for (const candidate of candidates) {
      normalizeAndAppendMode(rigidModes, candidate);
    }
  }
  return rigidModes;
}

function removeRigidDisplacementModes(
  modes: readonly Float64Array[],
  rigidModes: readonly Float64Array[],
  requestedCount: number,
): readonly Float64Array[] {
  const deformationalModes: Float64Array[] = [];
  for (const source of modes) {
    const candidate = source.slice();
    for (const rigid of rigidModes) {
      const projection = vectorDot(candidate, rigid);
      for (let entry = 0; entry < candidate.length; entry += 1) {
        candidate[entry] -= projection * rigid[entry]!;
      }
    }
    normalizeAndAppendMode(deformationalModes, candidate);
    if (deformationalModes.length >= requestedCount) break;
  }
  if (deformationalModes.length === 0) {
    throw new Error(
      "Nonlinear Cubature free components have no deformational modes.",
    );
  }
  return deformationalModes;
}

export const NONLINEAR_CUBATURE_VALIDATION_ROTATIONS = [
  new Float64Array([1, 0, 0, 0, 0, -1, 0, 1, 0]),
  new Float64Array([0, 0, 1, 0, 1, 0, -1, 0, 0]),
  new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]),
  new Float64Array([0, 0, 1, 1, 0, 0, 0, 1, 0]),
] as const;

/**
 * Deterministic nonlinear training and held-out poses. The paper freezes the
 * low-mode source but omits amplitude/sign details; v1 uses deformation-
 * gradient normalization, both signs, mixed modes, altered inertial targets,
 * and exact rigid rotations as separately testable choices.
 */
export function buildNonlinearCubaturePoseCorpus(
  options: NonlinearCubaturePoseCorpusOptions,
): NonlinearCubaturePoseCorpus {
  const vertexCount = getVertexCount(options.mesh);
  if (options.mesh.positions.length !== vertexCount * 3) {
    throw new RangeError("Nonlinear Cubature mesh positions have the wrong length.");
  }
  const referencedMaterialIds = new Set(options.mesh.materialIds);
  for (const materialId of referencedMaterialIds) {
    const material = options.materials[materialId];
    if (!material) {
      throw new RangeError(
        `Nonlinear Cubature material ID ${materialId} is out of range.`,
      );
    }
    if (material.model !== "stable-neo-hookean") {
      throw new RangeError(
        `Nonlinear Cubature material ${material.name} must select stable-neo-hookean.`,
      );
    }
  }
  const modeCount = requirePositiveInteger(
    options.modeCount ?? NONLINEAR_CUBATURE_MODE_COUNT,
    "Nonlinear Cubature mode count",
  );
  const inverseIterations = requirePositiveInteger(
    options.inverseIterations ?? NONLINEAR_CUBATURE_INVERSE_ITERATIONS,
    "Nonlinear Cubature inverse iterations",
  );
  const trainingAmplitude = requirePositive(
    options.trainingAmplitude ?? NONLINEAR_CUBATURE_TRAINING_AMPLITUDE,
    "Nonlinear Cubature training amplitude",
  );
  const validationLowAmplitude = requirePositive(
    options.validationLowAmplitude ?? NONLINEAR_CUBATURE_VALIDATION_LOW_AMPLITUDE,
    "Nonlinear Cubature low validation amplitude",
  );
  const validationHighAmplitude = requirePositive(
    options.validationHighAmplitude ?? NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE,
    "Nonlinear Cubature high validation amplitude",
  );
  const validationCount = requirePositiveInteger(
    options.validationCount ?? NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
    "Nonlinear Cubature validation count",
  );
  const minimumDeterminant = requirePositive(
    options.minimumDeterminant ?? NONLINEAR_CUBATURE_MINIMUM_POSE_DETERMINANT,
    "Nonlinear Cubature minimum determinant",
  );
  const sceneScale = computeSceneScale(options.mesh.positions);
  const hasFixedVertices = [...options.mesh.fixed].some((fixed) => fixed !== 0);
  const requestedDeformationalModes =
    modeCount + NONLINEAR_CUBATURE_VALIDATION_ONLY_MODE_COUNT;
  const rigidModes = buildFreeComponentRigidDisplacementModes(
    options.mesh,
    options.lumpedMasses,
  );
  const rawModes = buildLowFrequencyTrainingPoses(
    options.mesh,
    options.restSystem,
    requestedDeformationalModes + rigidModes.length,
    inverseIterations,
  );
  const modes = rigidModes.length === 0
    ? rawModes
    : removeRigidDisplacementModes(
        rawModes,
        rigidModes,
        requestedDeformationalModes,
      );
  if (modes.length === 0) {
    throw new Error("Nonlinear Cubature requires at least one low-frequency mode.");
  }

  const trainingModes = modes.slice(0, modeCount);
  const training: NonlinearCubaturePose[] = [];
  for (const mode of trainingModes) {
    for (const sign of [1, -1] as const) {
      const direction = Float64Array.from(mode, (value) => sign * value);
      training.push(
        createScaledPose(
          options,
          "training-mode",
          training.length,
          direction,
          trainingAmplitude,
          0,
          sceneScale,
          minimumDeterminant,
        ),
      );
    }
  }

  const validation: NonlinearCubaturePose[] = Array.from(
    { length: validationCount },
    (_unused, index) => {
      const combination = new Float64Array(vertexCount * 3);
      for (let modeIndex = 0; modeIndex < modes.length; modeIndex += 1) {
        const coefficient =
          Math.sin((index + 1) * (modeIndex + 1) * 0.754_877_666) +
          0.5 * Math.cos((index + 1) * (modeIndex + 3) * 0.569_840_291);
        const mode = modes[modeIndex]!;
        for (let entry = 0; entry < combination.length; entry += 1) {
          combination[entry] += coefficient * mode[entry]!;
        }
      }
      return createScaledPose(
        options,
        "validation-combination",
        index,
        combination,
        index < NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE_START
          ? validationLowAmplitude
          : validationHighAmplitude,
        NONLINEAR_CUBATURE_VALIDATION_PREDICTED_BLEND,
        sceneScale,
        minimumDeterminant,
      );
    },
  );

  if (!hasFixedVertices) {
    const center = restCenterOfMass(options.mesh.positions, options.lumpedMasses);
    for (
      let rotationIndex = 0;
      rotationIndex < NONLINEAR_CUBATURE_VALIDATION_ROTATIONS.length;
      rotationIndex += 1
    ) {
      const source = validation[rotationIndex]!;
      const index = validation.length;
      validation.push(
        evaluatePose(
          options,
          `phase1.nonlinear-cubature-validation/${index
            .toString()
            .padStart(2, "0")}`,
          "validation-rotated",
          index,
          rotatePositions(
            source.positions,
            center,
            NONLINEAR_CUBATURE_VALIDATION_ROTATIONS[rotationIndex]!,
          ),
          rotatePositions(
            source.predictedPositions,
            center,
            NONLINEAR_CUBATURE_VALIDATION_ROTATIONS[rotationIndex]!,
          ),
          source.perturbationAmplitude,
          minimumDeterminant,
        ),
      );
    }
  }

  return {
    version: NONLINEAR_CUBATURE_CORPUS_VERSION,
    sceneScale,
    training: Object.freeze(training),
    validation: Object.freeze(validation),
  };
}
