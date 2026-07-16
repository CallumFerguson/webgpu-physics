import type {
  LinearMaterial,
  SceneDefinition,
  TetrahedralMesh,
  Vec3,
} from "../simulation/cpu/types";
import { generateRegularCuboidMesh } from "../simulation/cpu/mesh";

/** Frozen IDs from manifests/phase1-scenes.v1.json. */
export const PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID =
  "phase1.stable-neo-hookean-single-tetrahedron" as const;
export const PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID =
  "phase1.nonlinear-cubature-beam" as const;
export const PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID =
  "phase1.stable-neo-hookean-poses" as const;
export const PHASE1_STABLE_NEO_HOOKEAN_CORPUS_VERSION =
  "phase1-v1" as const;
export const PHASE1_STABLE_NEO_HOOKEAN_GENERATOR_VERSION = "1" as const;
export const PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED = 1_048_583;
export const PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT = 64;
export const PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX = 9;
export const PHASE1_STABLE_NEO_HOOKEAN_SEEDED_ENTRY_SCALE = 0.35;
export const PHASE1_STABLE_NEO_HOOKEAN_MINIMUM_SEEDED_DETERMINANT = 0.05;
export const PHASE1_STABLE_NEO_HOOKEAN_MAXIMUM_SEEDED_DETERMINANT = 2.5;
export const PHASE1_STABLE_NEO_HOOKEAN_TIMESTEP = 1 / 60;

export const PHASE1_STABLE_NEO_HOOKEAN_CPU_TOLERANCES = {
  restEnergyAndForce: 1e-10,
  rigidEnergyAndForce: 1e-8,
  gradientRelativeError: 1e-5,
  hessianRelativeError: 1e-4,
} as const;

export const PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES = {
  restEnergyAndForce: 1e-5,
  rigidEnergyAndForce: 1e-4,
  cpuParityRelativeError: 1e-3,
} as const;

export interface Phase1StableNeoHookeanMaterial extends LinearMaterial {
  readonly model: "stable-neo-hookean";
}

export const PHASE1_STABLE_NEO_HOOKEAN_MATERIAL: Phase1StableNeoHookeanMaterial = {
  name: "phase1-stable-neo-hookean-reference",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 80_000,
  poissonRatio: 0.3,
  color: [0.55, 0.38, 0.95, 1],
};

export const PHASE1_NONLINEAR_CUBATURE_MATERIAL: Phase1StableNeoHookeanMaterial = {
  ...PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
  name: "phase1-nonlinear-cubature-reference",
  youngModulus: 5_000,
};

export const PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS = new Float64Array([
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
]);

export const PHASE1_STABLE_NEO_HOOKEAN_RIGID_EULER_RADIANS = [
  [Math.PI / 2, 0, 0],
  [0, Math.PI / 2, 0],
  [0, 0, Math.PI / 2],
] as const satisfies readonly Vec3[];

export const PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS = [
  0.5, 0.1, 0.01,
] as const;

export const PHASE1_STABLE_NEO_HOOKEAN_SHEAR_GRADIENT = new Float64Array([
  1, 0.55, -0.15,
  0, 1, 0.25,
  0, 0, 1,
]);

export const PHASE1_STABLE_NEO_HOOKEAN_STRETCH_GRADIENT = new Float64Array([
  1.65, 0, 0,
  0, 0.72, 0,
  0, 0, 0.95,
]);

export type Phase1StableNeoHookeanPoseKind =
  | "rest"
  | "rigid"
  | "compression"
  | "shear"
  | "stretch"
  | "seeded";

export interface Phase1StableNeoHookeanPose {
  readonly id: string;
  readonly index: number;
  readonly kind: Phase1StableNeoHookeanPoseKind;
  /** Row-major affine deformation gradient applied about rest vertex zero. */
  readonly deformationGradient: Float64Array;
  readonly positions: Float64Array;
  readonly determinant: number;
}

function createCorpusRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b_79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function determinant3(matrix: ArrayLike<number>): number {
  return (
    matrix[0]! * (matrix[4]! * matrix[8]! - matrix[5]! * matrix[7]!) -
    matrix[1]! * (matrix[3]! * matrix[8]! - matrix[5]! * matrix[6]!) +
    matrix[2]! * (matrix[3]! * matrix[7]! - matrix[4]! * matrix[6]!)
  );
}

function eulerRotation([x, y, z]: Vec3): Float64Array {
  const cx = Math.cos(x);
  const sx = Math.sin(x);
  const cy = Math.cos(y);
  const sy = Math.sin(y);
  const cz = Math.cos(z);
  const sz = Math.sin(z);
  return new Float64Array([
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
    -sy, cy * sx, cy * cx,
  ]);
}

function positionsFromDeformationGradient(
  deformationGradient: ArrayLike<number>,
): Float64Array {
  const positions = new Float64Array(
    PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS.length,
  );
  for (
    let vertex = 0;
    vertex < PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS.length / 3;
    vertex += 1
  ) {
    const source = vertex * 3;
    const x = PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS[source]!;
    const y = PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS[source + 1]!;
    const z = PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS[source + 2]!;
    positions[source] =
      deformationGradient[0]! * x +
      deformationGradient[1]! * y +
      deformationGradient[2]! * z;
    positions[source + 1] =
      deformationGradient[3]! * x +
      deformationGradient[4]! * y +
      deformationGradient[5]! * z;
    positions[source + 2] =
      deformationGradient[6]! * x +
      deformationGradient[7]! * y +
      deformationGradient[8]! * z;
  }
  return positions;
}

function pose(
  index: number,
  kind: Phase1StableNeoHookeanPoseKind,
  deformationGradient: ArrayLike<number>,
): Phase1StableNeoHookeanPose {
  const gradient = Float64Array.from(deformationGradient);
  return {
    id: `${PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID}/${index
      .toString()
      .padStart(2, "0")}`,
    index,
    kind,
    deformationGradient: gradient,
    positions: positionsFromDeformationGradient(gradient),
    determinant: determinant3(gradient),
  };
}

function explicitPhase1Poses(): Phase1StableNeoHookeanPose[] {
  const poses = [
    pose(0, "rest", [1, 0, 0, 0, 1, 0, 0, 0, 1]),
    ...PHASE1_STABLE_NEO_HOOKEAN_RIGID_EULER_RADIANS.map(
      (rotation, offset) => pose(offset + 1, "rigid", eulerRotation(rotation)),
    ),
    ...PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS.map(
      (determinant, offset) =>
        pose(offset + 4, "compression", [
          determinant, 0, 0,
          0, 1, 0,
          0, 0, 1,
        ]),
    ),
    pose(7, "shear", PHASE1_STABLE_NEO_HOOKEAN_SHEAR_GRADIENT),
    pose(8, "stretch", PHASE1_STABLE_NEO_HOOKEAN_STRETCH_GRADIENT),
  ];
  if (poses.length !== PHASE1_STABLE_NEO_HOOKEAN_SEEDED_START_INDEX) {
    throw new Error("Phase 1 explicit pose layout drifted.");
  }
  return poses;
}

/**
 * Frozen v1 material-oracle corpus. The first nine cases are named edge cases;
 * the remainder are seeded affine strains accepted only in the recorded
 * positive-determinant interval.
 */
export function generatePhase1StableNeoHookeanPoseCorpus(): readonly Phase1StableNeoHookeanPose[] {
  const poses = explicitPhase1Poses();
  const random = createCorpusRandom(
    PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED,
  );
  let attempts = 0;
  while (poses.length < PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT) {
    attempts += 1;
    if (attempts > 100_000) {
      throw new Error("Phase 1 corpus generator exhausted its attempt budget.");
    }
    const gradient = new Float64Array(9);
    for (let entry = 0; entry < gradient.length; entry += 1) {
      gradient[entry] =
        (entry % 4 === 0 ? 1 : 0) +
        (random() * 2 - 1) *
          PHASE1_STABLE_NEO_HOOKEAN_SEEDED_ENTRY_SCALE;
    }
    const determinant = determinant3(gradient);
    if (
      determinant < PHASE1_STABLE_NEO_HOOKEAN_MINIMUM_SEEDED_DETERMINANT ||
      determinant > PHASE1_STABLE_NEO_HOOKEAN_MAXIMUM_SEEDED_DETERMINANT
    ) {
      continue;
    }
    poses.push(pose(poses.length, "seeded", gradient));
  }
  return poses;
}

function explicitOracleMesh(): TetrahedralMesh {
  return {
    positions: PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS.slice(),
    tetrahedra: new Uint32Array([0, 1, 2, 3]),
    materialIds: new Uint16Array([0]),
    fixed: new Uint8Array([1, 0, 0, 0]),
    bodyIds: new Uint16Array([0, 0, 0, 0]),
  };
}

/** Build the private oracle definition without adding it to the public demos. */
export function buildPhase1StableNeoHookeanOracleDefinition(): SceneDefinition {
  return {
    id: PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID,
    title: "Canonical stable Neo-Hookean single-tetrahedron oracle",
    description:
      "The frozen Phase 1 CPU/GPU material parity fixture; it is intentionally not a public demo scene.",
    mesh: explicitOracleMesh(),
    materials: [PHASE1_STABLE_NEO_HOOKEAN_MATERIAL],
    settings: {
      timestep: PHASE1_STABLE_NEO_HOOKEAN_TIMESTEP,
      gravity: [0, 0, 0],
      floorY: 0,
      solverIterations: 1,
      cubatureSamples: 4,
    },
    camera: {
      eye: [2.5, 2, 3.5],
      target: [0.3, 0.3, 0.25],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 3,
      label: "stable Neo-Hookean oracle apex",
      expectedMotion: [1, 1, 1],
    },
  };
}

/** Private multi-element fixture for nonlinear Cubature training and parity. */
export function buildPhase1NonlinearCubatureDefinition(): SceneDefinition {
  return {
    id: PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID,
    title: "Canonical nonlinear Cubature beam",
    description:
      "A private twelve-tetrahedron stable Neo-Hookean solid used for nonredundant nonlinear Cubature training and held-out update validation.",
    mesh: generateRegularCuboidMesh({
      cells: [2, 1, 1],
      origin: [0, 0, 0],
      size: [1, 1, 1],
      fixed: (_position, [x, y]) => x === 0 && y === 0,
    }),
    materials: [PHASE1_NONLINEAR_CUBATURE_MATERIAL],
    settings: {
      timestep: PHASE1_STABLE_NEO_HOOKEAN_TIMESTEP,
      gravity: [0, 0, 0],
      floorY: -10,
      solverIterations: 1,
      cubatureSamples: 6,
    },
    camera: {
      eye: [4.5, 3, 5.5],
      target: [0.5, 0.5, 0.5],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 11,
      label: "nonlinear Cubature beam tip",
      expectedMotion: [0, -1, 0],
    },
  };
}
