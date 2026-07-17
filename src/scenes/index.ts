import {
  appendTetrahedralMeshes,
  buildStaticIpcContactCandidates,
  buildPrecomputedScene,
  computeStableNeoHookeanParameters,
  precomputeQuadraticDihedralRest,
  precomputeStVKTriangleRest,
  generateRegularCuboidMesh,
  getTetrahedronCount,
  getVertexCount,
  transformTetrahedralMesh,
  type LinearMaterial,
  type PrecomputedScene,
  type SceneDefinition,
  type TetrahedralMesh,
} from "../simulation/cpu";
import {
  JGS2_MATERIAL_COROTATED_LINEAR,
  JGS2_MATERIAL_STABLE_NEO_HOOKEAN,
  validateJGS2GpuInput,
  type JGS2GpuInput,
} from "../simulation/gpu/layout";
import type { JGS2GpuClothInput } from "../simulation/gpu/cloth-layout";
import {
  PHASE0_FORCE_FREE_BASE_STATE,
  PHASE0_FORCE_FREE_FIXTURE_ID,
  buildPhase0ForceFreeDefinition,
  buildPhase0ForceFreeInitialPositions,
  buildPhase0ForceFreeScene,
  buildPhase0RigidVelocityField,
  type Phase0RigidInitialState,
} from "./canonical-phase0";

export {
  PHASE0_FORCE_FREE_BASE_STATE,
  PHASE0_FORCE_FREE_CORPUS_CASE_COUNT,
  PHASE0_FORCE_FREE_CORPUS_GENERATOR_VERSION,
  PHASE0_FORCE_FREE_CORPUS_ID,
  PHASE0_FORCE_FREE_CORPUS_SEED,
  PHASE0_FORCE_FREE_FIXTURE_ID,
  PHASE0_FORCE_FREE_FRAME_COUNT,
  PHASE0_FORCE_FREE_INITIAL_EULER,
  PHASE0_FORCE_FREE_ITERATIONS,
  PHASE0_FORCE_FREE_TIMESTEP,
  PHASE0_REFERENCE_MATERIAL,
  buildPhase0ForceFreeDefinition,
  buildPhase0ForceFreeInitialPositions,
  buildPhase0ForceFreeScene,
  buildPhase0RigidVelocityField,
  generatePhase0ForceFreeInitialStateCorpus,
  type Phase0RigidInitialState,
} from "./canonical-phase0";

export {
  PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID,
  PHASE1_NONLINEAR_CUBATURE_MATERIAL,
  PHASE1_STABLE_NEO_HOOKEAN_COMPRESSION_DETERMINANTS,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_CASE_COUNT,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_ID,
  PHASE1_STABLE_NEO_HOOKEAN_CORPUS_SEED,
  PHASE1_STABLE_NEO_HOOKEAN_FIXTURE_ID,
  PHASE1_STABLE_NEO_HOOKEAN_GPU_TOLERANCES,
  PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
  PHASE1_STABLE_NEO_HOOKEAN_REST_POSITIONS,
  PHASE1_STABLE_NEO_HOOKEAN_TIMESTEP,
  buildPhase1StableNeoHookeanOracleDefinition,
  buildPhase1NonlinearCubatureDefinition,
  generatePhase1StableNeoHookeanPoseCorpus,
  type Phase1StableNeoHookeanPose,
} from "./phase1";

export const SCENE_IDS = [
  "minimal",
  "stiffness",
  "drop",
  "contact",
  "cloth",
  "stress",
] as const;
export type SceneId = (typeof SCENE_IDS)[number];
export const DEFAULT_SCENE_ID: SceneId = "minimal";

/**
 * Test-only scene identifier. It is deliberately excluded from SCENE_IDS so
 * the four public demo links remain stable.
 */
export const FORCE_FREE_CONSERVATION_FIXTURE_ID =
  PHASE0_FORCE_FREE_FIXTURE_ID;

const SOFT_MATERIAL: LinearMaterial = {
  name: "soft rubber",
  density: 850,
  youngModulus: 35_000,
  poissonRatio: 0.35,
  color: [0.13, 0.82, 0.7, 1],
};

const MINIMAL_MATERIAL: LinearMaterial = {
  ...SOFT_MATERIAL,
  name: "stable Neo-Hookean soft rubber",
  model: "stable-neo-hookean",
};

const STIFF_MATERIAL: LinearMaterial = {
  name: "stiff rubber",
  density: 950,
  youngModulus: 280_000,
  poissonRatio: 0.32,
  color: [0.98, 0.38, 0.25, 1],
};

const DROP_MATERIAL: LinearMaterial = {
  name: "drop block",
  model: "stable-neo-hookean",
  density: 900,
  youngModulus: 70_000,
  poissonRatio: 0.36,
  color: [0.26, 0.63, 0.96, 1],
};

const CONTACT_SLAB_MATERIAL: LinearMaterial = {
  name: "pinned contact slab",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 120_000,
  poissonRatio: 0.34,
  color: [0.34, 0.4, 0.48, 1],
};

const CONTACT_LOWER_MATERIAL: LinearMaterial = {
  name: "lower contact block",
  model: "stable-neo-hookean",
  density: 900,
  youngModulus: 65_000,
  poissonRatio: 0.36,
  color: [0.15, 0.74, 0.66, 1],
};

const CONTACT_UPPER_MATERIAL: LinearMaterial = {
  name: "sliding contact block",
  model: "stable-neo-hookean",
  density: 850,
  youngModulus: 75_000,
  poissonRatio: 0.35,
  color: [0.97, 0.48, 0.2, 1],
};

const CONTACT_SELF_MATERIAL: LinearMaterial = {
  name: "self-contact pair",
  model: "stable-neo-hookean",
  density: 825,
  youngModulus: 62_000,
  poissonRatio: 0.35,
  color: [0.72, 0.38, 0.92, 1],
};

const CLOTH_COLLIDER_MATERIAL: LinearMaterial = {
  name: "cloth collider",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 110_000,
  poissonRatio: 0.34,
  color: [0.32, 0.39, 0.5, 1],
};

export type MinimalScriptedTargetPhase =
  | "waiting"
  | "pulling"
  | "holding"
  | "released";

export interface MinimalScriptedTargetState {
  readonly vertex: number;
  readonly phase: MinimalScriptedTargetPhase;
  readonly active: boolean;
  readonly position: readonly [number, number, number];
  readonly stiffness: number;
  /** Changes only when the queue-visible objective record must be updated. */
  readonly revision: number;
}

export const MINIMAL_SCRIPTED_TARGET_VERTEX = 1;
export const MINIMAL_SCRIPTED_TARGET_PULL_START_FRAME = 48;
export const MINIMAL_SCRIPTED_TARGET_PULL_END_FRAME = 90;
export const MINIMAL_SCRIPTED_TARGET_RELEASE_FRAME = 108;
const MINIMAL_SCRIPTED_TARGET_KEYFRAME_INTERVAL = 3;
const MINIMAL_SCRIPTED_TARGET_STIFFNESS = 30_000;
const MINIMAL_SCRIPTED_TARGET_REST_POSITION = [1, 1.35, -0.4] as const;
const MINIMAL_SCRIPTED_TARGET_PULL_OFFSET = [0.45, 0.65, 0.25] as const;

/**
 * Small deterministic user-manipulation stand-in for the public cantilever.
 * The target moves on coarse keyframes so deterministic test stepping can
 * batch identical objective records, then becomes a true zero-stiffness
 * release without touching the simulated position or velocity buffers.
 */
export function minimalScriptedTargetAtFrame(
  frame: number,
): MinimalScriptedTargetState {
  if (!Number.isSafeInteger(frame) || frame < 0) {
    throw new RangeError("Minimal scripted-target frame must be nonnegative.");
  }
  if (frame < MINIMAL_SCRIPTED_TARGET_PULL_START_FRAME) {
    return {
      vertex: MINIMAL_SCRIPTED_TARGET_VERTEX,
      phase: "waiting",
      active: false,
      position: MINIMAL_SCRIPTED_TARGET_REST_POSITION,
      stiffness: 0,
      revision: 0,
    };
  }

  const pullFrame = Math.min(frame, MINIMAL_SCRIPTED_TARGET_PULL_END_FRAME);
  const keyframe = Math.min(
    MINIMAL_SCRIPTED_TARGET_PULL_END_FRAME,
    MINIMAL_SCRIPTED_TARGET_PULL_START_FRAME +
      Math.floor(
        (pullFrame - MINIMAL_SCRIPTED_TARGET_PULL_START_FRAME) /
          MINIMAL_SCRIPTED_TARGET_KEYFRAME_INTERVAL,
      ) * MINIMAL_SCRIPTED_TARGET_KEYFRAME_INTERVAL,
  );
  const linearProgress =
    (keyframe - MINIMAL_SCRIPTED_TARGET_PULL_START_FRAME) /
    (MINIMAL_SCRIPTED_TARGET_PULL_END_FRAME -
      MINIMAL_SCRIPTED_TARGET_PULL_START_FRAME);
  const progress = linearProgress * linearProgress * (3 - 2 * linearProgress);
  const position: readonly [number, number, number] = [
    MINIMAL_SCRIPTED_TARGET_REST_POSITION[0] +
      MINIMAL_SCRIPTED_TARGET_PULL_OFFSET[0] * progress,
    MINIMAL_SCRIPTED_TARGET_REST_POSITION[1] +
      MINIMAL_SCRIPTED_TARGET_PULL_OFFSET[1] * progress,
    MINIMAL_SCRIPTED_TARGET_REST_POSITION[2] +
      MINIMAL_SCRIPTED_TARGET_PULL_OFFSET[2] * progress,
  ];

  if (frame >= MINIMAL_SCRIPTED_TARGET_RELEASE_FRAME) {
    return {
      vertex: MINIMAL_SCRIPTED_TARGET_VERTEX,
      phase: "released",
      active: false,
      position,
      stiffness: 0,
      revision: MINIMAL_SCRIPTED_TARGET_RELEASE_FRAME,
    };
  }
  return {
    vertex: MINIMAL_SCRIPTED_TARGET_VERTEX,
    phase:
      frame < MINIMAL_SCRIPTED_TARGET_PULL_END_FRAME ? "pulling" : "holding",
    active: true,
    position,
    stiffness: MINIMAL_SCRIPTED_TARGET_STIFFNESS,
    revision: keyframe + 1,
  };
}

function minimalDefinition(): SceneDefinition {
  const mesh = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-1, 1.35, -0.4],
    size: [2, 0.8, 0.8],
    fixed: (_position, [x]) => x === 0,
  });

  return {
    id: "minimal",
    title: "Minimal cantilever",
    description:
      "A left-face-fixed tetrahedral beam sags under gravity; its left end must remain still while the free tip moves down.",
    mesh,
    materials: [MINIMAL_MATERIAL],
    settings: {
      timestep: 1 / 30,
      gravity: [0, -9.81, 0],
      floorY: 0,
      solverIterations: 7,
      cubatureSamples: 6,
    },
    camera: {
      eye: [4.8, 3.2, 6.2],
      target: [0, 1.35, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 1,
      label: "free tip",
      expectedMotion: [0, -1, 0],
    },
  };
}

function stiffnessDefinition(): SceneDefinition {
  const soft = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-1, 1.35, -1.05],
    size: [2, 0.7, 0.7],
    materialId: 0,
    bodyId: 0,
    fixed: (_position, [x]) => x === 0,
  });
  const stiff = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-1, 1.35, 0.35],
    size: [2, 0.7, 0.7],
    materialId: 1,
    bodyId: 1,
    fixed: (_position, [x]) => x === 0,
  });

  return {
    id: "stiffness",
    title: "Soft versus stiff",
    description:
      "Identical cantilevers receive identical gravity loading; the teal soft beam should bend much farther than the red stiff beam.",
    mesh: appendTetrahedralMeshes([soft, stiff]),
    materials: [SOFT_MATERIAL, STIFF_MATERIAL],
    settings: {
      timestep: 1 / 30,
      gravity: [0, -9.81, 0],
      floorY: 0,
      solverIterations: 9,
      cubatureSamples: 6,
    },
    camera: {
      eye: [5.2, 3.5, 7.6],
      target: [0, 1.3, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 1,
      label: "soft free tip",
      expectedMotion: [0, -1, 0],
    },
  };
}

function dropDefinition(): SceneDefinition {
  const centered = generateRegularCuboidMesh({
    // The public roadmap demo deliberately stays at six tetrahedra so the
    // nonlinear Cubature may sample the complete tiny mesh at runtime.
    cells: [1, 1, 1],
    origin: [-0.75, -0.5, -0.45],
    size: [1.5, 1, 0.9],
    materialId: 0,
    bodyId: 0,
  });
  const mesh = transformTetrahedralMesh(centered, {
    translation: [0, 2.45, 0],
    rotationEuler: [0, 0.12, 0],
  });

  return {
    id: "drop",
    title: "Falling deformable block",
    description:
      "A rotated tetrahedral block falls onto the floor and should visibly compress and rebound without passing through it.",
    mesh,
    materials: [DROP_MATERIAL],
    settings: {
      timestep: 1 / 60,
      gravity: [0, -9.81, 0],
      floorY: 0,
      solverIterations: 11,
      cubatureSamples: 6,
    },
    camera: {
      eye: [4.2, 3.4, 6.5],
      target: [0, 1.2, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 0,
      label: "falling block corner",
      expectedMotion: [0, -1, 0],
    },
  };
}

function contactDefinition(): SceneDefinition {
  const slab = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-1.4, -0.18, -0.7],
    size: [2.8, 0.18, 1.4],
    materialId: 0,
    bodyId: 0,
    fixed: () => true,
  });
  const lowerBlock = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-0.45, 0.08, -0.38],
    size: [0.9, 0.56, 0.76],
    materialId: 1,
    bodyId: 1,
  });
  const upperBlock = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-0.42, 0.74, -0.34],
    size: [0.84, 0.52, 0.68],
    materialId: 2,
    bodyId: 2,
  });
  // Two disconnected components deliberately share one body id. The upper
  // component falls onto its pinned mate, exercising topology-filtered
  // same-body candidates as a compact self-contact regression.
  const selfContactAnchor: TetrahedralMesh = {
    positions: new Float64Array([
      -1.22, 0.02, -0.58,
      -1.22, 0.24, -0.58,
      -1.22, 0.24, -0.16,
      -0.76, 0.24, -0.58,
    ]),
    tetrahedra: new Uint32Array([0, 1, 2, 3]),
    materialIds: new Uint16Array([3]),
    fixed: new Uint8Array([1, 1, 1, 1]),
    bodyIds: new Uint16Array([3, 3, 3, 3]),
  };
  const selfContactFaller: TetrahedralMesh = {
    positions: new Float64Array([
      -1.2, 0.28, -0.56,
      -1.2, 0.28, -0.18,
      -0.78, 0.28, -0.56,
      -1.2, 0.5, -0.56,
    ]),
    tetrahedra: new Uint32Array([0, 1, 2, 3]),
    materialIds: new Uint16Array([3]),
    fixed: new Uint8Array(4),
    bodyIds: new Uint16Array([3, 3, 3, 3]),
  };

  return {
    id: "contact",
    title: "IPC collision and friction",
    description:
      "Two deformable blocks collide and slide on a pinned slab while a small same-body pair exercises self-contact; the floor plane is visual only.",
    mesh: appendTetrahedralMeshes([
      slab,
      lowerBlock,
      upperBlock,
      selfContactAnchor,
      selfContactFaller,
    ]),
    materials: [
      CONTACT_SLAB_MATERIAL,
      CONTACT_LOWER_MATERIAL,
      CONTACT_UPPER_MATERIAL,
      CONTACT_SELF_MATERIAL,
    ],
    settings: {
      timestep: 1 / 60,
      gravity: [0, -9.81, 0],
      floorY: -0.18,
      solverIterations: 7,
      cubatureSamples: 6,
      initialBodyVelocities: [
        [0, 0, 0],
        [0, 0, 0],
        [0.8, 0, 0],
        [0, 0, 0],
      ],
    },
    camera: {
      eye: [3.4, 2.25, 4.7],
      target: [0.15, 0.46, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 17,
      label: "sliding upper block corner",
      expectedMotion: [1, -1, 0],
    },
  };
}

function clothDefinition(): SceneDefinition {
  const collider = generateRegularCuboidMesh({
    cells: [1, 1, 1],
    origin: [-0.65, 0, -0.45],
    size: [0.9, 0.5, 0.9],
    materialId: 0,
    bodyId: 0,
    fixed: () => true,
  });
  const cells = 4;
  const clothVertexCount = (cells + 1) * (cells + 1);
  const clothPositions = new Float64Array(clothVertexCount * 3);
  const clothFixed = new Uint8Array(clothVertexCount);
  const clothBodyIds = new Uint16Array(clothVertexCount);
  clothBodyIds.fill(1);
  const vertex = (x: number, z: number) => x + z * (cells + 1);
  for (let z = 0; z <= cells; z += 1) {
    for (let x = 0; x <= cells; x += 1) {
      const index = vertex(x, z);
      // Pre-fold two columns back over the sheet with a narrow feasible IPC
      // gap. The rest metric and rest dihedrals are computed from this shape,
      // while gravity makes the free flap drape over the collider and exercise
      // nonlocal same-sheet candidates deterministically.
      const folded = x >= 3;
      const worldX = folded ? 0.8 - 0.4 * x : -0.8 + 0.4 * x;
      clothPositions.set(
        [worldX, folded ? 1.229 : 1.15, -0.8 + (1.6 * z) / cells],
        index * 3,
      );
      clothFixed[index] = z === 0 && (x === 0 || x === 2) ? 1 : 0;
    }
  }
  const clothMesh: TetrahedralMesh = {
    positions: clothPositions,
    tetrahedra: new Uint32Array(),
    materialIds: new Uint16Array(),
    fixed: clothFixed,
    bodyIds: clothBodyIds,
  };
  const clothOffset = getVertexCount(collider);
  const triangles: number[] = [];
  for (let z = 0; z < cells; z += 1) {
    for (let x = 0; x < cells; x += 1) {
      const i00 = clothOffset + vertex(x, z);
      const i10 = clothOffset + vertex(x + 1, z);
      const i01 = clothOffset + vertex(x, z + 1);
      const i11 = clothOffset + vertex(x + 1, z + 1);
      triangles.push(i00, i01, i10, i10, i01, i11);
    }
  }
  return {
    id: "cloth",
    title: "Pinned cloth drape",
    description:
      "A two-point-pinned, pre-folded StVK triangle sheet bends and drapes over a fixed collider through the same JGS2 and IPC path.",
    mesh: appendTetrahedralMeshes([collider, clothMesh]),
    cloth: {
      triangles: Uint32Array.from(triangles),
      density: 480,
      youngModulus: 9_000,
      poissonRatio: 0.3,
      thickness: 0.012,
      bendingStiffness: 0.045,
      color: [0.72, 0.28, 0.88, 1],
    },
    materials: [CLOTH_COLLIDER_MATERIAL],
    settings: {
      timestep: 1 / 120,
      gravity: [0, -9.81, 0],
      floorY: 0,
      solverIterations: 7,
      cubatureSamples: 4,
    },
    camera: {
      eye: [3.1, 2.25, 4.1],
      target: [0, 0.65, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: clothOffset + vertex(2, 2),
      label: "cloth center",
      expectedMotion: [0, -1, 0],
    },
  };
}

function stressBlock(body: number): TetrahedralMesh {
  const base = generateRegularCuboidMesh({
    cells: [2, 1, 1],
    origin: [-0.6, -0.32, -0.32],
    size: [1.2, 0.64, 0.64],
    materialId: body % 2,
    bodyId: body,
  });
  const column = body % 3;
  const lane = Math.floor(body / 3);

  return transformTetrahedralMesh(base, {
    translation: [
      (column - 1) * 1.45,
      1.25 + 0.08 * (body % 2),
      (lane - 0.5) * 1.25,
    ],
    rotationEuler: [0.04 * body, 0.08 * (body - 2), 0.1 * (body % 2)],
  });
}

function stressDefinition(): SceneDefinition {
  const blocks = Array.from({ length: 6 }, (_unused, body) => stressBlock(body));
  const mesh = appendTetrahedralMeshes(blocks);

  return {
    id: "stress",
    title: "Multi-block stress test",
    description:
      "Six alternating soft and stiff blocks fall in separated lanes, exercising parallel solves, body IDs, and floor contact without unsupported body-body contact.",
    mesh,
    materials: [SOFT_MATERIAL, STIFF_MATERIAL],
    settings: {
      timestep: 1 / 120,
      gravity: [0, -9.81, 0],
      floorY: 0,
      solverIterations: 17,
      cubatureSamples: 6,
    },
    camera: {
      eye: [5.6, 4.1, 8.2],
      target: [0, 1.15, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 0,
      label: "first block corner",
      expectedMotion: [0, -1, 0],
    },
  };
}

export function buildForceFreeConservationDefinition(): SceneDefinition {
  return buildPhase0ForceFreeDefinition();
}

export function buildSceneDefinition(id: SceneId): SceneDefinition {
  switch (id) {
    case "minimal":
      return minimalDefinition();
    case "stiffness":
      return stiffnessDefinition();
    case "drop":
      return dropDefinition();
    case "contact":
      return contactDefinition();
    case "cloth":
      return clothDefinition();
    case "stress":
      return stressDefinition();
  }
}

export function buildScene(id: SceneId = DEFAULT_SCENE_ID): PrecomputedScene {
  return buildPrecomputedScene(buildSceneDefinition(id));
}

export function buildForceFreeConservationScene(): PrecomputedScene {
  return buildPhase0ForceFreeScene();
}

function buildAdjacency(mesh: TetrahedralMesh): {
  readonly starts: Uint32Array;
  readonly counts: Uint32Array;
  readonly tetrahedra: Uint32Array;
} {
  const vertexCount = getVertexCount(mesh);
  const counts = new Uint32Array(vertexCount);
  for (const vertex of mesh.tetrahedra) {
    counts[vertex] += 1;
  }
  const starts = new Uint32Array(vertexCount);
  let total = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    starts[vertex] = total;
    total += counts[vertex]!;
  }
  const tetrahedra = new Uint32Array(total);
  const cursors = starts.slice();
  for (let tetrahedron = 0; tetrahedron < getTetrahedronCount(mesh); tetrahedron += 1) {
    for (let localVertex = 0; localVertex < 4; localVertex += 1) {
      const vertex = mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
      tetrahedra[cursors[vertex]!] = tetrahedron;
      cursors[vertex] += 1;
    }
  }
  return { starts, counts, tetrahedra };
}

function computeVertexColors(scene: PrecomputedScene): Float32Array {
  const vertexCount = getVertexCount(scene.mesh);
  const colors = new Float32Array(vertexCount * 4);
  const contributionCounts = new Uint32Array(vertexCount);
  for (let tetrahedron = 0; tetrahedron < getTetrahedronCount(scene.mesh); tetrahedron += 1) {
    const material = scene.materials[scene.mesh.materialIds[tetrahedron]!]!;
    for (let localVertex = 0; localVertex < 4; localVertex += 1) {
      const vertex = scene.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
      for (let channel = 0; channel < 4; channel += 1) {
        colors[vertex * 4 + channel] += material.color[channel]!;
      }
      contributionCounts[vertex] += 1;
    }
  }
  if (scene.cloth) {
    for (const vertex of scene.cloth.triangles) {
      for (let channel = 0; channel < 4; channel += 1) {
        colors[vertex * 4 + channel] += scene.cloth.color[channel]!;
      }
      contributionCounts[vertex] += 1;
    }
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (let channel = 0; channel < 4; channel += 1) {
      colors[vertex * 4 + channel] /= contributionCounts[vertex]!;
    }
  }
  return colors;
}

function packedRestPositions(
  positions: Float64Array,
  indices: readonly number[],
): Float64Array {
  const packed = new Float64Array(indices.length * 3);
  for (let local = 0; local < indices.length; local += 1) {
    const vertex = indices[local]!;
    packed.set(positions.subarray(vertex * 3, vertex * 3 + 3), local * 3);
  }
  return packed;
}

/** Build the paper's compact StVK-triangle/quadratic-dihedral GPU records. */
function buildJGS2GpuClothInput(
  scene: PrecomputedScene,
): JGS2GpuClothInput | undefined {
  const cloth = scene.cloth;
  if (!cloth) return undefined;
  const triangleCount = cloth.triangles.length / 3;
  const triangleInverseRestBases = new Float32Array(triangleCount * 4);
  const triangleRestAreas = new Float32Array(triangleCount);
  type OrientedHalfEdge = {
    readonly first: number;
    readonly second: number;
    readonly opposite: number;
  };
  const openEdges = new Map<string, OrientedHalfEdge>();
  const closedEdges = new Set<string>();
  const hingeIndices: number[] = [];
  const addHalfEdge = (
    first: number,
    second: number,
    opposite: number,
  ): void => {
    const key = `${Math.min(first, second)}:${Math.max(first, second)}`;
    if (closedEdges.has(key)) {
      throw new RangeError(`Cloth edge ${key} is non-manifold.`);
    }
    const previous = openEdges.get(key);
    if (!previous) {
      openEdges.set(key, { first, second, opposite });
      return;
    }
    if (previous.first !== second || previous.second !== first) {
      throw new RangeError(
        `Cloth triangles sharing edge ${key} must have consistent orientation.`,
      );
    }
    hingeIndices.push(
      previous.first,
      previous.second,
      previous.opposite,
      opposite,
    );
    openEdges.delete(key);
    closedEdges.add(key);
  };

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const base = triangle * 3;
    const i0 = cloth.triangles[base]!;
    const i1 = cloth.triangles[base + 1]!;
    const i2 = cloth.triangles[base + 2]!;
    const rest = precomputeStVKTriangleRest(
      packedRestPositions(scene.mesh.positions, [i0, i1, i2]),
    );
    triangleInverseRestBases.set(rest.inverseRestBasis, triangle * 4);
    triangleRestAreas[triangle] = rest.restArea;
    addHalfEdge(i0, i1, i2);
    addHalfEdge(i1, i2, i0);
    addHalfEdge(i2, i0, i1);
  }

  const hinges = Uint32Array.from(hingeIndices);
  const hingeCount = hinges.length / 4;
  const hingeRestAngles = new Float32Array(hingeCount);
  const hingeRestEdgeLengths = new Float32Array(hingeCount);
  for (let hinge = 0; hinge < hingeCount; hinge += 1) {
    const base = hinge * 4;
    const rest = precomputeQuadraticDihedralRest(
      packedRestPositions(scene.mesh.positions, [
        hinges[base]!,
        hinges[base + 1]!,
        hinges[base + 2]!,
        hinges[base + 3]!,
      ]),
    );
    hingeRestAngles[hinge] = rest.restAngle;
    hingeRestEdgeLengths[hinge] = rest.restEdgeLength;
  }

  return {
    vertexCount: getVertexCount(scene.mesh),
    triangleIndices: cloth.triangles.slice(),
    triangleInverseRestBases,
    triangleRestAreas,
    hingeIndices: hinges,
    hingeRestAngles,
    hingeRestEdgeLengths,
    youngModulus: cloth.youngModulus,
    poissonRatio: cloth.poissonRatio,
    thickness: cloth.thickness,
    bendingStiffness: cloth.bendingStiffness,
  };
}

function packJGS2GpuInput(
  scene: PrecomputedScene,
  cubatureK: number,
): JGS2GpuInput {
  const vertexCount = getVertexCount(scene.mesh);
  const tetCount = getTetrahedronCount(scene.mesh);
  if (!Number.isSafeInteger(cubatureK) || cubatureK < 1) {
    throw new RangeError("The packed Cubature width must be a positive integer.");
  }
  const positions = new Float32Array(vertexCount * 4);
  const bodyVelocities = scene.settings.initialBodyVelocities;
  const velocities = bodyVelocities
    ? new Float32Array(vertexCount * 4)
    : undefined;
  const vertexRest = new Float32Array(vertexCount * 4);
  const adjacency = buildAdjacency(scene.mesh);
  const vertexInfo = new Uint32Array(vertexCount * 4);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const value = scene.mesh.positions[vertex * 3 + coordinate]!;
      positions[vertex * 4 + coordinate] = value;
      vertexRest[vertex * 4 + coordinate] = value;
    }
    positions[vertex * 4 + 3] = 1;
    if (velocities) {
      const bodyId = scene.mesh.bodyIds[vertex]!;
      const velocity = bodyVelocities?.[bodyId] ?? [0, 0, 0];
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        velocities[vertex * 4 + coordinate] = velocity[coordinate];
      }
    }
    vertexRest[vertex * 4 + 3] = scene.lumpedMasses[vertex]!;
    vertexInfo.set(
      [
        adjacency.starts[vertex]!,
        adjacency.counts[vertex]!,
        scene.mesh.fixed[vertex]!,
        scene.mesh.bodyIds[vertex]!,
      ],
      vertex * 4,
    );
  }

  const tetInverseDm = new Float32Array(tetCount * 12);
  const tetMeta = new Float32Array(tetCount * 4);
  for (let tetrahedron = 0; tetrahedron < tetCount; tetrahedron += 1) {
    const inverseOffset = tetrahedron * 9;
    const paddedOffset = tetrahedron * 12;
    for (let column = 0; column < 3; column += 1) {
      for (let row = 0; row < 3; row += 1) {
        tetInverseDm[paddedOffset + column * 4 + row] =
          scene.restTetraData.inverseRestMatrices[
            inverseOffset + row * 3 + column
          ]!;
      }
    }
    const material = scene.materials[scene.mesh.materialIds[tetrahedron]!]!;
    const materialModel = material.model ?? "corotated-linear";
    let lambda =
      (material.youngModulus * material.poissonRatio) /
      ((1 + material.poissonRatio) * (1 - 2 * material.poissonRatio));
    let mu = material.youngModulus / (2 * (1 + material.poissonRatio));
    let materialTag = JGS2_MATERIAL_COROTATED_LINEAR;
    if (materialModel === "stable-neo-hookean") {
      const parameters = computeStableNeoHookeanParameters(material);
      lambda = parameters.lambda;
      mu = parameters.mu;
      materialTag = JGS2_MATERIAL_STABLE_NEO_HOOKEAN;
    } else if (materialModel !== "corotated-linear") {
      throw new RangeError(
        `Material ${material.name} has unknown model ${String(materialModel)}.`,
      );
    }
    tetMeta.set(
      [scene.restTetraData.volumes[tetrahedron]!, lambda, mu, materialTag],
      tetrahedron * 4,
    );
  }

  const recordCount = vertexCount * cubatureK;
  const cubatureTetIds = new Uint32Array(recordCount);
  cubatureTetIds.fill(0xffff_ffff);
  const cubatureWeights = new Float32Array(recordCount);
  const cubatureBasis = new Float32Array(recordCount * 36);
  for (const precomputation of scene.vertexPrecomputations) {
    for (let sampleIndex = 0; sampleIndex < precomputation.cubature.length; sampleIndex += 1) {
      const sample = precomputation.cubature[sampleIndex]!;
      const record = precomputation.vertex * cubatureK + sampleIndex;
      cubatureTetIds[record] = sample.tetrahedron;
      cubatureWeights[record] = sample.weight;
      cubatureBasis.set(sample.basisBlocks, record * 36);
    }
  }

  const input: JGS2GpuInput = {
    vertexCount,
    tetCount,
    cubatureK,
    positions,
    ...(velocities ? { velocities } : {}),
    vertexRest,
    vertexColors: computeVertexColors(scene),
    vertexInfo,
    tetIndices: scene.mesh.tetrahedra.slice(),
    tetInverseDm,
    tetMeta,
    tetRestStiffness: Float32Array.from(
      scene.restTetraData.stiffnessMatrices,
    ),
    adjacency: adjacency.tetrahedra,
    cubatureTetIds,
    cubatureWeights,
    cubatureBasis,
    ...(scene.cloth ? { cloth: buildJGS2GpuClothInput(scene)! } : {}),
    ...(scene.id === "contact" || scene.id === "cloth"
      ? {
          contactCandidates: buildStaticIpcContactCandidates(
            vertexCount,
            scene.surface,
          ),
        }
      : {}),
  };
  validateJGS2GpuInput(input);
  return input;
}

/** Convert selected CPU Cubature to the GPU module's padded, fixed-width ABI. */
export function toJGS2GpuInput(scene: PrecomputedScene): JGS2GpuInput {
  return packJGS2GpuInput(scene, scene.settings.cubatureSamples);
}

/**
 * Add a deterministic rigid velocity field to the force-free fixture. A rigid
 * field v(x) = v_cm + omega x (x - x_cm) contains translation and rotation
 * without introducing an artificial initial strain rate.
 */
export function toForceFreeConservationGpuInput(
  scene: PrecomputedScene,
  state: Phase0RigidInitialState = PHASE0_FORCE_FREE_BASE_STATE,
): JGS2GpuInput {
  if (scene.id !== FORCE_FREE_CONSERVATION_FIXTURE_ID) {
    throw new RangeError(
      `Expected ${FORCE_FREE_CONSERVATION_FIXTURE_ID}; got ${scene.id}.`,
    );
  }

  const input = packJGS2GpuInput(scene, getTetrahedronCount(scene.mesh));
  const positions = buildPhase0ForceFreeInitialPositions(scene);
  const velocities = buildPhase0RigidVelocityField(scene, positions, state);
  const fixtureInput: JGS2GpuInput = { ...input, positions, velocities };
  validateJGS2GpuInput(fixtureInput);
  return fixtureInput;
}

export type { JGS2GpuInput, PrecomputedScene, SceneDefinition };
