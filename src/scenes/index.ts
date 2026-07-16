import {
  appendTetrahedralMeshes,
  buildPrecomputedScene,
  computeStableNeoHookeanParameters,
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

export const SCENE_IDS = ["minimal", "stiffness", "drop", "stress"] as const;
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

const STIFF_MATERIAL: LinearMaterial = {
  name: "stiff rubber",
  density: 950,
  youngModulus: 280_000,
  poissonRatio: 0.32,
  color: [0.98, 0.38, 0.25, 1],
};

const DROP_MATERIAL: LinearMaterial = {
  name: "drop block",
  density: 900,
  youngModulus: 70_000,
  poissonRatio: 0.36,
  color: [0.26, 0.63, 0.96, 1],
};

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
    materials: [SOFT_MATERIAL],
    settings: {
      timestep: 1 / 30,
      gravity: [0, -9.81, 0],
      floorY: 0,
      solverIterations: 7,
      cubatureSamples: 4,
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
    cells: [2, 1, 1],
    origin: [-0.75, -0.5, -0.45],
    size: [1.5, 1, 0.9],
    materialId: 0,
    bodyId: 0,
  });
  const mesh = transformTetrahedralMesh(centered, {
    translation: [0, 2.45, 0],
    rotationEuler: [0.18, 0.12, 0.32],
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
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    for (let channel = 0; channel < 4; channel += 1) {
      colors[vertex * 4 + channel] /= contributionCounts[vertex]!;
    }
  }
  return colors;
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
