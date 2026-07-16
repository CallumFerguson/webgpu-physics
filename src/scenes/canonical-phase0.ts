import {
  assembleRestLinearSystem,
  computeExactVertexBasis,
  computeLumpedMasses,
  computeRestTetraData,
  extractBoundarySurface,
  generateRegularCuboidMesh,
  getTetrahedronCount,
  PHASE0_REFERENCE_MATERIAL,
  transformTetrahedralMesh,
  type CubatureSample,
  type PrecomputedScene,
  type SceneDefinition,
  type Vec3,
  type VertexPrecomputation,
} from "../simulation/cpu";

export { PHASE0_REFERENCE_MATERIAL } from "../simulation/cpu";

/** Frozen IDs from manifests/canonical-scenes.v1.json. */
export const PHASE0_FORCE_FREE_FIXTURE_ID =
  "phase0.force-free-cuboid" as const;
export const PHASE0_FORCE_FREE_CORPUS_ID =
  "phase0.force-free-initial-states" as const;

export const PHASE0_FORCE_FREE_CORPUS_SEED = 104_729;
export const PHASE0_FORCE_FREE_CORPUS_GENERATOR_VERSION = "1" as const;
export const PHASE0_FORCE_FREE_CORPUS_CASE_COUNT = 32;
export const PHASE0_FORCE_FREE_FRAME_COUNT = 1_200;
export const PHASE0_FORCE_FREE_TIMESTEP = 1 / 120;
export const PHASE0_FORCE_FREE_ITERATIONS = 12;
export const PHASE0_FORCE_FREE_INITIAL_EULER: Vec3 = [0.17, -0.23, 0.31];

export interface Phase0RigidInitialState {
  readonly id: string;
  readonly linearVelocity: Vec3;
  readonly angularVelocity: Vec3;
}

export const PHASE0_FORCE_FREE_BASE_STATE: Phase0RigidInitialState = {
  id: `${PHASE0_FORCE_FREE_FIXTURE_ID}/base`,
  linearVelocity: [0.37, -0.11, 0.23],
  angularVelocity: [0.41, 0.29, -0.19],
};

/**
 * Version-one PRNG for the frozen corpora. Mulberry32 is deliberately encoded
 * here rather than delegated to Math.random; snapshot tests freeze the vectors
 * derived from this deterministic scalar stream.
 */
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

function randomUnitVector(random: () => number): Vec3 {
  for (;;) {
    const candidate: Vec3 = [
      random() * 2 - 1,
      random() * 2 - 1,
      random() * 2 - 1,
    ];
    const length = Math.hypot(...candidate);
    if (length > 1e-8 && length <= Math.sqrt(3)) {
      return [
        candidate[0] / length,
        candidate[1] / length,
        candidate[2] / length,
      ];
    }
  }
}

function randomVelocity(
  random: () => number,
  minimumSpeed: number,
  maximumSpeed: number,
): Vec3 {
  const direction = randomUnitVector(random);
  const speed = minimumSpeed + (maximumSpeed - minimumSpeed) * random();
  return [direction[0] * speed, direction[1] * speed, direction[2] * speed];
}

/** Frozen v1 generator for all 32 force-free rigid-velocity cases. */
export function generatePhase0ForceFreeInitialStateCorpus(): readonly Phase0RigidInitialState[] {
  const random = createCorpusRandom(PHASE0_FORCE_FREE_CORPUS_SEED);
  return Array.from(
    { length: PHASE0_FORCE_FREE_CORPUS_CASE_COUNT },
    (_unused, index): Phase0RigidInitialState => ({
      id: `${PHASE0_FORCE_FREE_CORPUS_ID}/${index.toString().padStart(2, "0")}`,
      linearVelocity: randomVelocity(random, 0.1, 1),
      angularVelocity: randomVelocity(random, 0.1, 1),
    }),
  );
}

export function buildPhase0ForceFreeDefinition(): SceneDefinition {
  const mesh = generateRegularCuboidMesh({
    cells: [2, 2, 2],
    origin: [-0.5, -0.5, -0.5],
    size: [1, 1, 1],
    materialId: 0,
    bodyId: 0,
  });

  return {
    id: PHASE0_FORCE_FREE_FIXTURE_ID,
    title: "Canonical force-free conservation fixture",
    description:
      "The frozen Phase 0 parity fixture: an unconstrained 2x2x2 cuboid with exact all-element Cubature, rigid translation, and rigid rotation.",
    mesh,
    materials: [PHASE0_REFERENCE_MATERIAL],
    settings: {
      timestep: PHASE0_FORCE_FREE_TIMESTEP,
      gravity: [0, 0, 0],
      // The runtime assigns zero floor stiffness. SceneDefinition predates
      // nullable floors, so zero is only a harmless renderer reference plane.
      floorY: 0,
      solverIterations: PHASE0_FORCE_FREE_ITERATIONS,
      // Exact packing replaces this legacy selected-Cubature width below.
      cubatureSamples: 6,
    },
    camera: {
      eye: [3.2, 2.4, 4.5],
      target: [0, 0, 0],
      up: [0, 1, 0],
      fovYRadians: Math.PI / 4,
    },
    landmark: {
      vertex: 0,
      label: "canonical free cuboid corner",
      expectedMotion: [1, -1, 1],
    },
  };
}

function exactAllElementSamples(
  scene: Pick<PrecomputedScene, "mesh">,
  exactBasis: Float64Array,
): readonly CubatureSample[] {
  return Array.from(
    { length: getTetrahedronCount(scene.mesh) },
    (_unused, tetrahedron): CubatureSample => {
      const basisBlocks = new Float64Array(36);
      for (let localVertex = 0; localVertex < 4; localVertex += 1) {
        const vertex = scene.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
        basisBlocks.set(
          exactBasis.subarray(vertex * 9, vertex * 9 + 9),
          localVertex * 9,
        );
      }
      return { tetrahedron, weight: 1, basisBlocks };
    },
  );
}

/**
 * Build the fixture without running the selected-Cubature trainer. Every
 * active source vertex retains its exact equilibrium basis and packs every
 * tetrahedron exactly once with weight one.
 */
export function buildPhase0ForceFreeScene(): PrecomputedScene {
  const definition = buildPhase0ForceFreeDefinition();
  const surface = extractBoundarySurface(definition.mesh);
  const restTetraData = computeRestTetraData(
    definition.mesh,
    definition.materials,
  );
  const lumpedMasses = computeLumpedMasses(
    definition.mesh,
    definition.materials,
    restTetraData,
  );
  const restSystem = assembleRestLinearSystem(
    definition.mesh,
    restTetraData,
    lumpedMasses,
    definition.settings.timestep,
  );
  const sceneWithoutPrecomputation = {
    ...definition,
    surface,
    restTetraData,
    lumpedMasses,
    restSystem,
  };
  const vertexPrecomputations: VertexPrecomputation[] = [];
  for (const vertex of restSystem.activeVertices) {
    const exact = computeExactVertexBasis(definition.mesh, restSystem, vertex);
    vertexPrecomputations.push({
      vertex,
      exactBasis: exact.basis,
      schurInverse: exact.schurInverse,
      cubature: exactAllElementSamples(sceneWithoutPrecomputation, exact.basis),
      trainingResidual: 0,
    });
  }

  return { ...sceneWithoutPrecomputation, vertexPrecomputations };
}

/** Padded initial current pose, rotated exactly as the frozen manifest states. */
export function buildPhase0ForceFreeInitialPositions(
  scene: PrecomputedScene,
): Float32Array {
  if (scene.id !== PHASE0_FORCE_FREE_FIXTURE_ID) {
    throw new RangeError(
      `Expected ${PHASE0_FORCE_FREE_FIXTURE_ID}; got ${scene.id}.`,
    );
  }
  const rotated = transformTetrahedralMesh(scene.mesh, {
    rotationEuler: PHASE0_FORCE_FREE_INITIAL_EULER,
  });
  const positions = new Float32Array((rotated.positions.length / 3) * 4);
  for (let vertex = 0; vertex < rotated.positions.length / 3; vertex += 1) {
    positions.set(
      rotated.positions.subarray(vertex * 3, vertex * 3 + 3),
      vertex * 4,
    );
    positions[vertex * 4 + 3] = 1;
  }
  return positions;
}

/** Build v(x) = v_cm + omega cross (x - x_cm) at the rotated current pose. */
export function buildPhase0RigidVelocityField(
  scene: PrecomputedScene,
  positions: Float32Array,
  state: Phase0RigidInitialState,
): Float32Array {
  const vertexCount = scene.mesh.positions.length / 3;
  if (positions.length !== vertexCount * 4) {
    throw new RangeError("Canonical initial positions must contain padded vec4s.");
  }
  const centerOfMass = [0, 0, 0];
  let totalMass = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const mass = scene.lumpedMasses[vertex]!;
    totalMass += mass;
    centerOfMass[0] += mass * positions[vertex * 4]!;
    centerOfMass[1] += mass * positions[vertex * 4 + 1]!;
    centerOfMass[2] += mass * positions[vertex * 4 + 2]!;
  }
  if (!(totalMass > 0)) {
    throw new RangeError("The canonical fixture must have positive mass.");
  }
  centerOfMass[0] /= totalMass;
  centerOfMass[1] /= totalMass;
  centerOfMass[2] /= totalMass;

  const [vx, vy, vz] = state.linearVelocity;
  const [wx, wy, wz] = state.angularVelocity;
  const velocities = new Float32Array(vertexCount * 4);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const rx = positions[vertex * 4]! - centerOfMass[0];
    const ry = positions[vertex * 4 + 1]! - centerOfMass[1];
    const rz = positions[vertex * 4 + 2]! - centerOfMass[2];
    velocities[vertex * 4] = vx + wy * rz - wz * ry;
    velocities[vertex * 4 + 1] = vy + wz * rx - wx * rz;
    velocities[vertex * 4 + 2] = vz + wx * ry - wy * rx;
  }
  return velocities;
}
