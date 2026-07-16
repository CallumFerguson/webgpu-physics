import type {
  DenseEnergyEvaluation,
  DifferentiableEnergyModel,
} from "./energy";
import { getTetrahedronCount, getVertexCount, validateTetrahedralMesh } from "./mesh";
import { fullPositionsFromActiveCoordinates } from "./oracle";
import { evaluateStableNeoHookeanMesh } from "./stable-neo-hookean";
import type {
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

const STABLE_ID = /^[a-z0-9]+(?:[._-][a-z0-9]+)*$/;

/** A diagonal quadratic potential over the oracle's active coordinates. */
export interface NonlinearQuadraticCoordinateTarget {
  readonly id: string;
  readonly target: Float64Array;
  readonly stiffness: Float64Array;
}

export interface StableNeoHookeanImplicitEulerComponents {
  readonly inertia: number;
  readonly material: number;
  readonly externalForce: number;
  readonly quadraticTargets: number;
}

export interface QuadraticTargetEnergy {
  readonly id: string;
  readonly energy: number;
}

export interface StableNeoHookeanImplicitEulerEvaluation
  extends DenseEnergyEvaluation {
  readonly components: StableNeoHookeanImplicitEulerComponents;
  readonly quadraticTargetEnergies: readonly QuadraticTargetEnergy[];
  /** Exact, unweighted material energy from every tetrahedron in mesh order. */
  readonly tetrahedronEnergies: Float64Array;
  /** det(F) for every tetrahedron in mesh order. */
  readonly deformationDeterminants: Float64Array;
  readonly minimumDeformationDeterminant: number;
}

export interface StableNeoHookeanImplicitEulerOptions {
  readonly mesh: TetrahedralMesh;
  readonly restData: RestTetraData;
  readonly materials: readonly LinearMaterial[];
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
  readonly timestep: number;
  /** Full xyz positions used as the implicit-Euler inertial target. */
  readonly predictedPositions: Float64Array;
  /** Optional constant force in active-coordinate order. */
  readonly externalForce?: Float64Array;
  /** Optional diagonal quadratic potentials in active-coordinate order. */
  readonly quadraticTargets?: readonly NonlinearQuadraticCoordinateTarget[];
}

export interface StableNeoHookeanImplicitEulerOracle
  extends DifferentiableEnergyModel<StableNeoHookeanImplicitEulerEvaluation> {}

interface FrozenQuadraticTarget {
  readonly id: string;
  readonly target: Float64Array;
  readonly stiffness: Float64Array;
}

function validateFiniteVector(
  values: Float64Array,
  expectedLength: number,
  label: string,
): void {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `${label} contains ${values.length} values; expected ${expectedLength}.`,
    );
  }
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${label} must contain only finite values.`);
    }
  }
}

function validateActiveCoordinateSystem(
  mesh: TetrahedralMesh,
  system: RestLinearSystem,
): void {
  const vertexCount = getVertexCount(mesh);
  if (system.vertexToActiveDof.length !== vertexCount) {
    throw new RangeError("The active-coordinate system does not match the mesh.");
  }
  if (
    system.dimension <= 0 ||
    system.dimension !== system.activeVertices.length * 3
  ) {
    throw new RangeError("The active-coordinate system has an invalid dimension.");
  }
  if (
    system.hessian.length !== system.dimension * system.dimension ||
    system.choleskyLower.length !== system.dimension * system.dimension
  ) {
    throw new RangeError("The active-coordinate system has invalid dense matrices.");
  }

  const activeVertexMask = new Uint8Array(vertexCount);
  const activeCoordinateMask = new Uint8Array(system.dimension);
  for (const vertex of system.activeVertices) {
    if (vertex >= vertexCount || activeVertexMask[vertex] !== 0) {
      throw new RangeError("The active-coordinate system contains an invalid vertex.");
    }
    if (mesh.fixed[vertex] !== 0) {
      throw new RangeError("A fixed mesh vertex cannot own active coordinates.");
    }
    activeVertexMask[vertex] = 1;
    const base = system.vertexToActiveDof[vertex]!;
    if (
      !Number.isSafeInteger(base) ||
      base < 0 ||
      base % 3 !== 0 ||
      base + 2 >= system.dimension
    ) {
      throw new RangeError("An active vertex has an invalid coordinate mapping.");
    }
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      if (activeCoordinateMask[base + coordinate] !== 0) {
        throw new RangeError("Active-coordinate mappings overlap.");
      }
      activeCoordinateMask[base + coordinate] = 1;
    }
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const expectedActive = mesh.fixed[vertex] === 0;
    if ((activeVertexMask[vertex] !== 0) !== expectedActive) {
      throw new RangeError("The active-coordinate system disagrees with the fixed mask.");
    }
    if (!expectedActive && system.vertexToActiveDof[vertex] !== -1) {
      throw new RangeError("A fixed mesh vertex has an active-coordinate mapping.");
    }
  }
  for (const present of activeCoordinateMask) {
    if (present === 0) {
      throw new RangeError("The active-coordinate system leaves a coordinate unmapped.");
    }
  }
}

function freezeQuadraticTargets(
  targets: readonly NonlinearQuadraticCoordinateTarget[],
  dimension: number,
): readonly FrozenQuadraticTarget[] {
  const ids = new Set<string>();
  return Object.freeze(
    targets.map((target) => {
      if (!STABLE_ID.test(target.id) || ids.has(target.id)) {
        throw new Error(
          `Quadratic-target ID ${JSON.stringify(target.id)} must be unique and stable.`,
        );
      }
      ids.add(target.id);
      validateFiniteVector(target.target, dimension, `Target ${target.id}`);
      validateFiniteVector(
        target.stiffness,
        dimension,
        `Target stiffness ${target.id}`,
      );
      for (const stiffness of target.stiffness) {
        if (stiffness < 0) {
          throw new RangeError(
            `Target stiffness ${target.id} must be nonnegative.`,
          );
        }
      }
      return Object.freeze({
        id: target.id,
        target: target.target.slice(),
        stiffness: target.stiffness.slice(),
      });
    }),
  );
}

/**
 * Exact Float64 active-coordinate oracle for a stable Neo-Hookean implicit-
 * Euler objective. Material derivatives are evaluated from every tetrahedron;
 * no Cubature, projection, or nonlinear solve regularization is applied.
 */
export function createStableNeoHookeanImplicitEulerOracle(
  options: StableNeoHookeanImplicitEulerOptions,
): StableNeoHookeanImplicitEulerOracle {
  validateTetrahedralMesh(options.mesh);
  const vertexCount = getVertexCount(options.mesh);
  const tetrahedronCount = getTetrahedronCount(options.mesh);
  if (tetrahedronCount === 0) {
    throw new RangeError("The nonlinear oracle requires at least one tetrahedron.");
  }
  if (
    options.restData.volumes.length !== tetrahedronCount ||
    options.restData.inverseRestMatrices.length !== tetrahedronCount * 9
  ) {
    throw new RangeError("The nonlinear oracle rest data does not match the mesh.");
  }
  validateActiveCoordinateSystem(options.mesh, options.restSystem);
  if (!(options.timestep > 0) || !Number.isFinite(options.timestep)) {
    throw new RangeError("The nonlinear oracle timestep must be finite and positive.");
  }
  if (options.lumpedMasses.length !== vertexCount) {
    throw new RangeError("The nonlinear oracle requires one lumped mass per vertex.");
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const mass = options.lumpedMasses[vertex]!;
    if (!Number.isFinite(mass) || mass < 0) {
      throw new RangeError(`Vertex ${vertex} has an invalid lumped mass.`);
    }
    if (options.mesh.fixed[vertex] === 0 && !(mass > 0)) {
      throw new RangeError(`Movable vertex ${vertex} must have positive mass.`);
    }
  }
  validateFiniteVector(
    options.predictedPositions,
    vertexCount * 3,
    "Predicted positions",
  );
  for (let tetrahedron = 0; tetrahedron < tetrahedronCount; tetrahedron += 1) {
    const material = options.materials[options.mesh.materialIds[tetrahedron]!];
    if (!material) {
      throw new RangeError(
        `Tetrahedron ${tetrahedron} references an unknown material.`,
      );
    }
    if (material.model !== "stable-neo-hookean") {
      throw new RangeError(
        `Stable Neo-Hookean oracle material ${material.name} must declare model stable-neo-hookean.`,
      );
    }
  }

  const dimension = options.restSystem.dimension;
  const predictedPositions = options.predictedPositions.slice();
  const externalForce = options.externalForce?.slice() ?? new Float64Array(dimension);
  validateFiniteVector(externalForce, dimension, "External force");
  const quadraticTargets = freezeQuadraticTargets(
    options.quadraticTargets ?? [],
    dimension,
  );
  const inverseTimestepSquared = 1 / (options.timestep * options.timestep);

  const evaluate = (
    coordinates: Float64Array,
  ): StableNeoHookeanImplicitEulerEvaluation => {
    validateFiniteVector(coordinates, dimension, "Active coordinates");
    const positions = fullPositionsFromActiveCoordinates(
      coordinates,
      options.mesh,
      options.restSystem,
    );
    const material = evaluateStableNeoHookeanMesh(
      options.mesh,
      options.restData,
      options.materials,
      positions,
    );
    const gradient = new Float64Array(dimension);
    const hessian = new Float64Array(dimension * dimension);
    let inertiaEnergy = 0;
    let externalForceEnergy = 0;
    let quadraticTargetsEnergy = 0;

    for (const rowVertex of options.restSystem.activeVertices) {
      const rowBase = options.restSystem.vertexToActiveDof[rowVertex]!;
      const inertia =
        options.lumpedMasses[rowVertex]! * inverseTimestepSquared;
      for (let rowCoordinate = 0; rowCoordinate < 3; rowCoordinate += 1) {
        const activeRow = rowBase + rowCoordinate;
        const fullRow = rowVertex * 3 + rowCoordinate;
        const difference = positions[fullRow]! - predictedPositions[fullRow]!;
        inertiaEnergy += 0.5 * inertia * difference * difference;
        externalForceEnergy -= externalForce[activeRow]! * coordinates[activeRow]!;
        gradient[activeRow] =
          material.gradient[fullRow]! +
          inertia * difference -
          externalForce[activeRow]!;
        hessian[activeRow * dimension + activeRow] += inertia;

        for (const columnVertex of options.restSystem.activeVertices) {
          const columnBase = options.restSystem.vertexToActiveDof[columnVertex]!;
          for (
            let columnCoordinate = 0;
            columnCoordinate < 3;
            columnCoordinate += 1
          ) {
            const activeColumn = columnBase + columnCoordinate;
            const fullColumn = columnVertex * 3 + columnCoordinate;
            hessian[activeRow * dimension + activeColumn] +=
              material.hessian[
                fullRow * (vertexCount * 3) + fullColumn
              ]!;
          }
        }
      }
    }

    const quadraticTargetEnergies: QuadraticTargetEnergy[] = [];
    for (const target of quadraticTargets) {
      let targetEnergy = 0;
      for (let coordinate = 0; coordinate < dimension; coordinate += 1) {
        const displacement =
          coordinates[coordinate]! - target.target[coordinate]!;
        const stiffness = target.stiffness[coordinate]!;
        targetEnergy += 0.5 * stiffness * displacement * displacement;
        gradient[coordinate] += stiffness * displacement;
        hessian[coordinate * dimension + coordinate] += stiffness;
      }
      quadraticTargetsEnergy += targetEnergy;
      quadraticTargetEnergies.push({ id: target.id, energy: targetEnergy });
    }

    let minimumDeformationDeterminant = Number.POSITIVE_INFINITY;
    for (const determinant of material.deformationDeterminants) {
      minimumDeformationDeterminant = Math.min(
        minimumDeformationDeterminant,
        determinant,
      );
    }
    const components: StableNeoHookeanImplicitEulerComponents = {
      inertia: inertiaEnergy,
      material: material.energy,
      externalForce: externalForceEnergy,
      quadraticTargets: quadraticTargetsEnergy,
    };
    return {
      energy:
        inertiaEnergy +
        material.energy +
        externalForceEnergy +
        quadraticTargetsEnergy,
      gradient,
      hessian,
      components,
      quadraticTargetEnergies: Object.freeze(quadraticTargetEnergies),
      tetrahedronEnergies: material.tetrahedronEnergies,
      deformationDeterminants: material.deformationDeterminants,
      minimumDeformationDeterminant,
    };
  };

  return {
    dimension,
    evaluate,
    energy: (coordinates) => evaluate(coordinates).energy,
    gradient: (coordinates) => evaluate(coordinates).gradient,
    hessian: (coordinates) => evaluate(coordinates).hessian,
  };
}
