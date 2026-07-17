import { getVertexCount } from "./mesh";
import type { TetrahedralMesh } from "./types";

/**
 * Frame-frozen, world-space objective data for the active nonlinear solve.
 * A zero target stiffness is the canonical inactive/released target state.
 */
export interface NonlinearPerVertexObjectiveInput {
  /** Constant force during one frame, tightly packed xyz per mesh vertex. */
  readonly externalForces: Float64Array;
  /** World-space target position, tightly packed xyz per mesh vertex. */
  readonly targetPositions: Float64Array;
  /** Isotropic quadratic-target stiffness per mesh vertex. */
  readonly targetStiffnesses: Float64Array;
}

/** Validated snapshot; typed arrays never alias the caller's input. */
export interface NonlinearPerVertexObjective
  extends NonlinearPerVertexObjectiveInput {
  readonly vertexCount: number;
  readonly active: boolean;
}

export interface NonlinearPerVertexObjectiveEvaluation {
  readonly energy: number;
  readonly externalForceEnergy: number;
  readonly targetEnergy: number;
  readonly gradient: Float64Array;
  readonly externalForceGradient: Float64Array;
  readonly targetGradient: Float64Array;
  /** Isotropic Hessian diagonal; the full Hessian is stiffness times I. */
  readonly targetStiffness: number;
}

function requireFiniteArray(
  values: Float64Array,
  expectedLength: number,
  label: string,
): void {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `${label} contains ${values.length} values; expected ${expectedLength}.`,
    );
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${label}[${index}] must be finite.`);
    }
  }
}

/** Validate and detach one frame's force/target objective from caller state. */
export function createNonlinearPerVertexObjective(
  mesh: TetrahedralMesh,
  input: NonlinearPerVertexObjectiveInput,
): NonlinearPerVertexObjective {
  const vertexCount = getVertexCount(mesh);
  requireFiniteArray(
    input.externalForces,
    vertexCount * 3,
    "Per-vertex external forces",
  );
  requireFiniteArray(
    input.targetPositions,
    vertexCount * 3,
    "Per-vertex target positions",
  );
  requireFiniteArray(
    input.targetStiffnesses,
    vertexCount,
    "Per-vertex target stiffnesses",
  );

  let active = false;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const stiffness = input.targetStiffnesses[vertex]!;
    if (stiffness < 0) {
      throw new RangeError(
        `Per-vertex target stiffness ${vertex} must be nonnegative.`,
      );
    }
    const forceBase = vertex * 3;
    const hasForce =
      input.externalForces[forceBase] !== 0 ||
      input.externalForces[forceBase + 1] !== 0 ||
      input.externalForces[forceBase + 2] !== 0;
    if (mesh.fixed[vertex] !== 0 && hasForce) {
      throw new RangeError(
        `Fixed vertex ${vertex} cannot receive an external force.`,
      );
    }
    if (mesh.fixed[vertex] !== 0 && stiffness !== 0) {
      throw new RangeError(
        `Fixed vertex ${vertex} cannot receive a quadratic target.`,
      );
    }
    active ||= hasForce || stiffness !== 0;
  }

  return Object.freeze({
    vertexCount,
    active,
    externalForces: input.externalForces.slice(),
    targetPositions: input.targetPositions.slice(),
    targetStiffnesses: input.targetStiffnesses.slice(),
  });
}

function requireVertexPosition(
  objective: NonlinearPerVertexObjective,
  vertex: number,
  position: ArrayLike<number>,
): void {
  if (
    !Number.isSafeInteger(vertex) ||
    vertex < 0 ||
    vertex >= objective.vertexCount
  ) {
    throw new RangeError("Per-vertex objective vertex is out of range.");
  }
  if (position.length !== 3) {
    throw new RangeError("Per-vertex objective position must contain xyz.");
  }
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    if (!Number.isFinite(position[coordinate])) {
      throw new RangeError("Per-vertex objective position must be finite.");
    }
  }
}

/** Evaluate the separable linear-force and isotropic target potentials. */
export function evaluateNonlinearPerVertexObjective(
  objective: NonlinearPerVertexObjective,
  vertex: number,
  position: ArrayLike<number>,
): NonlinearPerVertexObjectiveEvaluation {
  requireVertexPosition(objective, vertex, position);
  const base = vertex * 3;
  const stiffness = objective.targetStiffnesses[vertex]!;
  const externalForceGradient = new Float64Array(3);
  const targetGradient = new Float64Array(3);
  const gradient = new Float64Array(3);
  let externalForceEnergy = 0;
  let targetSquaredDistance = 0;
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    const force = objective.externalForces[base + coordinate]!;
    externalForceEnergy -= force * position[coordinate]!;
    externalForceGradient[coordinate] = -force;
    if (stiffness !== 0) {
      const displacement =
        position[coordinate]! - objective.targetPositions[base + coordinate]!;
      targetSquaredDistance += displacement * displacement;
      targetGradient[coordinate] = stiffness * displacement;
    }
    gradient[coordinate] =
      externalForceGradient[coordinate]! + targetGradient[coordinate]!;
  }
  const targetEnergy = 0.5 * stiffness * targetSquaredDistance;
  return {
    energy: externalForceEnergy + targetEnergy,
    externalForceEnergy,
    targetEnergy,
    gradient,
    externalForceGradient,
    targetGradient,
    targetStiffness: stiffness,
  };
}

/**
 * Stable objective difference for a displacement from a known base position.
 * This avoids subtracting two origin-dependent linear-force energies.
 */
export function nonlinearPerVertexObjectiveEnergyDelta(
  objective: NonlinearPerVertexObjective,
  vertex: number,
  basePosition: ArrayLike<number>,
  displacement: ArrayLike<number>,
): number {
  requireVertexPosition(objective, vertex, basePosition);
  if (displacement.length !== 3) {
    throw new RangeError("Per-vertex objective displacement must contain xyz.");
  }
  const base = vertex * 3;
  const stiffness = objective.targetStiffnesses[vertex]!;
  let energyDelta = 0;
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    const step = displacement[coordinate]!;
    if (!Number.isFinite(step)) {
      throw new RangeError("Per-vertex objective displacement must be finite.");
    }
    const force = objective.externalForces[base + coordinate]!;
    energyDelta -= force * step;
    if (stiffness !== 0) {
      const residual =
        basePosition[coordinate]! -
        objective.targetPositions[base + coordinate]!;
      energyDelta +=
        stiffness * residual * step +
        0.5 * stiffness * step * step;
    }
  }
  return energyDelta;
}
