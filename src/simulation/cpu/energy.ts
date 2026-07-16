export type EnergyTermKind =
  | "inertia"
  | "material"
  | "external-force"
  | "constraint"
  | "contact";

export interface DenseEnergyEvaluation {
  readonly energy: number;
  readonly gradient: Float64Array;
  /** Dense row-major Hessian. */
  readonly hessian: Float64Array;
}

export interface DifferentiableEnergyModel<
  Evaluation extends DenseEnergyEvaluation = DenseEnergyEvaluation,
> {
  readonly dimension: number;
  evaluate(coordinates: Float64Array): Evaluation;
  energy(coordinates: Float64Array): number;
  gradient(coordinates: Float64Array): Float64Array;
  hessian(coordinates: Float64Array): Float64Array;
}

export interface DifferentiableEnergyTerm {
  readonly id: string;
  readonly kind: EnergyTermKind;
  readonly dimension: number;
  evaluate(coordinates: Float64Array): DenseEnergyEvaluation;
}

export interface EnergyContribution {
  readonly id: string;
  readonly kind: EnergyTermKind;
  readonly energy: number;
}

export interface CompositeEnergyEvaluation extends DenseEnergyEvaluation {
  readonly contributions: readonly EnergyContribution[];
}

export interface CompositeEnergyModel
  extends DifferentiableEnergyModel<CompositeEnergyEvaluation> {
  readonly terms: readonly DifferentiableEnergyTerm[];
}

function validateDimension(dimension: number, label: string): void {
  if (!Number.isSafeInteger(dimension) || dimension <= 0) {
    throw new RangeError(label + " must be a positive safe integer.");
  }
}

function validateCoordinates(
  coordinates: Float64Array,
  dimension: number,
): void {
  if (coordinates.length !== dimension) {
    throw new RangeError(
      "Energy coordinates contain " +
        coordinates.length +
        " values; expected " +
        dimension +
        ".",
    );
  }
  for (const value of coordinates) {
    if (!Number.isFinite(value)) {
      throw new RangeError("Energy coordinates must be finite.");
    }
  }
}

function validateEvaluation(
  evaluation: DenseEnergyEvaluation,
  dimension: number,
  label: string,
): void {
  if (
    !Number.isFinite(evaluation.energy) ||
    evaluation.gradient.length !== dimension ||
    evaluation.hessian.length !== dimension * dimension
  ) {
    throw new Error(label + " returned an invalid energy evaluation shape.");
  }
  for (const value of evaluation.gradient) {
    if (!Number.isFinite(value)) {
      throw new Error(label + " returned a non-finite gradient.");
    }
  }
  for (const value of evaluation.hessian) {
    if (!Number.isFinite(value)) {
      throw new Error(label + " returned a non-finite Hessian.");
    }
  }
}

/**
 * Compose material, force, constraint, and contact energies without hiding
 * their individual contributions. This is the CPU reference contract mirrored
 * by specialized GPU kernels; runtime code does not dynamically dispatch JS
 * callbacks from the GPU.
 */
export function createCompositeEnergyModel(
  terms: readonly DifferentiableEnergyTerm[],
): CompositeEnergyModel {
  if (terms.length === 0) {
    throw new RangeError("A composite energy requires at least one term.");
  }
  const dimension = terms[0]!.dimension;
  validateDimension(dimension, "Energy dimension");
  const ids = new Set<string>();
  for (const term of terms) {
    if (!/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/.test(term.id)) {
      throw new Error(
        "Energy term ID " + JSON.stringify(term.id) + " is not stable.",
      );
    }
    if (ids.has(term.id)) {
      throw new Error("Duplicate energy term ID " + term.id + ".");
    }
    ids.add(term.id);
    if (term.dimension !== dimension) {
      throw new RangeError(
        "Energy term " +
          term.id +
          " has dimension " +
          term.dimension +
          "; expected " +
          dimension +
          ".",
      );
    }
  }
  const frozenTerms = Object.freeze([...terms]);

  const evaluate = (coordinates: Float64Array): CompositeEnergyEvaluation => {
    validateCoordinates(coordinates, dimension);
    const gradient = new Float64Array(dimension);
    const hessian = new Float64Array(dimension * dimension);
    const contributions: EnergyContribution[] = [];
    let energy = 0;
    for (const term of frozenTerms) {
      const result = term.evaluate(coordinates);
      validateEvaluation(result, dimension, "Energy term " + term.id);
      energy += result.energy;
      for (let index = 0; index < dimension; index += 1) {
        gradient[index] += result.gradient[index]!;
      }
      for (let index = 0; index < hessian.length; index += 1) {
        hessian[index] += result.hessian[index]!;
      }
      contributions.push({
        id: term.id,
        kind: term.kind,
        energy: result.energy,
      });
    }
    return { energy, gradient, hessian, contributions };
  };

  return {
    dimension,
    terms: frozenTerms,
    evaluate,
    energy: (coordinates) => evaluate(coordinates).energy,
    gradient: (coordinates) => evaluate(coordinates).gradient,
    hessian: (coordinates) => evaluate(coordinates).hessian,
  };
}

export interface LinearExternalForceOptions {
  readonly id: string;
  /** Force per active coordinate; potential is negative force dot position. */
  readonly force: Float64Array;
}

export function createLinearExternalForceTerm(
  options: LinearExternalForceOptions,
): DifferentiableEnergyTerm {
  const force = options.force.slice();
  validateDimension(force.length, "External-force dimension");
  for (const value of force) {
    if (!Number.isFinite(value)) {
      throw new RangeError("External forces must be finite.");
    }
  }
  return {
    id: options.id,
    kind: "external-force",
    dimension: force.length,
    evaluate: (coordinates) => {
      validateCoordinates(coordinates, force.length);
      let energy = 0;
      const gradient = new Float64Array(force.length);
      for (let index = 0; index < force.length; index += 1) {
        energy -= force[index]! * coordinates[index]!;
        gradient[index] = -force[index]!;
      }
      return {
        energy,
        gradient,
        hessian: new Float64Array(force.length * force.length),
      };
    },
  };
}

export interface QuadraticTargetOptions {
  readonly id: string;
  readonly kind: "constraint" | "contact";
  readonly target: Float64Array;
  readonly stiffness: Float64Array;
}

/**
 * Reference diagonal quadratic target. It supports simple constraints and
 * contact test oracles while nonlinear IPC supplies its own term later.
 */
export function createQuadraticTargetTerm(
  options: QuadraticTargetOptions,
): DifferentiableEnergyTerm {
  if (options.target.length !== options.stiffness.length) {
    throw new RangeError("Quadratic target and stiffness dimensions must match.");
  }
  const target = options.target.slice();
  const stiffness = options.stiffness.slice();
  validateDimension(target.length, "Quadratic-target dimension");
  for (let index = 0; index < target.length; index += 1) {
    if (
      !Number.isFinite(target[index]) ||
      !Number.isFinite(stiffness[index]) ||
      stiffness[index]! < 0
    ) {
      throw new RangeError(
        "Quadratic targets require finite targets and nonnegative stiffness.",
      );
    }
  }
  return {
    id: options.id,
    kind: options.kind,
    dimension: target.length,
    evaluate: (coordinates) => {
      validateCoordinates(coordinates, target.length);
      let energy = 0;
      const gradient = new Float64Array(target.length);
      const hessian = new Float64Array(target.length * target.length);
      for (let index = 0; index < target.length; index += 1) {
        const displacement = coordinates[index]! - target[index]!;
        energy += 0.5 * stiffness[index]! * displacement * displacement;
        gradient[index] = stiffness[index]! * displacement;
        hessian[index * target.length + index] = stiffness[index]!;
      }
      return { energy, gradient, hessian };
    },
  };
}
