import type {
  DenseEnergyEvaluation,
  DifferentiableEnergyModel,
} from "./energy";
import { solveDenseLinearSystem } from "./math";
import type {
  StableNeoHookeanImplicitEulerEvaluation,
  StableNeoHookeanImplicitEulerOracle,
} from "./nonlinear-oracle";
import type { RestLinearSystem } from "./types";

const LOCAL_DIMENSION = 3;
const IDENTITY_TOLERANCE = 1e-8;

/**
 * The exact restriction of a dense active-coordinate term through one JGS2
 * equilibrium basis. The source block is kept separate because runtime
 * cubature gathers it exactly and approximates only the remainder.
 */
export interface JGS2LocalProjection {
  /** B^T g. */
  readonly gradient: Float64Array;
  /** B^T H B, row-major 3 by 3. */
  readonly hessian: Float64Array;
  /** The unprojected three-coordinate block at the source vertex. */
  readonly sourceGradient: Float64Array;
  /** The unprojected source-source Hessian block. */
  readonly sourceHessian: Float64Array;
  /** B^T g - g_source. */
  readonly remainderGradient: Float64Array;
  /** B^T H B - H_source,source. */
  readonly remainderHessian: Float64Array;
}

export interface ExactStableNeoHookeanJGS2LocalEvaluation
  extends DenseEnergyEvaluation,
    JGS2LocalProjection {
  /** Local coordinates relative to the model's current pose. */
  readonly localDisplacement: Float64Array;
  /** x + B localDisplacement in active-coordinate order. */
  readonly activeCoordinates: Float64Array;
  readonly oracleEvaluation: StableNeoHookeanImplicitEulerEvaluation;
  /** Solution of H_local update = -g_local, or undefined if singular. */
  readonly newtonUpdate: Float64Array | undefined;
  /** ||H_local update + g_local||_2, or undefined if singular. */
  readonly newtonResidualNorm: number | undefined;
}

export interface ExactStableNeoHookeanJGS2LocalModel
  extends DifferentiableEnergyModel<ExactStableNeoHookeanJGS2LocalEvaluation> {
  readonly dimension: 3;
  readonly vertex: number;
  readonly localBase: number;
  /** Frozen active-coordinate equilibrium basis, with three columns. */
  readonly equilibriumBasis: Float64Array;
  /** Active-coordinate pose about which local displacements are expressed. */
  readonly baseCoordinates: Float64Array;
}

export interface ExactStableNeoHookeanJGS2LocalOptions {
  readonly oracle: StableNeoHookeanImplicitEulerOracle;
  readonly restSystem: RestLinearSystem;
  readonly vertex: number;
  readonly coordinates: Float64Array;
  /** Dense active-coordinate equilibrium basis, row-major dimension by 3. */
  readonly equilibriumBasis: Float64Array;
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

function validateSourceBlock(
  equilibriumBasis: Float64Array,
  localBase: number,
): void {
  for (let row = 0; row < LOCAL_DIMENSION; row += 1) {
    for (let column = 0; column < LOCAL_DIMENSION; column += 1) {
      const actual =
        equilibriumBasis[(localBase + row) * LOCAL_DIMENSION + column]!;
      const expected = row === column ? 1 : 0;
      if (Math.abs(actual - expected) > IDENTITY_TOLERANCE) {
        throw new Error(
          "A JGS2 equilibrium basis must have an identity source block.",
        );
      }
    }
  }
}

/**
 * Project a dense objective (or one embedded element/force term) through B.
 *
 * For a term incident on the source vertex, `remainder*` is exactly the
 * source-subtracted candidate used by JGS2 cubature. For a non-incident term
 * its source block is zero, so the remainder is the full projection.
 */
export function projectDenseJGS2LocalTerm(
  term: Pick<DenseEnergyEvaluation, "gradient" | "hessian">,
  equilibriumBasis: Float64Array,
  localBase: number,
): JGS2LocalProjection {
  const dimension = term.gradient.length;
  if (!Number.isSafeInteger(localBase) || localBase < 0 || localBase + 2 >= dimension) {
    throw new RangeError("The JGS2 source block is outside the active coordinates.");
  }
  validateFiniteVector(term.gradient, dimension, "Dense gradient");
  validateFiniteVector(term.hessian, dimension * dimension, "Dense Hessian");
  validateFiniteVector(
    equilibriumBasis,
    dimension * LOCAL_DIMENSION,
    "JGS2 equilibrium basis",
  );
  validateSourceBlock(equilibriumBasis, localBase);

  const gradient = new Float64Array(LOCAL_DIMENSION);
  const hessianTimesBasis = new Float64Array(dimension * LOCAL_DIMENSION);
  const hessian = new Float64Array(LOCAL_DIMENSION * LOCAL_DIMENSION);

  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < LOCAL_DIMENSION; column += 1) {
      gradient[column] +=
        equilibriumBasis[row * LOCAL_DIMENSION + column]! * term.gradient[row]!;

      let value = 0;
      for (let inner = 0; inner < dimension; inner += 1) {
        value +=
          term.hessian[row * dimension + inner]! *
          equilibriumBasis[inner * LOCAL_DIMENSION + column]!;
      }
      hessianTimesBasis[row * LOCAL_DIMENSION + column] = value;
    }
  }

  for (let row = 0; row < LOCAL_DIMENSION; row += 1) {
    for (let column = 0; column < LOCAL_DIMENSION; column += 1) {
      let value = 0;
      for (let inner = 0; inner < dimension; inner += 1) {
        value +=
          equilibriumBasis[inner * LOCAL_DIMENSION + row]! *
          hessianTimesBasis[inner * LOCAL_DIMENSION + column]!;
      }
      hessian[row * LOCAL_DIMENSION + column] = value;
    }
  }

  const sourceGradient = term.gradient.slice(localBase, localBase + LOCAL_DIMENSION);
  const sourceHessian = new Float64Array(LOCAL_DIMENSION * LOCAL_DIMENSION);
  const remainderGradient = new Float64Array(LOCAL_DIMENSION);
  const remainderHessian = new Float64Array(LOCAL_DIMENSION * LOCAL_DIMENSION);
  for (let row = 0; row < LOCAL_DIMENSION; row += 1) {
    remainderGradient[row] = gradient[row]! - sourceGradient[row]!;
    for (let column = 0; column < LOCAL_DIMENSION; column += 1) {
      const localIndex = row * LOCAL_DIMENSION + column;
      sourceHessian[localIndex] =
        term.hessian[
          (localBase + row) * dimension + localBase + column
        ]!;
      remainderHessian[localIndex] =
        hessian[localIndex]! - sourceHessian[localIndex]!;
    }
  }

  return {
    gradient,
    hessian,
    sourceGradient,
    sourceHessian,
    remainderGradient,
    remainderHessian,
  };
}

function solveNewtonUpdate(
  gradient: Float64Array,
  hessian: Float64Array,
): { update: Float64Array | undefined; residualNorm: number | undefined } {
  const rightHandSide = Float64Array.from(gradient, (value) => -value);
  const update = solveDenseLinearSystem(
    hessian,
    rightHandSide,
    LOCAL_DIMENSION,
  );
  if (!update) {
    return { update: undefined, residualNorm: undefined };
  }

  let squaredResidual = 0;
  for (let row = 0; row < LOCAL_DIMENSION; row += 1) {
    let residual = gradient[row]!;
    for (let column = 0; column < LOCAL_DIMENSION; column += 1) {
      residual += hessian[row * LOCAL_DIMENSION + column]! * update[column]!;
    }
    squaredResidual += residual * residual;
  }
  return { update, residualNorm: Math.sqrt(squaredResidual) };
}

/**
 * Build the bounded Float64 oracle for the exact current-pose JGS2
 * restriction E_local(q) = E(x + Bq). B is frozen for this local solve.
 */
export function createExactStableNeoHookeanJGS2LocalModel(
  options: ExactStableNeoHookeanJGS2LocalOptions,
): ExactStableNeoHookeanJGS2LocalModel {
  const { oracle, restSystem, vertex } = options;
  if (oracle.dimension !== restSystem.dimension) {
    throw new RangeError("The nonlinear oracle and rest system dimensions differ.");
  }
  if (
    !Number.isSafeInteger(vertex) ||
    vertex < 0 ||
    vertex >= restSystem.vertexToActiveDof.length
  ) {
    throw new RangeError("The JGS2 source vertex is outside the rest system.");
  }
  const localBase = restSystem.vertexToActiveDof[vertex]!;
  if (localBase < 0) {
    throw new Error(`Cannot build a JGS2 local model for fixed vertex ${vertex}.`);
  }
  validateFiniteVector(
    options.coordinates,
    oracle.dimension,
    "Current active coordinates",
  );
  validateFiniteVector(
    options.equilibriumBasis,
    oracle.dimension * LOCAL_DIMENSION,
    "JGS2 equilibrium basis",
  );
  validateSourceBlock(options.equilibriumBasis, localBase);

  const baseCoordinates = options.coordinates.slice();
  const equilibriumBasis = options.equilibriumBasis.slice();

  const evaluate = (
    localDisplacement: Float64Array,
  ): ExactStableNeoHookeanJGS2LocalEvaluation => {
    validateFiniteVector(
      localDisplacement,
      LOCAL_DIMENSION,
      "JGS2 local displacement",
    );
    const activeCoordinates = baseCoordinates.slice();
    for (let row = 0; row < oracle.dimension; row += 1) {
      for (let column = 0; column < LOCAL_DIMENSION; column += 1) {
        activeCoordinates[row] +=
          equilibriumBasis[row * LOCAL_DIMENSION + column]! *
          localDisplacement[column]!;
      }
    }

    const oracleEvaluation = oracle.evaluate(activeCoordinates);
    const projection = projectDenseJGS2LocalTerm(
      oracleEvaluation,
      equilibriumBasis,
      localBase,
    );
    const newton = solveNewtonUpdate(projection.gradient, projection.hessian);
    return {
      energy: oracleEvaluation.energy,
      ...projection,
      localDisplacement: localDisplacement.slice(),
      activeCoordinates,
      oracleEvaluation,
      newtonUpdate: newton.update,
      newtonResidualNorm: newton.residualNorm,
    };
  };

  return Object.freeze({
    dimension: LOCAL_DIMENSION,
    vertex,
    localBase,
    equilibriumBasis,
    baseCoordinates,
    evaluate,
    energy: (localDisplacement: Float64Array) =>
      evaluate(localDisplacement).energy,
    gradient: (localDisplacement: Float64Array) =>
      evaluate(localDisplacement).gradient,
    hessian: (localDisplacement: Float64Array) =>
      evaluate(localDisplacement).hessian,
  });
}

/** Evaluate the current pose (q = 0) without retaining a local model. */
export function evaluateExactStableNeoHookeanJGS2Local(
  options: ExactStableNeoHookeanJGS2LocalOptions,
): ExactStableNeoHookeanJGS2LocalEvaluation {
  return createExactStableNeoHookeanJGS2LocalModel(options).evaluate(
    new Float64Array(LOCAL_DIMENSION),
  );
}
