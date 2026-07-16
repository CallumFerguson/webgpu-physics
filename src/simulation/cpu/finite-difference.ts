/** A scalar function over a dense coordinate vector. */
export type ScalarFunction = (coordinates: Float64Array) => number;

/** A dense vector-valued function over a dense coordinate vector. */
export type VectorFunction = (coordinates: Float64Array) => Float64Array;

const DEFAULT_GRADIENT_STEP = 1e-6;
const DEFAULT_HESSIAN_STEP = 1e-5;

function validateRelativeStep(relativeStep: number): void {
  if (!(relativeStep > 0) || !Number.isFinite(relativeStep)) {
    throw new RangeError("A finite, positive finite-difference step is required.");
  }
}

function coordinateStep(
  coordinates: Float64Array,
  coordinate: number,
  relativeStep: number,
): number {
  return relativeStep * Math.max(1, Math.abs(coordinates[coordinate]!));
}

/**
 * Evaluate a scalar gradient with a centered, two-sided finite difference.
 * The step for coordinate i is relativeStep * max(1, abs(x_i)).
 */
export function centralDifferenceGradient(
  energy: ScalarFunction,
  coordinates: Float64Array,
  relativeStep = DEFAULT_GRADIENT_STEP,
): Float64Array {
  validateRelativeStep(relativeStep);
  const result = new Float64Array(coordinates.length);
  const positive = coordinates.slice();
  const negative = coordinates.slice();

  for (let coordinate = 0; coordinate < coordinates.length; coordinate += 1) {
    const step = coordinateStep(coordinates, coordinate, relativeStep);
    positive[coordinate] = coordinates[coordinate]! + step;
    negative[coordinate] = coordinates[coordinate]! - step;
    result[coordinate] = (energy(positive) - energy(negative)) / (2 * step);
    positive[coordinate] = coordinates[coordinate]!;
    negative[coordinate] = coordinates[coordinate]!;
  }

  return result;
}

/**
 * Evaluate the Jacobian of a gradient with centered differences. The result is
 * a row-major Hessian: H[row, column] = d gradient[row] / d x[column].
 */
export function centralDifferenceHessian(
  gradient: VectorFunction,
  coordinates: Float64Array,
  relativeStep = DEFAULT_HESSIAN_STEP,
): Float64Array {
  validateRelativeStep(relativeStep);
  const dimension = coordinates.length;
  const result = new Float64Array(dimension * dimension);
  const positive = coordinates.slice();
  const negative = coordinates.slice();

  for (let column = 0; column < dimension; column += 1) {
    const step = coordinateStep(coordinates, column, relativeStep);
    positive[column] = coordinates[column]! + step;
    negative[column] = coordinates[column]! - step;
    const positiveGradient = gradient(positive);
    const negativeGradient = gradient(negative);
    if (
      positiveGradient.length !== dimension ||
      negativeGradient.length !== dimension
    ) {
      throw new RangeError(
        "The finite-difference gradient dimension must match its coordinates.",
      );
    }
    for (let row = 0; row < dimension; row += 1) {
      result[row * dimension + column] =
        (positiveGradient[row]! - negativeGradient[row]!) / (2 * step);
    }
    positive[column] = coordinates[column]!;
    negative[column] = coordinates[column]!;
  }

  return result;
}

/**
 * Roadmap relative-error definition:
 * ||actual-reference|| / max(1, ||actual||, ||reference||).
 */
export function relativeError(
  actual: ArrayLike<number>,
  reference: ArrayLike<number>,
): number {
  if (actual.length !== reference.length) {
    throw new RangeError("Relative-error vectors must have equal lengths.");
  }

  let squaredDifference = 0;
  let squaredActual = 0;
  let squaredReference = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const actualValue = actual[index]!;
    const referenceValue = reference[index]!;
    const difference = actualValue - referenceValue;
    squaredDifference += difference * difference;
    squaredActual += actualValue * actualValue;
    squaredReference += referenceValue * referenceValue;
  }

  return (
    Math.sqrt(squaredDifference) /
    Math.max(1, Math.sqrt(squaredActual), Math.sqrt(squaredReference))
  );
}
