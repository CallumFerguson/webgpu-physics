const MATRIX_EPSILON = 1e-14;

export function determinant3(matrix: ArrayLike<number>): number {
  return (
    matrix[0]! * (matrix[4]! * matrix[8]! - matrix[5]! * matrix[7]!) -
    matrix[1]! * (matrix[3]! * matrix[8]! - matrix[5]! * matrix[6]!) +
    matrix[2]! * (matrix[3]! * matrix[7]! - matrix[4]! * matrix[6]!)
  );
}

export function invert3(matrix: ArrayLike<number>): Float64Array {
  const determinant = determinant3(matrix);
  const scale = Math.max(...Array.from(matrix, (value) => Math.abs(value)));

  if (
    scale === 0 ||
    Math.abs(determinant) <= MATRIX_EPSILON * scale * scale * scale
  ) {
    throw new Error("Cannot invert a singular 3 by 3 matrix.");
  }

  const inverseDeterminant = 1 / determinant;

  return new Float64Array([
    (matrix[4]! * matrix[8]! - matrix[5]! * matrix[7]!) * inverseDeterminant,
    (matrix[2]! * matrix[7]! - matrix[1]! * matrix[8]!) * inverseDeterminant,
    (matrix[1]! * matrix[5]! - matrix[2]! * matrix[4]!) * inverseDeterminant,
    (matrix[5]! * matrix[6]! - matrix[3]! * matrix[8]!) * inverseDeterminant,
    (matrix[0]! * matrix[8]! - matrix[2]! * matrix[6]!) * inverseDeterminant,
    (matrix[2]! * matrix[3]! - matrix[0]! * matrix[5]!) * inverseDeterminant,
    (matrix[3]! * matrix[7]! - matrix[4]! * matrix[6]!) * inverseDeterminant,
    (matrix[1]! * matrix[6]! - matrix[0]! * matrix[7]!) * inverseDeterminant,
    (matrix[0]! * matrix[4]! - matrix[1]! * matrix[3]!) * inverseDeterminant,
  ]);
}

export function multiply3(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
): Float64Array {
  const result = new Float64Array(9);

  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      let value = 0;
      for (let inner = 0; inner < 3; inner += 1) {
        value += left[row * 3 + inner]! * right[inner * 3 + column]!;
      }
      result[row * 3 + column] = value;
    }
  }

  return result;
}

export function choleskyFactor(
  matrix: Float64Array,
  dimension: number,
): Float64Array {
  if (matrix.length !== dimension * dimension) {
    throw new RangeError("The matrix size does not match its dimension.");
  }

  const lower = new Float64Array(matrix.length);
  let largestDiagonal = 0;
  for (let index = 0; index < dimension; index += 1) {
    largestDiagonal = Math.max(
      largestDiagonal,
      Math.abs(matrix[index * dimension + index]!),
    );
  }
  const tolerance = Math.max(largestDiagonal, 1) * 1e-13;

  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let value = matrix[row * dimension + column]!;

      for (let inner = 0; inner < column; inner += 1) {
        value -=
          lower[row * dimension + inner]! *
          lower[column * dimension + inner]!;
      }

      if (row === column) {
        if (!(value > tolerance) || !Number.isFinite(value)) {
          throw new Error(
            `Rest Hessian is not positive definite at diagonal ${row} (${value}).`,
          );
        }
        lower[row * dimension + column] = Math.sqrt(value);
      } else {
        lower[row * dimension + column] =
          value / lower[column * dimension + column]!;
      }
    }
  }

  return lower;
}

export function solveCholesky(
  lower: Float64Array,
  rightHandSide: ArrayLike<number>,
  dimension: number,
): Float64Array {
  if (
    lower.length !== dimension * dimension ||
    rightHandSide.length !== dimension
  ) {
    throw new RangeError("The Cholesky solve dimensions do not match.");
  }

  const result = new Float64Array(dimension);

  for (let row = 0; row < dimension; row += 1) {
    let value = rightHandSide[row]!;
    for (let column = 0; column < row; column += 1) {
      value -= lower[row * dimension + column]! * result[column]!;
    }
    result[row] = value / lower[row * dimension + row]!;
  }

  for (let row = dimension - 1; row >= 0; row -= 1) {
    let value = result[row]!;
    for (let column = row + 1; column < dimension; column += 1) {
      value -= lower[column * dimension + row]! * result[column]!;
    }
    result[row] = value / lower[row * dimension + row]!;
  }

  return result;
}

export function solveDenseLinearSystem(
  matrix: Float64Array,
  rightHandSide: Float64Array,
  dimension: number,
): Float64Array | undefined {
  if (
    matrix.length !== dimension * dimension ||
    rightHandSide.length !== dimension
  ) {
    throw new RangeError("The dense solve dimensions do not match.");
  }

  const coefficients = matrix.slice();
  const result = rightHandSide.slice();

  for (let pivot = 0; pivot < dimension; pivot += 1) {
    let bestRow = pivot;
    let bestMagnitude = Math.abs(coefficients[pivot * dimension + pivot]!);

    for (let row = pivot + 1; row < dimension; row += 1) {
      const magnitude = Math.abs(coefficients[row * dimension + pivot]!);
      if (magnitude > bestMagnitude) {
        bestMagnitude = magnitude;
        bestRow = row;
      }
    }

    if (bestMagnitude <= MATRIX_EPSILON || !Number.isFinite(bestMagnitude)) {
      return undefined;
    }

    if (bestRow !== pivot) {
      for (let column = pivot; column < dimension; column += 1) {
        const first = pivot * dimension + column;
        const second = bestRow * dimension + column;
        const temporary = coefficients[first]!;
        coefficients[first] = coefficients[second]!;
        coefficients[second] = temporary;
      }
      const temporary = result[pivot]!;
      result[pivot] = result[bestRow]!;
      result[bestRow] = temporary;
    }

    const diagonal = coefficients[pivot * dimension + pivot]!;
    for (let row = pivot + 1; row < dimension; row += 1) {
      const multiplier = coefficients[row * dimension + pivot]! / diagonal;
      coefficients[row * dimension + pivot] = 0;
      for (let column = pivot + 1; column < dimension; column += 1) {
        coefficients[row * dimension + column] -=
          multiplier * coefficients[pivot * dimension + column]!;
      }
      result[row] -= multiplier * result[pivot]!;
    }
  }

  for (let row = dimension - 1; row >= 0; row -= 1) {
    let value = result[row]!;
    for (let column = row + 1; column < dimension; column += 1) {
      value -= coefficients[row * dimension + column]! * result[column]!;
    }
    result[row] = value / coefficients[row * dimension + row]!;
  }

  return result;
}

export function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) {
    throw new RangeError("Dot-product vectors must have equal lengths.");
  }

  let result = 0;
  for (let index = 0; index < left.length; index += 1) {
    result += left[index]! * right[index]!;
  }
  return result;
}

export function squaredNorm(vector: ArrayLike<number>): number {
  return dot(vector, vector);
}
