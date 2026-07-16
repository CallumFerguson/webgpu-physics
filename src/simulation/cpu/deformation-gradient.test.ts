import { describe, expect, it } from "vitest";

import { computeRestTetraData } from "./fem";
import { relativeError } from "./finite-difference";
import { multiply3 } from "./math";
import { generateRegularCuboidMesh } from "./mesh";
import { polarRotation3 } from "./oracle";
import {
  buildVertexDeformationGradientStencil,
  computeVertexPolarFrames,
  evaluateVertexDeformationGradients,
} from "./deformation-gradient";

const MATERIAL = {
  name: "frame reference",
  density: 1_000,
  youngModulus: 50_000,
  poissonRatio: 0.3,
  color: [0.3, 0.8, 0.6, 1] as const,
};

function applyAffine(
  positions: Float64Array,
  matrix: Float64Array,
  translation: readonly [number, number, number],
): Float64Array {
  const result = new Float64Array(positions.length);
  for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
    const x = positions[vertex * 3]!;
    const y = positions[vertex * 3 + 1]!;
    const z = positions[vertex * 3 + 2]!;
    for (let row = 0; row < 3; row += 1) {
      result[vertex * 3 + row] =
        matrix[row * 3]! * x +
        matrix[row * 3 + 1]! * y +
        matrix[row * 3 + 2]! * z +
        translation[row]!;
    }
  }
  return result;
}

describe("affine-exact vertex deformation-gradient stencil", () => {
  const mesh = generateRegularCuboidMesh({
    cells: [2, 2, 2],
    origin: [-0.8, -0.5, -0.65],
    size: [1.6, 1.1, 1.3],
  });
  const restData = computeRestTetraData(mesh, [MATERIAL]);
  const stencil = buildVertexDeformationGradientStencil(mesh, restData);

  it("satisfies translation and affine moment identities at every vertex", () => {
    let minimumRecordCount = Number.POSITIVE_INFINITY;
    let maximumRecordCount = 0;
    for (let center = 0; center < stencil.vertexCount; center += 1) {
      const coefficientSum = new Float64Array(3);
      const restMoment = new Float64Array(9);
      const start = stencil.starts[center]!;
      const end = stencil.starts[center + 1]!;
      minimumRecordCount = Math.min(minimumRecordCount, end - start);
      maximumRecordCount = Math.max(maximumRecordCount, end - start);
      for (let record = start; record < end; record += 1) {
        const source = stencil.vertices[record]!;
        for (let materialCoordinate = 0; materialCoordinate < 3; materialCoordinate += 1) {
          const coefficient =
            stencil.coefficients[record * 3 + materialCoordinate]!;
          coefficientSum[materialCoordinate] += coefficient;
          for (let worldCoordinate = 0; worldCoordinate < 3; worldCoordinate += 1) {
            restMoment[worldCoordinate * 3 + materialCoordinate] +=
              mesh.positions[source * 3 + worldCoordinate]! * coefficient;
          }
        }
      }
      expect(
        Math.hypot(...coefficientSum),
        `vertex ${center} translation identity`,
      ).toBeLessThan(1e-12);
      expect(
        relativeError(
          restMoment,
          new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        ),
        `vertex ${center} affine identity`,
      ).toBeLessThan(1e-12);
    }
    expect(minimumRecordCount).toBeLessThan(maximumRecordCount);
  });

  it("recovers the same arbitrary affine gradient at boundary and interior vertices", () => {
    const affine = new Float64Array([
      1.12, 0.18, -0.07,
      -0.04, 0.86, 0.13,
      0.09, -0.11, 1.07,
    ]);
    const positions = applyAffine(mesh.positions, affine, [2.4, -1.2, 0.7]);
    const gradients = evaluateVertexDeformationGradients(stencil, positions);
    let worst = 0;
    for (let vertex = 0; vertex < stencil.vertexCount; vertex += 1) {
      worst = Math.max(
        worst,
        relativeError(
          gradients.subarray(vertex * 9, (vertex + 1) * 9),
          affine,
        ),
      );
    }
    expect(worst).toBeLessThan(1e-12);
  });

  it("produces objective polar frames for a translated rigid motion", () => {
    const angle = 0.71;
    const rotation = new Float64Array([
      Math.cos(angle), -Math.sin(angle), 0,
      Math.sin(angle), Math.cos(angle), 0,
      0, 0, 1,
    ]);
    const positions = applyAffine(mesh.positions, rotation, [-3.1, 0.8, 1.6]);
    const frames = computeVertexPolarFrames(stencil, positions);
    let worst = 0;
    for (let vertex = 0; vertex < stencil.vertexCount; vertex += 1) {
      worst = Math.max(
        worst,
        relativeError(
          frames.subarray(vertex * 9, (vertex + 1) * 9),
          rotation,
        ),
      );
    }
    expect(worst).toBeLessThan(1e-12);
  });

  it("uses polar of the averaged deformation gradient for a non-affine pose", () => {
    const positions = mesh.positions.slice();
    for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
      const x = mesh.positions[vertex * 3]!;
      const y = mesh.positions[vertex * 3 + 1]!;
      const z = mesh.positions[vertex * 3 + 2]!;
      positions[vertex * 3] = x + 0.25 * y * z;
      positions[vertex * 3 + 1] = y - 0.15 * x * x;
      positions[vertex * 3 + 2] = z + 0.18 * x * y;
    }
    const frames = computeVertexPolarFrames(stencil, positions);
    let worstExpectedError = 0;
    let largestShortcutDifference = 0;

    for (let center = 0; center < stencil.vertexCount; center += 1) {
      const averageGradient = new Float64Array(9);
      const averageTetRotation = new Float64Array(9);
      let totalVolume = 0;
      for (let tetrahedron = 0; tetrahedron < mesh.tetrahedra.length / 4; tetrahedron += 1) {
        const vertices = mesh.tetrahedra.subarray(
          tetrahedron * 4,
          tetrahedron * 4 + 4,
        );
        if (!vertices.includes(center)) {
          continue;
        }
        const [a, b, c, d] = vertices;
        const deformedShape = new Float64Array([
          positions[b! * 3]! - positions[a! * 3]!,
          positions[c! * 3]! - positions[a! * 3]!,
          positions[d! * 3]! - positions[a! * 3]!,
          positions[b! * 3 + 1]! - positions[a! * 3 + 1]!,
          positions[c! * 3 + 1]! - positions[a! * 3 + 1]!,
          positions[d! * 3 + 1]! - positions[a! * 3 + 1]!,
          positions[b! * 3 + 2]! - positions[a! * 3 + 2]!,
          positions[c! * 3 + 2]! - positions[a! * 3 + 2]!,
          positions[d! * 3 + 2]! - positions[a! * 3 + 2]!,
        ]);
        const deformationGradient = multiply3(
          deformedShape,
          restData.inverseRestMatrices.subarray(
            tetrahedron * 9,
            (tetrahedron + 1) * 9,
          ),
        );
        const tetRotation = polarRotation3(deformationGradient);
        const volume = restData.volumes[tetrahedron]!;
        totalVolume += volume;
        for (let index = 0; index < 9; index += 1) {
          averageGradient[index] += volume * deformationGradient[index]!;
          averageTetRotation[index] += volume * tetRotation[index]!;
        }
      }
      for (let index = 0; index < 9; index += 1) {
        averageGradient[index] /= totalVolume;
        averageTetRotation[index] /= totalVolume;
      }
      const expected = polarRotation3(averageGradient);
      const oldShortcut = polarRotation3(averageTetRotation);
      const actual = frames.subarray(center * 9, (center + 1) * 9);
      worstExpectedError = Math.max(
        worstExpectedError,
        relativeError(actual, expected),
      );
      largestShortcutDifference = Math.max(
        largestShortcutDifference,
        relativeError(expected, oldShortcut),
      );
    }

    expect(worstExpectedError).toBeLessThan(1e-12);
    expect(largestShortcutDifference).toBeGreaterThan(1e-3);
  });
});
