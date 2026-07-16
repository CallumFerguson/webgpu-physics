import { beforeAll, describe, expect, it } from "vitest";

import { buildSceneDefinition, SCENE_IDS } from "../../scenes";
import { invert3, solveCholesky } from "./math";
import { buildLowFrequencyTrainingPoses } from "./cubature";
import {
  computeLumpedMasses,
  computeRestTetraData,
} from "./fem";
import {
  appendTetrahedralMeshes,
  extractBoundarySurface,
  generateRegularCuboidMesh,
  getTetrahedronCount,
  getVertexCount,
  transformTetrahedralMesh,
} from "./mesh";
import {
  buildPrecomputedScene,
  exactBasisToActiveMatrix,
} from "./precompute";
import type { PrecomputedScene, VertexPrecomputation } from "./types";

interface WeightedCandidate {
  readonly tetrahedron: number;
  readonly weight: number;
  readonly basisBlocks: Float64Array;
}

function multiplyMatrixVector(
  matrix: Float64Array,
  vector: Float64Array,
  dimension: number,
): Float64Array {
  const result = new Float64Array(dimension);
  for (let row = 0; row < dimension; row += 1) {
    for (let column = 0; column < dimension; column += 1) {
      result[row] += matrix[row * dimension + column]! * vector[column]!;
    }
  }
  return result;
}

function maxAbsolute(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(values[index]!));
  }
  return maximum;
}

function incidentCounts(scene: PrecomputedScene): Uint32Array {
  const counts = new Uint32Array(getVertexCount(scene.mesh));
  for (const vertex of scene.mesh.tetrahedra) {
    counts[vertex] += 1;
  }
  return counts;
}

function candidateBasisBlocks(
  scene: PrecomputedScene,
  precomputation: VertexPrecomputation,
  tetrahedron: number,
): Float64Array {
  const exactBasis = precomputation.exactBasis;
  if (!exactBasis) {
    throw new Error("The validation build did not retain exact bases.");
  }
  const blocks = new Float64Array(36);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = scene.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    blocks.set(
      exactBasis.subarray(vertex * 9, vertex * 9 + 9),
      localVertex * 9,
    );
  }
  return blocks;
}

function distributedElementHessian(
  scene: PrecomputedScene,
  tetrahedron: number,
  counts: Uint32Array,
): Float64Array {
  const hessian = scene.restTetraData.stiffnessMatrices.slice(
    tetrahedron * 144,
    tetrahedron * 144 + 144,
  );
  const inverseTimestepSquared =
    1 / (scene.settings.timestep * scene.settings.timestep);
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = scene.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
    const inertia =
      (scene.lumpedMasses[vertex]! * inverseTimestepSquared) /
      counts[vertex]!;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      const row = localVertex * 3 + coordinate;
      hessian[row * 12 + row] += inertia;
    }
  }
  return hessian;
}

function assembleLocalReducedSystem(
  scene: PrecomputedScene,
  precomputation: VertexPrecomputation,
  fullPose: Float64Array,
  globalGradient: Float64Array,
  candidates: readonly WeightedCandidate[],
  counts: Uint32Array,
): { readonly hessian: Float64Array; readonly gradient: Float64Array } {
  const dimension = scene.restSystem.dimension;
  const localBase = scene.restSystem.vertexToActiveDof[precomputation.vertex]!;
  const hessian = new Float64Array(9);
  const gradient = new Float64Array(3);
  for (let row = 0; row < 3; row += 1) {
    gradient[row] = globalGradient[localBase + row]!;
    for (let column = 0; column < 3; column += 1) {
      hessian[row * 3 + column] =
        scene.restSystem.hessian[
          (localBase + row) * dimension + localBase + column
        ]!;
    }
  }

  for (const candidate of candidates) {
    const elementHessian = distributedElementHessian(
      scene,
      candidate.tetrahedron,
      counts,
    );
    const elementPose = new Float64Array(12);
    let sourceSlot = -1;
    for (let localVertex = 0; localVertex < 4; localVertex += 1) {
      const vertex =
        scene.mesh.tetrahedra[candidate.tetrahedron * 4 + localVertex]!;
      elementPose.set(
        fullPose.subarray(vertex * 3, vertex * 3 + 3),
        localVertex * 3,
      );
      if (vertex === precomputation.vertex) {
        sourceSlot = localVertex;
      }
    }
    const elementGradient = multiplyMatrixVector(
      elementHessian,
      elementPose,
      12,
    );
    const basis = candidate.basisBlocks;
    for (let row = 0; row < 3; row += 1) {
      for (let elementRow = 0; elementRow < 12; elementRow += 1) {
        gradient[row] +=
          candidate.weight *
          basis[elementRow * 3 + row]! *
          elementGradient[elementRow]!;
      }
      if (sourceSlot >= 0) {
        gradient[row] -=
          candidate.weight * elementGradient[sourceSlot * 3 + row]!;
      }
      for (let column = 0; column < 3; column += 1) {
        let reducedEntry = 0;
        for (let elementRow = 0; elementRow < 12; elementRow += 1) {
          for (let elementColumn = 0; elementColumn < 12; elementColumn += 1) {
            reducedEntry +=
              basis[elementRow * 3 + row]! *
              elementHessian[elementRow * 12 + elementColumn]! *
              basis[elementColumn * 3 + column]!;
          }
        }
        if (sourceSlot >= 0) {
          reducedEntry -=
            elementHessian[
              (sourceSlot * 3 + row) * 12 + sourceSlot * 3 + column
            ]!;
        }
        hessian[row * 3 + column] += candidate.weight * reducedEntry;
      }
    }
  }
  return { hessian, gradient };
}

function solveReducedSystem(
  hessian: Float64Array,
  gradient: Float64Array,
): Float64Array {
  const inverse = invert3(hessian);
  const update = multiplyMatrixVector(inverse, gradient, 3);
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    update[coordinate] *= -1;
  }
  return update;
}

function selectedCubatureUpdateRms(scene: PrecomputedScene): number {
  const modes = buildLowFrequencyTrainingPoses(
    scene.mesh,
    scene.restSystem,
    8,
    24,
  );
  const counts = incidentCounts(scene);
  let squaredError = 0;
  let squaredReference = 0;

  for (const fullPose of modes) {
    const activePose = new Float64Array(scene.restSystem.dimension);
    for (const vertex of scene.restSystem.activeVertices) {
      const activeBase = scene.restSystem.vertexToActiveDof[vertex]!;
      activePose.set(
        fullPose.subarray(vertex * 3, vertex * 3 + 3),
        activeBase,
      );
    }
    const globalGradient = multiplyMatrixVector(
      scene.restSystem.hessian,
      activePose,
      scene.restSystem.dimension,
    );
    for (const precomputation of scene.vertexPrecomputations) {
      const reduced = assembleLocalReducedSystem(
        scene,
        precomputation,
        fullPose,
        globalGradient,
        precomputation.cubature,
        counts,
      );
      const update = solveReducedSystem(reduced.hessian, reduced.gradient);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const reference =
          -fullPose[precomputation.vertex * 3 + coordinate]!;
        const error = update[coordinate]! - reference;
        squaredError += error * error;
        squaredReference += reference * reference;
      }
    }
  }

  return Math.sqrt(squaredError / squaredReference);
}

describe("tetrahedral CPU preprocessing", () => {
  it("extracts the deterministic triangulated cuboid boundary", () => {
    const mesh = generateRegularCuboidMesh({ cells: [1, 1, 1] });
    const surface = extractBoundarySurface(mesh);

    expect(getVertexCount(mesh)).toBe(8);
    expect(getTetrahedronCount(mesh)).toBe(6);
    expect(surface.triangles.length / 3).toBe(12);
    expect(surface.edges.length / 2).toBe(18);

    const keys = new Set<string>();
    for (let offset = 0; offset < surface.triangles.length; offset += 3) {
      const key = [
        surface.triangles[offset]!,
        surface.triangles[offset + 1]!,
        surface.triangles[offset + 2]!,
      ]
        .sort((left, right) => left - right)
        .join(",");
      keys.add(key);
    }
    expect(keys.size).toBe(12);

    const twoCellSurface = extractBoundarySurface(
      generateRegularCuboidMesh({ cells: [2, 1, 1] }),
    );
    expect(twoCellSurface.triangles.length / 3).toBe(20);
    expect(twoCellSurface.edges.length / 2).toBe(30);
  });

  it("transforms and appends meshes while preserving topology metadata", () => {
    const first = generateRegularCuboidMesh({
      cells: [1, 1, 1],
      bodyId: 3,
    });
    const second = transformTetrahedralMesh(first, {
      translation: [2, 3, 4],
      scale: [0.5, 1, 1.5],
      rotationEuler: [0, 0, Math.PI / 2],
    });
    const combined = appendTetrahedralMeshes([first, second]);

    expect(getVertexCount(combined)).toBe(16);
    expect(getTetrahedronCount(combined)).toBe(12);
    expect([...combined.bodyIds]).toEqual(Array(16).fill(3));
    expect(combined.tetrahedra[24]).toBe(first.tetrahedra[0]! + 8);
    expect(second.positions[0]).toBeCloseTo(2);
    expect(second.positions[1]).toBeCloseTo(3);
    expect(second.positions[2]).toBeCloseTo(4);
  });

  it("assembles symmetric linear elasticity and conservative lumped mass", () => {
    const mesh = generateRegularCuboidMesh({ cells: [1, 1, 1] });
    const material = {
      name: "test",
      density: 2,
      youngModulus: 10_000,
      poissonRatio: 0.3,
      color: [1, 1, 1, 1] as const,
    };
    const rest = computeRestTetraData(mesh, [material]);
    const masses = computeLumpedMasses(mesh, [material], rest);
    const totalVolume = [...rest.volumes].reduce((sum, value) => sum + value, 0);

    expect(totalVolume).toBeCloseTo(1, 12);
    expect([...masses].reduce((sum, value) => sum + value, 0)).toBeCloseTo(
      material.density * totalVolume,
      12,
    );
    for (let row = 0; row < 12; row += 1) {
      for (let column = 0; column < 12; column += 1) {
        expect(rest.stiffnessMatrices[row * 12 + column]).toBeCloseTo(
          rest.stiffnessMatrices[column * 12 + row],
          10,
        );
      }
    }
  });
});

describe("exact JGS2 rest perturbation basis", () => {
  let scene: PrecomputedScene;

  beforeAll(() => {
    scene = buildPrecomputedScene(buildSceneDefinition("minimal"), {
      retainExactBases: true,
    });
  });

  it("satisfies S_i Ubar_i = I and complementary equilibrium", () => {
    const precomputation = scene.vertexPrecomputations[0]!;
    const exactBasis = precomputation.exactBasis;
    expect(exactBasis).toBeDefined();
    if (!exactBasis) {
      throw new Error("The validation build did not retain exact bases.");
    }

    const localBlock = exactBasis.subarray(
      precomputation.vertex * 9,
      precomputation.vertex * 9 + 9,
    );
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        expect(localBlock[row * 3 + column]).toBeCloseTo(
          row === column ? 1 : 0,
          9,
        );
      }
    }

    const system = scene.restSystem;
    const activeBasis = exactBasisToActiveMatrix(exactBasis, system);
    const equilibrium = new Float64Array(system.dimension * 3);
    for (let row = 0; row < system.dimension; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        for (let inner = 0; inner < system.dimension; inner += 1) {
          equilibrium[row * 3 + column] +=
            system.hessian[row * system.dimension + inner]! *
            activeBasis[inner * 3 + column]!;
        }
      }
    }

    const localBase = system.vertexToActiveDof[precomputation.vertex]!;
    let complementaryMaximum = 0;
    for (let row = 0; row < system.dimension; row += 1) {
      if (row >= localBase && row < localBase + 3) {
        for (let column = 0; column < 3; column += 1) {
          expect(equilibrium[row * 3 + column]).toBeCloseTo(
            precomputation.schurInverse[(row - localBase) * 3 + column],
            6,
          );
        }
      } else {
        for (let column = 0; column < 3; column += 1) {
          complementaryMaximum = Math.max(
            complementaryMaximum,
            Math.abs(equilibrium[row * 3 + column]!),
          );
        }
      }
    }
    expect(complementaryMaximum / maxAbsolute(precomputation.schurInverse)).toBeLessThan(
      1e-9,
    );
  });

  it("builds ordered orthonormal low-frequency rest-Hessian modes", () => {
    const modes = buildLowFrequencyTrainingPoses(
      scene.mesh,
      scene.restSystem,
      8,
      24,
    );
    const activeModes = modes.map((mode) => {
      const active = new Float64Array(scene.restSystem.dimension);
      for (const vertex of scene.restSystem.activeVertices) {
        const activeBase = scene.restSystem.vertexToActiveDof[vertex]!;
        active.set(mode.subarray(vertex * 3, vertex * 3 + 3), activeBase);
      }
      return active;
    });

    expect(activeModes).toHaveLength(8);
    for (let row = 0; row < activeModes.length; row += 1) {
      for (let column = 0; column < activeModes.length; column += 1) {
        let innerProduct = 0;
        for (let index = 0; index < scene.restSystem.dimension; index += 1) {
          innerProduct += activeModes[row]![index]! * activeModes[column]![index]!;
        }
        expect(innerProduct).toBeCloseTo(row === column ? 1 : 0, 8);
      }
    }

    const rayleighQuotients = activeModes.map((mode) => {
      const hessianTimesMode = multiplyMatrixVector(
        scene.restSystem.hessian,
        mode,
        scene.restSystem.dimension,
      );
      let quotient = 0;
      for (let index = 0; index < mode.length; index += 1) {
        quotient += mode[index]! * hessianTimesMode[index]!;
      }
      return quotient;
    });
    for (let mode = 1; mode < rayleighQuotients.length; mode += 1) {
      expect(rayleighQuotients[mode]!).toBeGreaterThanOrEqual(
        rayleighQuotients[mode - 1]! * (1 - 1e-9),
      );
    }
  });

  it("matches the selected block of a full Newton solve (Eq. 7-13 oracle)", () => {
    const precomputation = scene.vertexPrecomputations.at(-1)!;
    const exactBasis = precomputation.exactBasis;
    if (!exactBasis) {
      throw new Error("The validation build did not retain exact bases.");
    }
    const system = scene.restSystem;
    const gradient = Float64Array.from(
      { length: system.dimension },
      (_unused, index) =>
        Math.sin((index + 1) * 0.37) + 0.2 * Math.cos((index + 1) * 0.11),
    );
    const globalNewton = solveCholesky(
      system.choleskyLower,
      gradient,
      system.dimension,
    );
    for (let index = 0; index < globalNewton.length; index += 1) {
      globalNewton[index] *= -1;
    }

    const activeBasis = exactBasisToActiveMatrix(exactBasis, system);
    const hessianTimesBasis = new Float64Array(system.dimension * 3);
    for (let row = 0; row < system.dimension; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        for (let inner = 0; inner < system.dimension; inner += 1) {
          hessianTimesBasis[row * 3 + column] +=
            system.hessian[row * system.dimension + inner]! *
            activeBasis[inner * 3 + column]!;
        }
      }
    }
    const reducedHessian = new Float64Array(9);
    const reducedGradient = new Float64Array(3);
    for (let column = 0; column < 3; column += 1) {
      for (let row = 0; row < system.dimension; row += 1) {
        reducedGradient[column] +=
          activeBasis[row * 3 + column]! * gradient[row]!;
      }
      for (let other = 0; other < 3; other += 1) {
        for (let row = 0; row < system.dimension; row += 1) {
          reducedHessian[column * 3 + other] +=
            activeBasis[row * 3 + column]! *
            hessianTimesBasis[row * 3 + other]!;
        }
      }
    }
    const inverseReducedHessian = invert3(reducedHessian);
    const localUpdate = multiplyMatrixVector(
      inverseReducedHessian,
      reducedGradient,
      3,
    );
    const localBase = system.vertexToActiveDof[precomputation.vertex]!;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      expect(-localUpdate[coordinate]!).toBeCloseTo(
        globalNewton[localBase + coordinate],
        9,
      );
    }
  });

  it("matches every global Newton block with local plus all candidate remainders", () => {
    const fullPose = new Float64Array(getVertexCount(scene.mesh) * 3);
    const activePose = new Float64Array(scene.restSystem.dimension);
    for (const vertex of scene.restSystem.activeVertices) {
      const activeBase = scene.restSystem.vertexToActiveDof[vertex]!;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const value =
          Math.sin((activeBase + coordinate + 1) * 0.29) +
          0.15 * Math.cos((activeBase + coordinate + 1) * 0.47);
        activePose[activeBase + coordinate] = value;
        fullPose[vertex * 3 + coordinate] = value;
      }
    }
    const globalGradient = multiplyMatrixVector(
      scene.restSystem.hessian,
      activePose,
      scene.restSystem.dimension,
    );
    const counts = incidentCounts(scene);

    for (const precomputation of scene.vertexPrecomputations) {
      const candidates = Array.from(
        { length: getTetrahedronCount(scene.mesh) },
        (_unused, tetrahedron): WeightedCandidate => ({
          tetrahedron,
          weight: 1,
          basisBlocks: candidateBasisBlocks(
            scene,
            precomputation,
            tetrahedron,
          ),
        }),
      );
      const reduced = assembleLocalReducedSystem(
        scene,
        precomputation,
        fullPose,
        globalGradient,
        candidates,
        counts,
      );
      const update = solveReducedSystem(reduced.hessian, reduced.gradient);
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        expect(update[coordinate]).toBeCloseTo(
          -fullPose[precomputation.vertex * 3 + coordinate]!,
          8,
        );
      }
    }
  });

  it("keeps every scene's selected-Cubature low-mode solves within 2%", () => {
    for (const sceneId of SCENE_IDS) {
      const candidateScene = buildPrecomputedScene(
        buildSceneDefinition(sceneId),
      );
      const maximumTrainingResidual = Math.max(
        ...candidateScene.vertexPrecomputations.map(
          (precomputation) => precomputation.trainingResidual,
        ),
      );
      const relativeRootMeanSquareError =
        selectedCubatureUpdateRms(candidateScene);

      console.log(
        `${sceneId}: selected-Cubature update RMS ` +
          `${relativeRootMeanSquareError.toFixed(6)}, max raw gradient ` +
          `residual ${maximumTrainingResidual.toFixed(6)}`,
      );
      expect(
        relativeRootMeanSquareError,
        `${sceneId} selected-Cubature reduced-update RMS`,
      ).toBeLessThan(0.02);
      expect(
        maximumTrainingResidual,
        `${sceneId} raw Cubature gradient residual diagnostic`,
      ).toBeLessThan(0.3);
    }
  }, 20_000);

  it("produces bounded, nonnegative complementary cubature", () => {
    let maximumResidual = 0;
    let foundIncidentSample = false;
    for (const precomputation of scene.vertexPrecomputations) {
      maximumResidual = Math.max(
        maximumResidual,
        precomputation.trainingResidual,
      );
      expect(precomputation.cubature.length).toBeLessThanOrEqual(
        scene.settings.cubatureSamples,
      );
      for (const sample of precomputation.cubature) {
        expect(sample.weight).toBeGreaterThanOrEqual(0);
        expect(sample.basisBlocks.length).toBe(36);
        const tetVertices = scene.mesh.tetrahedra.subarray(
          sample.tetrahedron * 4,
          sample.tetrahedron * 4 + 4,
        );
        foundIncidentSample ||= [...tetVertices].includes(
          precomputation.vertex,
        );
      }
    }
    expect(foundIncidentSample).toBe(true);
    expect(maximumResidual).toBeLessThan(0.3);
  });
});
