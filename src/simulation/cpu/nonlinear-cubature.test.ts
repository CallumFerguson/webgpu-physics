import { describe, expect, it } from "vitest";

import {
  PHASE1_STABLE_NEO_HOOKEAN_MATERIAL,
  buildPhase1NonlinearCubatureDefinition,
} from "../../scenes/phase1";
import {
  assembleRestLinearSystem,
  computeLumpedMasses,
  computeRestTetraData,
} from "./fem";
import { relativeError } from "./finite-difference";
import {
  appendTetrahedralMeshes,
  generateRegularCuboidMesh,
} from "./mesh";
import { buildPrecomputedScene } from "./precompute";
import {
  NONLINEAR_CUBATURE_MINIMUM_POSE_DETERMINANT,
  NONLINEAR_CUBATURE_MODE_COUNT,
  NONLINEAR_CUBATURE_TRAINING_AMPLITUDE,
  NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE,
  NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE_START,
  NONLINEAR_CUBATURE_VALIDATION_LOW_AMPLITUDE,
  NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
  NONLINEAR_CUBATURE_VALIDATION_PREDICTED_BLEND,
  buildNonlinearCubaturePoseCorpus,
} from "./nonlinear-cubature";
import { evaluateStableNeoHookeanMesh } from "./stable-neo-hookean";

function dot(left: Float64Array, right: Float64Array): number {
  let value = 0;
  for (let entry = 0; entry < left.length; entry += 1) {
    value += left[entry]! * right[entry]!;
  }
  return value;
}

function orthonormalSpan(vectors: readonly Float64Array[]): Float64Array[] {
  const basis: Float64Array[] = [];
  for (const vector of vectors) {
    const candidate = vector.slice();
    for (const existing of basis) {
      const projection = dot(candidate, existing);
      for (let entry = 0; entry < candidate.length; entry += 1) {
        candidate[entry] -= projection * existing[entry]!;
      }
    }
    const norm = Math.sqrt(dot(candidate, candidate));
    if (norm <= 1e-10) continue;
    for (let entry = 0; entry < candidate.length; entry += 1) {
      candidate[entry] /= norm;
    }
    basis.push(candidate);
  }
  return basis;
}

function relativeDistanceFromSpan(
  vector: Float64Array,
  basis: readonly Float64Array[],
): number {
  const residual = vector.slice();
  for (const existing of basis) {
    const projection = dot(residual, existing);
    for (let entry = 0; entry < residual.length; entry += 1) {
      residual[entry] -= projection * existing[entry]!;
    }
  }
  return Math.sqrt(dot(residual, residual) / dot(vector, vector));
}

function buildCorpusFixture() {
  const definition = buildPhase1NonlinearCubatureDefinition();
  const restData = computeRestTetraData(definition.mesh, definition.materials);
  const lumpedMasses = computeLumpedMasses(
    definition.mesh,
    definition.materials,
    restData,
  );
  const restSystem = assembleRestLinearSystem(
    definition.mesh,
    restData,
    lumpedMasses,
    definition.settings.timestep,
  );
  const options = {
    mesh: definition.mesh,
    restData,
    materials: definition.materials,
    lumpedMasses,
    restSystem,
  };
  return {
    definition,
    restData,
    corpus: buildNonlinearCubaturePoseCorpus(options),
    second: buildNonlinearCubaturePoseCorpus(options),
  };
}

describe("nonlinear Cubature pose corpus", () => {
  it("builds deterministic low-mode training and held-out combination poses", () => {
    const { corpus, second } = buildCorpusFixture();

    expect(corpus).toEqual(second);
    expect(corpus.training).toHaveLength(NONLINEAR_CUBATURE_MODE_COUNT * 2);
    expect(corpus.validation).toHaveLength(
      NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
    );
    expect(corpus.training.map((pose) => pose.id)).toEqual(
      Array.from(
        { length: NONLINEAR_CUBATURE_MODE_COUNT * 2 },
        (_unused, index) =>
          `phase1.nonlinear-cubature-training/${index
            .toString()
            .padStart(2, "0")}`,
      ),
    );
    expect(corpus.validation.map((pose) => pose.id)).toEqual(
      Array.from(
        {
          length: NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
        },
        (_unused, index) =>
          `phase1.nonlinear-cubature-validation/${index
            .toString()
            .padStart(2, "0")}`,
      ),
    );
  });

  it("freezes feasible nonlinear amplitudes and distinct inertial targets", () => {
    const { definition, corpus } = buildCorpusFixture();
    for (const pose of [...corpus.training, ...corpus.validation]) {
      const expectedAmplitude = pose.kind === "training-mode"
        ? NONLINEAR_CUBATURE_TRAINING_AMPLITUDE
        : pose.index < NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE_START ||
            pose.kind === "validation-rotated"
          ? NONLINEAR_CUBATURE_VALIDATION_LOW_AMPLITUDE
          : NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE;
      expect(pose.perturbationAmplitude).toBe(expectedAmplitude);
      expect(pose.minimumDeformationDeterminant).toBeGreaterThanOrEqual(
        NONLINEAR_CUBATURE_MINIMUM_POSE_DETERMINANT,
      );
      expect([...pose.positions].every(Number.isFinite)).toBe(true);
      expect([...pose.predictedPositions].every(Number.isFinite)).toBe(true);
      if (pose.kind === "training-mode") {
        expect(pose.predictedPositions).toEqual(definition.mesh.positions);
      } else if (pose.kind === "validation-combination") {
        for (let entry = 0; entry < pose.positions.length; entry += 1) {
          expect(pose.predictedPositions[entry]).toBeCloseTo(
            definition.mesh.positions[entry]! +
              NONLINEAR_CUBATURE_VALIDATION_PREDICTED_BLEND *
                pose.displacement[entry]!,
            13,
          );
        }
      }
    }
  });

  it("holds out deformational directions outside the fitted modal span", () => {
    const { corpus } = buildCorpusFixture();
    const trainingSpan = orthonormalSpan(
      corpus.training.map((pose) => pose.displacement),
    );
    expect(trainingSpan).toHaveLength(NONLINEAR_CUBATURE_MODE_COUNT);
    for (const pose of corpus.validation) {
      expect(
        relativeDistanceFromSpan(pose.displacement, trainingSpan),
        pose.id,
      ).toBeGreaterThan(1e-3);
    }
  });

  it("adds objective rotated validation cases for an unconstrained body", () => {
    const definition = buildPhase1NonlinearCubatureDefinition();
    const mesh = {
      ...definition.mesh,
      fixed: new Uint8Array(definition.mesh.fixed.length),
    };
    const restData = computeRestTetraData(mesh, definition.materials);
    const lumpedMasses = computeLumpedMasses(
      mesh,
      definition.materials,
      restData,
    );
    const restSystem = assembleRestLinearSystem(
      mesh,
      restData,
      lumpedMasses,
      definition.settings.timestep,
    );
    const corpus = buildNonlinearCubaturePoseCorpus({
      mesh,
      restData,
      materials: definition.materials,
      lumpedMasses,
      restSystem,
    });
    expect(corpus.validation).toHaveLength(
      NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT + 4,
    );
    for (let index = 0; index < 4; index += 1) {
      const source = corpus.validation[index]!;
      const rotated = corpus.validation[
        NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT + index
      ]!;
      expect(rotated.kind).toBe("validation-rotated");
      expect(rotated.minimumDeformationDeterminant).toBeCloseTo(
        source.minimumDeformationDeterminant,
        12,
      );
      const sourceMaterial = evaluateStableNeoHookeanMesh(
        mesh,
        restData,
        definition.materials,
        source.positions,
      );
      const rotatedMaterial = evaluateStableNeoHookeanMesh(
        mesh,
        restData,
        definition.materials,
        rotated.positions,
      );
      expect(rotatedMaterial.energy).toBeCloseTo(sourceMaterial.energy, 9);
    }
  });

  it("uses genuinely current nonlinear tangents rather than a rest-only duplicate", () => {
    const { definition, restData, corpus } = buildCorpusFixture();
    const rest = evaluateStableNeoHookeanMesh(
      definition.mesh,
      restData,
      definition.materials,
      definition.mesh.positions,
    );
    const training = evaluateStableNeoHookeanMesh(
      definition.mesh,
      restData,
      definition.materials,
      corpus.training[0]!.positions,
    );

    expect(relativeError(training.hessian, rest.hessian)).toBeGreaterThan(1e-3);
    expect(Math.max(...training.deformationDeterminants.map((value) => Math.abs(value - 1)))).toBeGreaterThan(1e-3);
  });

  it("preprocesses unconstrained stable solids after removing rigid modes", () => {
    const definition = buildPhase1NonlinearCubatureDefinition();
    const scene = buildPrecomputedScene({
      ...definition,
      id: "phase1.unconstrained-stable-precompute-regression",
      mesh: generateRegularCuboidMesh({
        cells: [1, 1, 1],
        origin: [0, 0, 0],
        size: [1, 1, 1],
      }),
    });

    expect(scene.vertexPrecomputations).toHaveLength(
      scene.mesh.positions.length / 3,
    );
    for (const precomputation of scene.vertexPrecomputations) {
      expect(precomputation.cubatureModel).toBe("stable-neo-hookean");
      expect(precomputation.trainingResidual).toBeLessThanOrEqual(0.01);
    }
  });

  it("removes a separate rigid subspace for every unconstrained component", () => {
    const definition = buildPhase1NonlinearCubatureDefinition();
    const mesh = appendTetrahedralMeshes([
      generateRegularCuboidMesh({
        cells: [1, 1, 1],
        origin: [0, 0, 0],
        size: [1, 1, 1],
        bodyId: 0,
      }),
      generateRegularCuboidMesh({
        cells: [1, 1, 1],
        origin: [2, 0, 0],
        size: [1, 1, 1],
        bodyId: 1,
      }),
    ]);
    const scene = buildPrecomputedScene({
      ...definition,
      id: "phase1.two-free-component-precompute-regression",
      mesh,
    });
    const corpus = buildNonlinearCubaturePoseCorpus({
      mesh,
      restData: scene.restTetraData,
      materials: scene.materials,
      lumpedMasses: scene.lumpedMasses,
      restSystem: scene.restSystem,
    });

    expect(scene.vertexPrecomputations).toHaveLength(16);
    for (const precomputation of scene.vertexPrecomputations) {
      expect(precomputation.validTrainingPoseCount).toBe(
        corpus.training.length,
      );
      expect(precomputation.trivialTrainingPoseCount).toBe(0);
      expect(precomputation.trainingResidual).toBeLessThanOrEqual(0.01);
    }

    for (const pose of corpus.training) {
      for (const body of [0, 1]) {
        const vertices = [...mesh.bodyIds]
          .map((bodyId, vertex) => ({ bodyId, vertex }))
          .filter((entry) => entry.bodyId === body)
          .map((entry) => entry.vertex);
        const center = [0, 0, 0];
        let totalMass = 0;
        for (const vertex of vertices) {
          const mass = scene.lumpedMasses[vertex]!;
          totalMass += mass;
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            center[coordinate] +=
              mass * mesh.positions[vertex * 3 + coordinate]!;
          }
        }
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          center[coordinate] /= totalMass;
          const translationDot = vertices.reduce(
            (sum, vertex) =>
              sum + pose.displacement[vertex * 3 + coordinate]!,
            0,
          );
          expect(Math.abs(translationDot), pose.id).toBeLessThan(1e-9);
        }
        const rotationDots = [0, 0, 0];
        for (const vertex of vertices) {
          const x = mesh.positions[vertex * 3]! - center[0]!;
          const y = mesh.positions[vertex * 3 + 1]! - center[1]!;
          const z = mesh.positions[vertex * 3 + 2]! - center[2]!;
          const dx = pose.displacement[vertex * 3]!;
          const dy = pose.displacement[vertex * 3 + 1]!;
          const dz = pose.displacement[vertex * 3 + 2]!;
          rotationDots[0] += -z * dy + y * dz;
          rotationDots[1] += z * dx - x * dz;
          rotationDots[2] += -y * dx + x * dy;
        }
        for (const value of rotationDots) {
          expect(Math.abs(value), pose.id).toBeLessThan(1e-9);
        }
      }
    }
  });

  it("ignores unreferenced materials when selecting the nonlinear path", () => {
    const definition = buildPhase1NonlinearCubatureDefinition();
    const scene = buildPrecomputedScene({
      ...definition,
      materials: [
        ...definition.materials,
        {
          ...definition.materials[0]!,
          name: "unused linear regression material",
          model: "corotated-linear",
        },
      ],
    });

    expect(scene.vertexPrecomputations.every(
      (entry) => entry.cubatureModel === "stable-neo-hookean",
    )).toBe(true);
  });

  it("rejects stable preprocessing that misses the packed residual gate", () => {
    const definition = buildPhase1NonlinearCubatureDefinition();
    expect(() =>
      buildPrecomputedScene({
        ...definition,
        id: "phase1.bad-cubature-quality-regression",
        mesh: generateRegularCuboidMesh({
          cells: [2, 1, 1],
          origin: [0, 0, 0],
          size: [1, 1, 1],
          fixed: (_position, [x]) => x === 0,
        }),
        materials: [PHASE1_STABLE_NEO_HOOKEAN_MATERIAL],
      }),
    ).toThrow(/packed training residual/i);
  });
});
