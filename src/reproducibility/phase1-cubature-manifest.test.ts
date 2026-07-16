import { describe, expect, it } from "vitest";

import manifestJson from "../../manifests/phase1-cubature.v1.json?raw";
import {
  PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID,
  PHASE1_NONLINEAR_CUBATURE_MATERIAL,
  buildPhase1NonlinearCubatureDefinition,
} from "../scenes/phase1";
import {
  NONLINEAR_CUBATURE_CORPUS_VERSION,
  NONLINEAR_CUBATURE_COLUMN_RANK_RELATIVE_TOLERANCE,
  NONLINEAR_CUBATURE_CO_ROTATION,
  NONLINEAR_CUBATURE_DIRECTION_NORMALIZATION,
  NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS,
  NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE,
  NONLINEAR_CUBATURE_INVERSE_ITERATIONS,
  NONLINEAR_CUBATURE_MINIMUM_POSE_DETERMINANT,
  NONLINEAR_CUBATURE_MODE_COUNT,
  NONLINEAR_CUBATURE_MIXTURE_COEFFICIENT_FORMULA,
  NONLINEAR_CUBATURE_ROTATED_VALIDATION_COUNT,
  NONLINEAR_CUBATURE_TRAINING_AMPLITUDE,
  NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
  NONLINEAR_CUBATURE_TARGET_NORMALIZATION,
  NONLINEAR_CUBATURE_RESIDUAL_METRIC,
  NONLINEAR_CUBATURE_SELECTION_METHOD,
  NONLINEAR_CUBATURE_UPDATE_RMS_METRIC,
  NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
  NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE,
  NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE_START,
  NONLINEAR_CUBATURE_VALIDATION_LOW_AMPLITUDE,
  NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
  NONLINEAR_CUBATURE_VALIDATION_PREDICTED_BLEND,
  NONLINEAR_CUBATURE_VALIDATION_ONLY_MODE_COUNT,
  NONLINEAR_CUBATURE_VERTEX_FRAME,
  buildNonlinearCubaturePoseCorpus,
  buildPrecomputedScene,
} from "../simulation/cpu";
import { validatePhase1CubatureManifest } from "./phase1-cubature-manifest";

function manifestValue(): unknown {
  return JSON.parse(manifestJson) as unknown;
}

describe("versioned Phase 1 nonlinear Cubature manifest", () => {
  it("P1-CUBATURE-MANIFEST-001 binds the executable fixture and corpus", () => {
    const manifest = validatePhase1CubatureManifest(manifestValue());
    const definition = buildPhase1NonlinearCubatureDefinition();
    const scene = buildPrecomputedScene(definition, {
      retainExactBases: true,
    });
    const corpus = buildNonlinearCubaturePoseCorpus({
      mesh: scene.mesh,
      restData: scene.restTetraData,
      materials: scene.materials,
      lumpedMasses: scene.lumpedMasses,
      restSystem: scene.restSystem,
    });

    expect(manifest.corpusVersion).toBe(NONLINEAR_CUBATURE_CORPUS_VERSION);
    expect(manifest.fixture).toMatchObject({
      id: PHASE1_NONLINEAR_CUBATURE_FIXTURE_ID,
      generator: {
        id: "regular-cuboid-twelve-tetrahedra-edge-fixed",
        version: "1",
        parameters: {
          positions: [...definition.mesh.positions],
          tetrahedra: [...definition.mesh.tetrahedra],
          fixed: [...definition.mesh.fixed],
          materialIds: [...definition.mesh.materialIds],
          bodyIds: [...definition.mesh.bodyIds],
        },
      },
      material: {
        id: PHASE1_NONLINEAR_CUBATURE_MATERIAL.name,
        model: PHASE1_NONLINEAR_CUBATURE_MATERIAL.model,
        density: PHASE1_NONLINEAR_CUBATURE_MATERIAL.density,
        youngModulus: PHASE1_NONLINEAR_CUBATURE_MATERIAL.youngModulus,
        poissonRatio: PHASE1_NONLINEAR_CUBATURE_MATERIAL.poissonRatio,
      },
      simulation: {
        timestep: definition.settings.timestep,
        gravity: definition.settings.gravity,
        floorY: definition.settings.floorY,
        solverIterations: definition.settings.solverIterations,
      },
    });
    expect(manifest.basis).toMatchObject({
      modeCount: NONLINEAR_CUBATURE_MODE_COUNT,
      validationOnlyModeCount:
        NONLINEAR_CUBATURE_VALIDATION_ONLY_MODE_COUNT,
      inverseIterations: NONLINEAR_CUBATURE_INVERSE_ITERATIONS,
      coRotation: NONLINEAR_CUBATURE_CO_ROTATION,
      vertexFrame: NONLINEAR_CUBATURE_VERTEX_FRAME,
    });
    expect(manifest.training).toMatchObject({
      count: corpus.training.length,
      validCount: corpus.training.length,
      trivialCount: 0,
      signsPerMode: [1, -1],
      amplitude: NONLINEAR_CUBATURE_TRAINING_AMPLITUDE,
      normalization: NONLINEAR_CUBATURE_DIRECTION_NORMALIZATION,
      targetNormalization: NONLINEAR_CUBATURE_TARGET_NORMALIZATION,
      predictedBlend: 0,
    });
    expect(manifest.validation).toMatchObject({
      count: corpus.validation.length,
      mixtureCount: NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
      lowAmplitude: NONLINEAR_CUBATURE_VALIDATION_LOW_AMPLITUDE,
      highAmplitude: NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE,
      highAmplitudeStart:
        NONLINEAR_CUBATURE_VALIDATION_HIGH_AMPLITUDE_START,
      predictedBlend: NONLINEAR_CUBATURE_VALIDATION_PREDICTED_BLEND,
      rotatedCasesWhenUnconstrained:
        NONLINEAR_CUBATURE_ROTATED_VALIDATION_COUNT,
      minimumDeformationDeterminant:
        NONLINEAR_CUBATURE_MINIMUM_POSE_DETERMINANT,
      coefficientFormula:
        NONLINEAR_CUBATURE_MIXTURE_COEFFICIENT_FORMULA,
    });
    expect(manifest.cubature).toMatchObject({
      candidateCount: definition.mesh.tetrahedra.length / 4,
      maximumSamples: definition.settings.cubatureSamples,
      requireRankAboveSampleBudget: true,
      columnRankRelativeTolerance:
        NONLINEAR_CUBATURE_COLUMN_RANK_RELATIVE_TOLERANCE,
      normalizedTrainingResidual:
        NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
      nonnegativeWeights: true,
      runtimePacking: "f32",
      selection: NONLINEAR_CUBATURE_SELECTION_METHOD,
      residualMetric: NONLINEAR_CUBATURE_RESIDUAL_METRIC,
    });
    expect(manifest.updateValidation).toMatchObject({
      trainingRelativeRms: NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
      validationRelativeRms: NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
      combinedRelativeRms: NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
      productionGpuRelativeError:
        NONLINEAR_CUBATURE_GPU_UPDATE_TOLERANCE,
      regularization: "none",
      metric: NONLINEAR_CUBATURE_UPDATE_RMS_METRIC,
      exactReference: "independent-dense-current-projection",
      gpuPrediction: "production-f32-implicit-euler-predictor",
      productionIterations: NONLINEAR_CUBATURE_GPU_PARITY_ITERATIONS,
    });

    const first = corpus.training[0]!;
    const last = corpus.validation.at(-1)!;
    expect(manifest.anchors.firstTraining).toEqual({
      id: first.id,
      minimumDeformationDeterminant: first.minimumDeformationDeterminant,
      positions: [...first.positions],
      predictedPositions: [...first.predictedPositions],
    });
    expect(manifest.anchors.lastValidation).toEqual({
      id: last.id,
      minimumDeformationDeterminant: last.minimumDeformationDeterminant,
      positions: [...last.positions],
      predictedPositions: [...last.predictedPositions],
    });
    expect(manifest.expectedPackedSelections).toEqual(
      scene.vertexPrecomputations.map((precomputation) => ({
        vertex: precomputation.vertex,
        tetrahedra: precomputation.cubature.map(
          (sample) => sample.tetrahedron,
        ),
        weights: precomputation.cubature.map((sample) => sample.weight),
        nonzeroCandidateCount:
          precomputation.nonzeroTrainingCandidateCount,
        trainingColumnRank: precomputation.trainingColumnRank,
        packedTrainingColumnRank:
          precomputation.packedTrainingColumnRank,
        packedResidual: precomputation.trainingResidual,
      })),
    );
  });

  it("P1-CUBATURE-MANIFEST-002 rejects protocol, threshold, anchor, rank, and selection drift", () => {
    const badProtocol = structuredClone(manifestValue()) as {
      basis: { coRotation: string };
    };
    badProtocol.basis.coRotation = "unrotated-rest-basis";
    expect(() => validatePhase1CubatureManifest(badProtocol)).toThrow(
      /basis.coRotation/,
    );

    const badThreshold = structuredClone(manifestValue()) as {
      cubature: { normalizedTrainingResidual: number };
    };
    badThreshold.cubature.normalizedTrainingResidual = -0.1;
    expect(() => validatePhase1CubatureManifest(badThreshold)).toThrow(
      /normalizedTrainingResidual/,
    );

    const badAnchor = structuredClone(manifestValue()) as {
      anchors: { firstTraining: { positions: number[] } };
    };
    badAnchor.anchors.firstTraining.positions.pop();
    expect(() => validatePhase1CubatureManifest(badAnchor)).toThrow(
      /firstTraining.positions/,
    );

    const badSelection = structuredClone(manifestValue()) as {
      expectedPackedSelections: { weights: number[] }[];
    };
    badSelection.expectedPackedSelections[0]!.weights[0] = 1 + 2 ** -30;
    expect(() => validatePhase1CubatureManifest(badSelection)).toThrow(
      /positive f32/,
    );

    const badRank = structuredClone(manifestValue()) as {
      expectedPackedSelections: { packedTrainingColumnRank: number }[];
    };
    badRank.expectedPackedSelections[0]!.packedTrainingColumnRank = 6;
    expect(() => validatePhase1CubatureManifest(badRank)).toThrow(
      /rank above the sample budget/,
    );

    const missingSource = structuredClone(manifestValue()) as {
      expectedPackedSelections: unknown[];
    };
    missingSource.expectedPackedSelections.pop();
    expect(() => validatePhase1CubatureManifest(missingSource)).toThrow(
      /cover every active fixture vertex/,
    );
  });
});
