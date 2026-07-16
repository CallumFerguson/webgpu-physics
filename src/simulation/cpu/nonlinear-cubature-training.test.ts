import { describe, expect, it } from "vitest";

import { buildPhase1NonlinearCubatureDefinition } from "../../scenes/phase1";
import { relativeError } from "./finite-difference";
import {
  assembleRestLinearSystem,
  computeLumpedMasses,
  computeRestTetraData,
} from "./fem";
import { evaluateExactStableNeoHookeanJGS2Local } from "./jgs2-local";
import { choleskyFactor } from "./math";
import {
  NONLINEAR_CUBATURE_MODE_COUNT,
  NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT,
  NONLINEAR_CUBATURE_VALIDATION_ROTATIONS,
  buildNonlinearCubaturePoseCorpus,
  type NonlinearCubaturePose,
} from "./nonlinear-cubature";
import {
  assembleStableNeoHookeanJGS2LocalSystem,
  buildCurrentCoRotatedEquilibriumBasis,
  currentEquilibriumBasisToActiveMatrix,
  NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
  NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
  selectStableNeoHookeanCubatureSamples,
  type StableNeoHookeanCubatureContext,
} from "./nonlinear-cubature-training";
import { createStableNeoHookeanImplicitEulerOracle } from "./nonlinear-oracle";
import { activeCoordinatesFromFullPositions } from "./oracle";
import {
  buildPrecomputedScene,
  computeExactVertexBasis,
} from "./precompute";
import type { CubatureSample } from "./types";

interface TrainedSelection {
  readonly f64: readonly CubatureSample[];
  readonly packed: readonly CubatureSample[];
  readonly packedExactBasis: Float64Array;
}

interface UpdateQuality {
  readonly relativeRms: number;
  readonly maximumSystemRelativeError: number;
  readonly worstPoseId: string;
  readonly worstVertex: number;
  readonly systemCount: number;
}

function buildFixture() {
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
  const corpus = buildNonlinearCubaturePoseCorpus({
    mesh: definition.mesh,
    restData,
    materials: definition.materials,
    lumpedMasses,
    restSystem,
  });
  const contexts = new Map<number, StableNeoHookeanCubatureContext>();
  for (const vertex of restSystem.activeVertices) {
    contexts.set(vertex, {
      mesh: definition.mesh,
      restData,
      materials: definition.materials,
      lumpedMasses,
      restSystem,
      timestep: definition.settings.timestep,
      predictedPositions: definition.mesh.positions,
      sourceVertex: vertex,
      exactBasis: computeExactVertexBasis(
        definition.mesh,
        restSystem,
        vertex,
      ).basis,
    });
  }
  return { definition, restData, lumpedMasses, restSystem, corpus, contexts };
}

function trainSelections(
  fixture: ReturnType<typeof buildFixture>,
  enforceResidual = true,
): Map<number, TrainedSelection> {
  const selections = new Map<number, TrainedSelection>();
  let maximumResidual = 0;
  const residuals: {
    vertex: number;
    residual: number;
    packedResidual: number;
    samples: number;
    rank: number;
  }[] = [];
  for (const vertex of fixture.restSystem.activeVertices) {
    const selection = selectStableNeoHookeanCubatureSamples(
      fixture.contexts.get(vertex)!,
      fixture.corpus.training,
      fixture.definition.settings.cubatureSamples,
    );
    maximumResidual = Math.max(maximumResidual, selection.residual);
    residuals.push({
      vertex,
      residual: selection.residual,
      packedResidual: selection.packedResidual,
      samples: selection.samples.length,
      rank: selection.trainingColumnRank,
    });
    expect(selection.validTrainingPoseCount).toBe(
      fixture.corpus.training.length,
    );
    expect(selection.trivialTrainingPoseCount).toBe(0);
    expect(selection.nonzeroCandidateCount).toBeGreaterThan(
      fixture.definition.settings.cubatureSamples,
    );
    expect(selection.trainingColumnRank).toBeGreaterThan(
      fixture.definition.settings.cubatureSamples,
    );
    expect(selection.packedNonzeroCandidateCount).toBeGreaterThan(
      fixture.definition.settings.cubatureSamples,
    );
    expect(selection.packedTrainingColumnRank).toBeGreaterThan(
      fixture.definition.settings.cubatureSamples,
    );
    expect(selection.samples.length).toBeLessThanOrEqual(
      fixture.definition.settings.cubatureSamples,
    );
    for (const sample of selection.samples) {
      expect(sample.weight).toBeGreaterThan(0);
      expect(sample.basisBlocks).toEqual(
        Float64Array.from(
          { length: 36 },
          (_unused, index) => {
            const localVertex = Math.floor(index / 9);
            const blockIndex = index % 9;
            const meshVertex = fixture.definition.mesh.tetrahedra[
              sample.tetrahedron * 4 + localVertex
            ]!;
            return fixture.contexts.get(vertex)!.exactBasis[
              meshVertex * 9 + blockIndex
            ]!;
          },
        ),
      );
    }
    for (const sample of selection.packedSamples) {
      expect(Math.fround(sample.weight)).toBe(sample.weight);
      expect(sample.weight).toBeGreaterThan(0);
      for (const value of sample.basisBlocks) {
        expect(Math.fround(value)).toBe(value);
      }
    }
    const packedExactBasis = Float64Array.from(
      Float32Array.from(fixture.contexts.get(vertex)!.exactBasis),
    );
    let squaredPackedResidual = 0;
    let normalizedTargetBlocks = 0;
    for (const pose of fixture.corpus.training) {
      const exact = assembleStableNeoHookeanJGS2LocalSystem(
        fixture.contexts.get(vertex)!,
        pose.positions,
      );
      const packedSelected = assembleStableNeoHookeanJGS2LocalSystem(
        {
          ...fixture.contexts.get(vertex)!,
          exactBasis: packedExactBasis,
        },
        pose.positions,
        selection.packedSamples,
      );
      const targetNorm = Math.hypot(...exact.remainderGradient);
      if (targetNorm === 0) continue;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const normalizedError =
          (packedSelected.remainderGradient[coordinate]! -
            exact.remainderGradient[coordinate]!) /
          targetNorm;
        squaredPackedResidual += normalizedError * normalizedError;
      }
      normalizedTargetBlocks += 1;
    }
    expect(normalizedTargetBlocks).toBe(selection.validTrainingPoseCount);
    const reconstructedPackedResidual = Math.sqrt(
      squaredPackedResidual / normalizedTargetBlocks,
    );
    expect(
      Math.abs(reconstructedPackedResidual - selection.packedResidual),
    ).toBeLessThan(2e-9);
    selections.set(vertex, {
      f64: selection.samples,
      packed: selection.packedSamples,
      packedExactBasis,
    });
  }
  console.log(
    `Phase 1 nonlinear Cubature residuals: ${residuals
      .map(
        ({ vertex, residual, packedResidual, samples, rank }) =>
          `v${vertex}=${residual.toExponential(3)}/` +
          `${packedResidual.toExponential(3)}(${samples},rank ${rank})`,
      )
      .join(", ")}; maximum ${maximumResidual.toExponential(6)}`,
  );
  if (enforceResidual) {
    for (const entry of residuals) {
      expect(
        entry.residual,
        `vertex ${entry.vertex} training residual`,
      ).toBeLessThanOrEqual(
        NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE + 1e-12,
      );
      expect(
        entry.packedResidual,
        `vertex ${entry.vertex} packed training residual`,
      ).toBeLessThanOrEqual(
        NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE + 1e-12,
      );
    }
  }
  return selections;
}

function updateRms(
  fixture: ReturnType<typeof buildFixture>,
  poses: readonly NonlinearCubaturePose[],
  selections: ReadonlyMap<number, TrainedSelection>,
  packed: boolean,
): UpdateQuality {
  let squaredError = 0;
  let squaredReference = 0;
  let maximumSystemRelativeError = 0;
  let worstPoseId = "";
  let worstVertex = -1;
  let systemCount = 0;
  for (const pose of poses) {
    for (const vertex of fixture.restSystem.activeVertices) {
      const context = fixture.contexts.get(vertex)!;
      const trained = selections.get(vertex)!;
      const poseContext = {
        ...context,
        predictedPositions: pose.predictedPositions,
      };
      const currentBasis = buildCurrentCoRotatedEquilibriumBasis(
        poseContext,
        pose.positions,
      );
      const activeBasis = currentEquilibriumBasisToActiveMatrix(
        currentBasis,
        fixture.restSystem,
      );
      const oracle = createStableNeoHookeanImplicitEulerOracle({
        mesh: fixture.definition.mesh,
        restData: fixture.restData,
        materials: fixture.definition.materials,
        lumpedMasses: fixture.lumpedMasses,
        restSystem: fixture.restSystem,
        timestep: fixture.definition.settings.timestep,
        predictedPositions: pose.predictedPositions,
      });
      const exact = evaluateExactStableNeoHookeanJGS2Local({
        oracle,
        restSystem: fixture.restSystem,
        vertex,
        coordinates: activeCoordinatesFromFullPositions(
          pose.positions,
          fixture.restSystem,
        ),
        equilibriumBasis: activeBasis,
      });
      const allCandidates = assembleStableNeoHookeanJGS2LocalSystem(
        poseContext,
        pose.positions,
      );
      const selected = assembleStableNeoHookeanJGS2LocalSystem(
        packed
          ? { ...poseContext, exactBasis: trained.packedExactBasis }
          : poseContext,
        pose.positions,
        packed ? trained.packed : trained.f64,
      );
      expect(exact.newtonUpdate, `${pose.id} vertex ${vertex} exact update`).toBeDefined();
      expect(
        relativeError(allCandidates.gradient, exact.gradient),
        `${pose.id} vertex ${vertex} all-candidate gradient`,
      ).toBeLessThan(1e-10);
      expect(
        relativeError(allCandidates.hessian, exact.hessian),
        `${pose.id} vertex ${vertex} all-candidate Hessian`,
      ).toBeLessThan(1e-10);
      expect(selected.newtonUpdate, `${pose.id} vertex ${vertex} selected update`).toBeDefined();
      choleskyFactor(exact.hessian, 3);
      choleskyFactor(selected.hessian, 3);
      let systemSquaredError = 0;
      let systemSquaredReference = 0;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const reference = exact.newtonUpdate![coordinate]!;
        const error = selected.newtonUpdate![coordinate]! - reference;
        squaredError += error * error;
        squaredReference += reference * reference;
        systemSquaredError += error * error;
        systemSquaredReference += reference * reference;
      }
      const systemRelativeError = Math.sqrt(
        systemSquaredError / Math.max(systemSquaredReference, 1e-30),
      );
      if (systemRelativeError > maximumSystemRelativeError) {
        maximumSystemRelativeError = systemRelativeError;
        worstPoseId = pose.id;
        worstVertex = vertex;
      }
      systemCount += 1;
    }
  }
  if (!(squaredReference > 1e-30)) {
    throw new Error("Nonlinear Cubature update corpus has a zero reference norm.");
  }
  return {
    relativeRms: Math.sqrt(squaredError / squaredReference),
    maximumSystemRelativeError,
    worstPoseId,
    worstVertex,
    systemCount,
  };
}

describe("stable Neo-Hookean nonlinear Cubature", () => {
  it("reconstructs the exact current co-rotated dense restriction with all candidates", () => {
    const fixture = buildFixture();
    const vertex = fixture.restSystem.activeVertices.at(-1)!;
    const context = fixture.contexts.get(vertex)!;
    const pose = fixture.corpus.training[0]!;
    const currentBasis = buildCurrentCoRotatedEquilibriumBasis(
      context,
      pose.positions,
    );
    const activeBasis = currentEquilibriumBasisToActiveMatrix(
      currentBasis,
      fixture.restSystem,
    );
    const oracle = createStableNeoHookeanImplicitEulerOracle({
      mesh: fixture.definition.mesh,
      restData: fixture.restData,
      materials: fixture.definition.materials,
      lumpedMasses: fixture.lumpedMasses,
      restSystem: fixture.restSystem,
      timestep: fixture.definition.settings.timestep,
      predictedPositions: fixture.definition.mesh.positions,
    });
    const coordinates = activeCoordinatesFromFullPositions(
      pose.positions,
      fixture.restSystem,
    );
    const dense = evaluateExactStableNeoHookeanJGS2Local({
      oracle,
      restSystem: fixture.restSystem,
      vertex,
      coordinates,
      equilibriumBasis: activeBasis,
    });
    const decomposed = assembleStableNeoHookeanJGS2LocalSystem(
      context,
      pose.positions,
    );

    expect(relativeError(decomposed.gradient, dense.gradient)).toBeLessThan(1e-12);
    expect(relativeError(decomposed.hessian, dense.hessian)).toBeLessThan(1e-12);
    expect(relativeError(decomposed.newtonUpdate!, dense.newtonUpdate!)).toBeLessThan(1e-12);
  });

  it("meets the paper's one-percent nonlinear training residual with six samples", () => {
    trainSelections(buildFixture());
  });

  it("keeps selected updates within two-percent RMS on training and held-out poses", () => {
    const fixture = buildFixture();
    const selections = trainSelections(fixture, false);
    const trainingRms = updateRms(
      fixture,
      fixture.corpus.training,
      selections,
      false,
    );
    const validationRms = updateRms(
      fixture,
      fixture.corpus.validation,
      selections,
      false,
    );
    const packedTrainingRms = updateRms(
      fixture,
      fixture.corpus.training,
      selections,
      true,
    );
    const packedValidationRms = updateRms(
      fixture,
      fixture.corpus.validation,
      selections,
      true,
    );
    const combinedRms = updateRms(
      fixture,
      [...fixture.corpus.training, ...fixture.corpus.validation],
      selections,
      false,
    );
    const packedCombinedRms = updateRms(
      fixture,
      [...fixture.corpus.training, ...fixture.corpus.validation],
      selections,
      true,
    );
    console.log(
      `Phase 1 nonlinear Cubature update RMS: training ` +
        `${trainingRms.relativeRms.toExponential(6)}, validation ` +
        `${validationRms.relativeRms.toExponential(6)}, combined ` +
        `${combinedRms.relativeRms.toExponential(6)}; packed training ` +
        `${packedTrainingRms.relativeRms.toExponential(6)}, packed validation ` +
        `${packedValidationRms.relativeRms.toExponential(6)}, packed combined ` +
        `${packedCombinedRms.relativeRms.toExponential(6)}; packed worst ` +
        `${packedCombinedRms.maximumSystemRelativeError.toExponential(6)} at ` +
        `${packedCombinedRms.worstPoseId} v${packedCombinedRms.worstVertex}`,
    );
    expect(trainingRms.relativeRms).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
    );
    expect(validationRms.relativeRms).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
    );
    expect(combinedRms.relativeRms).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
    );
    expect(packedTrainingRms.relativeRms).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
    );
    expect(packedValidationRms.relativeRms).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
    );
    expect(packedCombinedRms.relativeRms).toBeLessThanOrEqual(
      NONLINEAR_CUBATURE_UPDATE_RMS_TOLERANCE,
    );
    expect(packedTrainingRms.systemCount).toBe(
      fixture.corpus.training.length * fixture.restSystem.activeVertices.length,
    );
    expect(packedValidationRms.systemCount).toBe(
      fixture.corpus.validation.length * fixture.restSystem.activeVertices.length,
    );
  });

  it("routes stable scene preprocessing through packed nonlinear Cubature", () => {
    const definition = buildPhase1NonlinearCubatureDefinition();
    const scene = buildPrecomputedScene(definition, {
      retainExactBases: true,
    });
    expect(scene.vertexPrecomputations).toHaveLength(
      scene.restSystem.activeVertices.length,
    );
    for (const precomputation of scene.vertexPrecomputations) {
      expect(precomputation.cubatureModel).toBe("stable-neo-hookean");
      expect(precomputation.trainingResidualF64).toBeLessThanOrEqual(
        NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
      );
      expect(precomputation.trainingResidual).toBeLessThanOrEqual(
        NONLINEAR_CUBATURE_TRAINING_RESIDUAL_TOLERANCE,
      );
      expect(precomputation.cubature.length).toBeLessThanOrEqual(
        definition.settings.cubatureSamples,
      );
      expect(precomputation.cubature).toHaveLength(
        definition.settings.cubatureSamples,
      );
      expect(precomputation.validTrainingPoseCount).toBe(
        NONLINEAR_CUBATURE_MODE_COUNT * 2,
      );
      expect(precomputation.trivialTrainingPoseCount).toBe(0);
      expect(precomputation.nonzeroTrainingCandidateCount).toBeGreaterThan(
        definition.settings.cubatureSamples,
      );
      expect(precomputation.trainingColumnRank).toBeGreaterThan(
        definition.settings.cubatureSamples,
      );
      expect(
        precomputation.packedNonzeroTrainingCandidateCount,
      ).toBeGreaterThan(definition.settings.cubatureSamples);
      expect(precomputation.packedTrainingColumnRank).toBeGreaterThan(
        definition.settings.cubatureSamples,
      );
      for (const sample of precomputation.cubature) {
        expect(Math.fround(sample.weight)).toBe(sample.weight);
        expect(sample.weight).toBeGreaterThan(0);
        for (const value of sample.basisBlocks) {
          expect(Math.fround(value)).toBe(value);
        }
      }
    }
  });

  it("rotates packed selected updates covariantly on an unconstrained body", () => {
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
    const sourceVertex = restSystem.activeVertices[0]!;
    const packedExactBasis = Float64Array.from(
      Float32Array.from(
        computeExactVertexBasis(mesh, restSystem, sourceVertex).basis,
      ),
    );
    const baseContext: StableNeoHookeanCubatureContext = {
      mesh,
      restData,
      materials: definition.materials,
      lumpedMasses,
      restSystem,
      timestep: definition.settings.timestep,
      predictedPositions: mesh.positions,
      sourceVertex,
      exactBasis: packedExactBasis,
    };
    const selection = selectStableNeoHookeanCubatureSamples(
      baseContext,
      corpus.training,
      definition.settings.cubatureSamples,
    );
    const sourcePose = corpus.validation[0]!;
    const rotatedPose = corpus.validation[
      NONLINEAR_CUBATURE_VALIDATION_MIXTURE_COUNT
    ]!;
    const source = assembleStableNeoHookeanJGS2LocalSystem(
      { ...baseContext, predictedPositions: sourcePose.predictedPositions },
      sourcePose.positions,
      selection.packedSamples,
    );
    const rotated = assembleStableNeoHookeanJGS2LocalSystem(
      { ...baseContext, predictedPositions: rotatedPose.predictedPositions },
      rotatedPose.positions,
      selection.packedSamples,
    );
    const rotation = NONLINEAR_CUBATURE_VALIDATION_ROTATIONS[0]!;
    const expected = new Float64Array(3);
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) {
        expected[row] +=
          rotation[row * 3 + column]! * source.newtonUpdate![column]!;
      }
    }

    expect(rotatedPose.kind).toBe("validation-rotated");
    expect(relativeError(rotated.newtonUpdate!, expected)).toBeLessThan(1e-10);
  });
});
