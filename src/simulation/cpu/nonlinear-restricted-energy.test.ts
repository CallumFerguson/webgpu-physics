import { describe, expect, it } from "vitest";

import { buildPhase1NonlinearCubatureDefinition } from "../../scenes/phase1";
import {
  centralDifferenceGradient,
  centralDifferenceHessian,
  relativeError,
} from "./finite-difference";
import {
  assembleRestLinearSystem,
  computeLumpedMasses,
  computeRestTetraData,
} from "./fem";
import {
  PHASE1_ACCEPTED_DETERMINANT_FLOOR,
  JGS2_LOCAL_MAX_BACKTRACKS,
  JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
  lineSearchRestrictedJGS2LocalDirection,
} from "./nonlinear-globalization";
import {
  buildNonlinearCubaturePoseCorpus,
} from "./nonlinear-cubature";
import {
  assembleStableNeoHookeanJGS2LocalSystem,
  selectStableNeoHookeanCubatureSamples,
  type StableNeoHookeanCubatureContext,
} from "./nonlinear-cubature-training";
import {
  createSelectedCubatureRestrictedLocalModel,
  globalizeSelectedCubatureRestrictedLocal,
} from "./nonlinear-restricted-energy";
import { computeExactVertexBasis } from "./precompute";
import type { CubatureSample, TetrahedralMesh } from "./types";

function buildRestrictedFixture() {
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
  const sourceVertex = restSystem.activeVertices[0]!;
  const context: StableNeoHookeanCubatureContext = {
    mesh: definition.mesh,
    restData,
    materials: definition.materials,
    lumpedMasses,
    restSystem,
    timestep: definition.settings.timestep,
    predictedPositions: definition.mesh.positions,
    sourceVertex,
    exactBasis: computeExactVertexBasis(
      definition.mesh,
      restSystem,
      sourceVertex,
    ).basis,
  };
  const selection = selectStableNeoHookeanCubatureSamples(
    context,
    corpus.training,
    definition.settings.cubatureSamples,
  );
  return {
    context,
    pose: corpus.validation[0]!,
    samples: selection.packedSamples,
  };
}

function buildOverflowRestrictedFixture() {
  const definition = buildPhase1NonlinearCubatureDefinition();
  const mesh: TetrahedralMesh = {
    positions: new Float64Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      3, 0, 0,
      4, 0, 0,
      3, 1, 0,
      3, 0, 1,
    ]),
    tetrahedra: new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7]),
    materialIds: new Uint16Array([0, 0]),
    fixed: new Uint8Array(8),
    bodyIds: new Uint16Array([0, 0, 0, 0, 1, 1, 1, 1]),
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
  const sourceVertex = 0;
  const context: StableNeoHookeanCubatureContext = {
    mesh,
    restData,
    materials: definition.materials,
    lumpedMasses,
    restSystem,
    timestep: definition.settings.timestep,
    predictedPositions: mesh.positions,
    sourceVertex,
    exactBasis: computeExactVertexBasis(
      mesh,
      restSystem,
      sourceVertex,
    ).basis,
  };
  const basisBlocks = new Float64Array(36);
  const localPositions = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ] as const;
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      basisBlocks[localVertex * 9 + coordinate * 3] =
        -localPositions[localVertex]![coordinate]!;
    }
  }
  return {
    context,
    positions: mesh.positions.slice(),
    samples: [{ tetrahedron: 1, weight: 1, basisBlocks }],
  };
}

describe("selected-Cubature restricted JGS2 energy", () => {
  it("matches its packed selected gradient/Hessian and satisfies globalization", () => {
    const fixture = buildRestrictedFixture();
    const model = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
    });
    const zero = new Float64Array(3);
    const finiteGradient = centralDifferenceGradient(model.energy, zero);
    const finiteHessian = centralDifferenceHessian(
      (local) => centralDifferenceGradient(model.energy, local),
      zero,
    );

    expect(
      relativeError(model.initialSystem.gradient, finiteGradient),
    ).toBeLessThan(2e-6);
    expect(
      relativeError(model.initialSystem.hessian, finiteHessian),
    ).toBeLessThan(2e-4);
    expect(model.minimumDeformationDeterminant(zero)).toBeGreaterThan(
      PHASE1_ACCEPTED_DETERMINANT_FLOOR,
    );

    const globalization = globalizeSelectedCubatureRestrictedLocal({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
    });
    expect(globalization.direction.accepted).toBe(true);
    expect(globalization.direction.gradientDotDirection).toBeLessThan(0);
    expect(globalization.direction.normalizedShift).toBeLessThanOrEqual(
      JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
    );
    expect(globalization.lineSearch?.accepted).toBe(true);
    expect(globalization.lineSearch!.acceptedEnergy).toBeLessThanOrEqual(
      globalization.lineSearch!.armijoBound,
    );
    expect(
      globalization.lineSearch!.minimumDeformationDeterminant,
    ).toBeGreaterThan(PHASE1_ACCEPTED_DETERMINANT_FLOOR);
  });

  it("uses the basis blocks stored in each selected runtime record", () => {
    const fixture = buildRestrictedFixture();
    const modified: CubatureSample[] = fixture.samples.map((sample, index) => ({
      tetrahedron: sample.tetrahedron,
      weight: sample.weight,
      basisBlocks: Float64Array.from(
        sample.basisBlocks,
        (value) => (index === 0 ? value * 0.9 : value),
      ),
    }));
    const packedSystem = assembleStableNeoHookeanJGS2LocalSystem(
      fixture.context,
      fixture.pose.positions,
      modified,
    );
    const legacyIgnoredBasisSystem = assembleStableNeoHookeanJGS2LocalSystem(
      fixture.context,
      fixture.pose.positions,
      modified.map(({ tetrahedron, weight }) => ({ tetrahedron, weight })),
    );

    expect(
      relativeError(packedSystem.gradient, legacyIgnoredBasisSystem.gradient),
    ).toBeGreaterThan(1e-5);
    expect(
      relativeError(packedSystem.hessian, legacyIgnoredBasisSystem.hessian),
    ).toBeGreaterThan(1e-5);

    const model = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: modified,
    });
    expect(
      relativeError(
        model.initialSystem.gradient,
        centralDifferenceGradient(model.energy, new Float64Array(3)),
      ),
    ).toBeLessThan(2e-6);
  });

  it("ignores zero-weight GPU ABI slots before geometry and basis validation", () => {
    const fixture = buildRestrictedFixture();
    const withoutPadding = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
    });
    const withPadding = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: [
        ...fixture.samples,
        {
          tetrahedron: 0xffffffff,
          weight: 0,
          basisBlocks: new Float64Array(0),
        },
        {
          tetrahedron: fixture.samples[0]!.tetrahedron,
          weight: 0,
          basisBlocks: new Float64Array(36).fill(1e6),
        },
      ],
    });
    const displacement = new Float64Array([1e-4, -2e-4, 3e-4]);
    expect(withPadding.frozenSamples).toHaveLength(fixture.samples.length);
    expect(withPadding.energy(displacement)).toBe(
      withoutPadding.energy(displacement),
    );
    expect(withPadding.minimumDeformationDeterminant(displacement)).toBe(
      withoutPadding.minimumDeformationDeterminant(displacement),
    );
    expect(withPadding.initialSystem.gradient).toEqual(
      withoutPadding.initialSystem.gradient,
    );
    expect(withPadding.initialSystem.hessian).toEqual(
      withoutPadding.initialSystem.hessian,
    );
  });

  it("propagates positive determinant overflow and skips every trial energy", () => {
    const fixture = buildOverflowRestrictedFixture();
    const model = createSelectedCubatureRestrictedLocalModel(fixture);
    const direction = new Float64Array([-Number.MAX_VALUE, 0, 0]);
    expect(model.minimumDeformationDeterminant(direction)).toBe(
      Number.POSITIVE_INFINITY,
    );
    let energyEvaluationCount = 0;
    const result = lineSearchRestrictedJGS2LocalDirection({
      initialEnergy: 0,
      gradient: new Float64Array([1, 0, 0]),
      direction,
      minimumDeformationDeterminant: (alpha) =>
        model.minimumDeformationDeterminant(
          Float64Array.from(direction, (value) => alpha * value),
        ),
      energy: () => {
        energyEvaluationCount += 1;
        return 0;
      },
    });
    expect(result.accepted).toBe(false);
    expect(result.nonFiniteTrialCount).toBe(JGS2_LOCAL_MAX_BACKTRACKS + 1);
    expect(result.energyEvaluationCount).toBe(0);
    expect(energyEvaluationCount).toBe(0);

    const invalidSource = fixture.positions.slice();
    invalidSource.set(
      [
        0, 0, 0,
        Number.MAX_VALUE, 0, 0,
        0, Number.MAX_VALUE, 0,
        0, 0, Number.MAX_VALUE,
      ],
      0,
    );
    expect(() =>
      createSelectedCubatureRestrictedLocalModel({
        ...fixture,
        positions: invalidSource,
      }),
    ).toThrow(/source pose/i);
  });

  it("keeps the frozen scalar and derivatives isolated from exposed mutable views", () => {
    const fixture = buildRestrictedFixture();
    const positions = fixture.pose.positions.slice();
    const samples = fixture.samples.map((sample) => ({
      tetrahedron: sample.tetrahedron,
      weight: sample.weight,
      basisBlocks: sample.basisBlocks.slice(),
    }));
    const model = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions,
      samples,
    });
    const displacement = new Float64Array([1e-4, -2e-4, 3e-4]);
    const expectedEnergy = model.energy(displacement);
    const expectedDeterminant =
      model.minimumDeformationDeterminant(displacement);
    const expectedSystem = model.initialSystem;

    model.basePositions.fill(123);
    model.sourceIncidentTetrahedra.fill(0);
    model.frozenSamples[0]!.currentBasisBlocks.fill(-456);
    const exposedSystem = model.initialSystem;
    exposedSystem.gradient.fill(789);
    exposedSystem.hessian.fill(789);
    exposedSystem.candidates[0]!.currentBasisBlocks.fill(789);
    positions.fill(-321);
    samples[0]!.basisBlocks.fill(654);
    fixture.context.predictedPositions.fill(42);
    fixture.context.lumpedMasses.fill(42);

    expect(model.basePositions).not.toBe(model.basePositions);
    expect(model.initialSystem).not.toBe(model.initialSystem);
    expect(model.energy(displacement)).toBe(expectedEnergy);
    expect(model.minimumDeformationDeterminant(displacement)).toBe(
      expectedDeterminant,
    );
    expect(model.initialSystem.gradient).toEqual(expectedSystem.gradient);
    expect(model.initialSystem.hessian).toEqual(expectedSystem.hessian);
    expect(model.initialSystem.candidates[0]!.currentBasisBlocks).toEqual(
      expectedSystem.candidates[0]!.currentBasisBlocks,
    );
  });

  it("globalizes all 240 nonzero canonical packed material/inertia systems", () => {
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
    let systemCount = 0;
    let maximumNormalizedShift = 0;
    let maximumLinearResidual = 0;
    let maximumBacktracks = 0;
    let minimumAcceptedDeterminant = Number.POSITIVE_INFINITY;

    for (const sourceVertex of restSystem.activeVertices) {
      const trainingContext: StableNeoHookeanCubatureContext = {
        mesh: definition.mesh,
        restData,
        materials: definition.materials,
        lumpedMasses,
        restSystem,
        timestep: definition.settings.timestep,
        predictedPositions: definition.mesh.positions,
        sourceVertex,
        exactBasis: computeExactVertexBasis(
          definition.mesh,
          restSystem,
          sourceVertex,
        ).basis,
      };
      const selection = selectStableNeoHookeanCubatureSamples(
        trainingContext,
        corpus.training,
        definition.settings.cubatureSamples,
      );
      const packedContext = {
        ...trainingContext,
        exactBasis: Float64Array.from(
          Float32Array.from(trainingContext.exactBasis),
        ),
      };
      for (const pose of [...corpus.training, ...corpus.validation]) {
        const result = globalizeSelectedCubatureRestrictedLocal({
          context: {
            ...packedContext,
            predictedPositions: pose.predictedPositions,
          },
          positions: pose.positions,
          samples: selection.packedSamples,
        });
        if (Math.hypot(...result.model.initialSystem.gradient) <= 1e-10) {
          continue;
        }
        expect(result.direction.accepted, `${pose.id} v${sourceVertex} direction`).toBe(true);
        expect(
          result.direction.gradientDotDirection,
          `${pose.id} v${sourceVertex} descent`,
        ).toBeLessThan(0);
        expect(
          result.direction.normalizedShift,
          `${pose.id} v${sourceVertex} shift`,
        ).toBeLessThanOrEqual(JGS2_LOCAL_MAX_NORMALIZED_SHIFT);
        expect(result.lineSearch?.accepted, `${pose.id} v${sourceVertex} Armijo`).toBe(true);
        expect(
          result.lineSearch!.acceptedEnergy,
          `${pose.id} v${sourceVertex} energy`,
        ).toBeLessThanOrEqual(result.lineSearch!.armijoBound);
        expect(
          result.lineSearch!.minimumDeformationDeterminant,
          `${pose.id} v${sourceVertex} determinant`,
        ).toBeGreaterThan(PHASE1_ACCEPTED_DETERMINANT_FLOOR);
        maximumNormalizedShift = Math.max(
          maximumNormalizedShift,
          result.direction.normalizedShift,
        );
        maximumLinearResidual = Math.max(
          maximumLinearResidual,
          result.direction.linearResidual,
        );
        maximumBacktracks = Math.max(
          maximumBacktracks,
          result.lineSearch!.backtrackCount,
        );
        minimumAcceptedDeterminant = Math.min(
          minimumAcceptedDeterminant,
          result.lineSearch!.minimumDeformationDeterminant,
        );
        systemCount += 1;
      }
    }
    expect(systemCount).toBe(
      restSystem.activeVertices.length *
        (corpus.training.length + corpus.validation.length),
    );
    expect(maximumLinearResidual).toBeLessThan(1e-8);
    console.log(
      `Phase 1 CPU globalization corpus: ${systemCount} systems, ` +
        `max rho=${maximumNormalizedShift.toExponential(6)}, ` +
        `max shifted residual=${maximumLinearResidual.toExponential(6)}, ` +
        `max backtracks=${maximumBacktracks}, ` +
        `min accepted J=${minimumAcceptedDeterminant.toExponential(6)}`,
    );
  });
});
