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
  buildCurrentCoRotatedEquilibriumBasis,
  selectStableNeoHookeanCubatureSamples,
  type StableNeoHookeanCubatureContext,
} from "./nonlinear-cubature-training";
import {
  createNonlinearPerVertexObjective,
  evaluateNonlinearPerVertexObjective,
  type NonlinearPerVertexObjectiveInput,
} from "./nonlinear-objective";
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

function buildPerVertexObjective(
  context: StableNeoHookeanCubatureContext,
  positions: Float64Array,
): NonlinearPerVertexObjectiveInput {
  const vertexCount = context.mesh.positions.length / 3;
  const externalForces = new Float64Array(vertexCount * 3);
  const targetPositions = positions.slice();
  const targetStiffnesses = new Float64Array(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (context.mesh.fixed[vertex] !== 0) continue;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      externalForces[vertex * 3 + coordinate] =
        0.35 * Math.cos(vertex * 1.1 + coordinate * 0.7);
      targetPositions[vertex * 3 + coordinate] +=
        0.012 * Math.sin(vertex * 0.9 + coordinate + 0.2);
    }
    targetStiffnesses[vertex] = 18 + vertex * 2;
  }
  return { externalForces, targetPositions, targetStiffnesses };
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

  it("matches force/target derivatives and Armijo on the complete restricted scalar", () => {
    const fixture = buildRestrictedFixture();
    const objective = buildPerVertexObjective(
      fixture.context,
      fixture.pose.positions,
    );
    const model = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
      objective,
    });
    const zero = new Float64Array(3);
    expect(
      relativeError(
        model.initialSystem.gradient,
        centralDifferenceGradient(model.energy, zero),
      ),
    ).toBeLessThan(2e-6);
    expect(
      relativeError(
        model.initialSystem.hessian,
        centralDifferenceHessian(
          (local) => centralDifferenceGradient(model.energy, local),
          zero,
        ),
      ),
    ).toBeLessThan(2e-4);

    const globalization = globalizeSelectedCubatureRestrictedLocal({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
      objective,
    });
    expect(globalization.direction.accepted).toBe(true);
    expect(globalization.direction.gradientDotDirection).toBeLessThan(0);
    expect(globalization.lineSearch?.accepted).toBe(true);
    expect(globalization.lineSearch!.acceptedEnergy).toBeLessThanOrEqual(
      globalization.lineSearch!.armijoBound,
    );
    expect(
      globalization.lineSearch!.minimumDeformationDeterminant,
    ).toBeGreaterThan(PHASE1_ACCEPTED_DETERMINANT_FLOOR);
  });

  it("keeps objective deltas accurate and globalizable after a huge world translation", () => {
    const fixture = buildRestrictedFixture();
    const translation = [1e10, -2e10, 0.5e10] as const;
    const translatedPositions = Float64Array.from(
      fixture.pose.positions,
      (value, index) => value + translation[index % 3]!,
    );
    const translatedPredicted = Float64Array.from(
      fixture.context.predictedPositions,
      (value, index) => value + translation[index % 3]!,
    );
    const context = {
      ...fixture.context,
      predictedPositions: translatedPredicted,
    };
    const objective = buildPerVertexObjective(context, translatedPositions);
    const baseline = createSelectedCubatureRestrictedLocalModel({
      context,
      positions: translatedPositions,
      samples: fixture.samples,
    });
    const model = createSelectedCubatureRestrictedLocalModel({
      context,
      positions: translatedPositions,
      samples: fixture.samples,
      objective,
    });
    const displacement = new Float64Array([1.25e-4, -2e-4, 1.5e-4]);
    const baselineSystem = baseline.initialSystem;
    const objectiveSystem = model.initialSystem;
    const objectiveGradient = Float64Array.from(
      objectiveSystem.gradient,
      (value, index) => value - baselineSystem.gradient[index]!,
    );
    const objectiveHessian = Float64Array.from(
      objectiveSystem.hessian,
      (value, index) => value - baselineSystem.hessian[index]!,
    );
    let expectedDelta = 0;
    for (let row = 0; row < 3; row += 1) {
      expectedDelta += objectiveGradient[row]! * displacement[row]!;
      for (let column = 0; column < 3; column += 1) {
        expectedDelta +=
          0.5 *
          displacement[row]! *
          objectiveHessian[row * 3 + column]! *
          displacement[column]!;
      }
    }
    const actualDelta = model.energy(displacement) - baseline.energy(displacement);

    expect(model.energy(new Float64Array(3))).toBe(
      baseline.energy(new Float64Array(3)),
    );
    expect(Number.isFinite(actualDelta)).toBe(true);
    expect(
      Math.abs(actualDelta - expectedDelta) /
        Math.max(1e-12, Math.abs(expectedDelta)),
    ).toBeLessThan(1e-8);

    const globalization = globalizeSelectedCubatureRestrictedLocal({
      context,
      positions: translatedPositions,
      samples: fixture.samples,
      objective,
    });
    expect(globalization.direction.accepted).toBe(true);
    expect(globalization.lineSearch?.accepted).toBe(true);
    expect(globalization.lineSearch!.acceptedEnergy).toBeLessThanOrEqual(
      globalization.lineSearch!.armijoBound,
    );
    expect(
      globalization.lineSearch!.minimumDeformationDeterminant,
    ).toBeGreaterThan(PHASE1_ACCEPTED_DETERMINANT_FLOOR);
  });

  it("matches the dense objective projection when every tetrahedron has unit weight", () => {
    const fixture = buildRestrictedFixture();
    const { context } = fixture;
    const fullSamples: CubatureSample[] = Array.from(
      { length: context.mesh.tetrahedra.length / 4 },
      (_unused, tetrahedron) => {
        const basisBlocks = new Float64Array(36);
        for (let localVertex = 0; localVertex < 4; localVertex += 1) {
          const vertex =
            context.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
          basisBlocks.set(
            context.exactBasis.subarray(vertex * 9, vertex * 9 + 9),
            localVertex * 9,
          );
        }
        return { tetrahedron, weight: 1, basisBlocks };
      },
    );
    const objectiveInput = buildPerVertexObjective(
      context,
      fixture.pose.positions,
    );
    const objective = createNonlinearPerVertexObjective(
      context.mesh,
      objectiveInput,
    );
    const baseline = createSelectedCubatureRestrictedLocalModel({
      context,
      positions: fixture.pose.positions,
      samples: fullSamples,
    }).initialSystem;
    const actual = createSelectedCubatureRestrictedLocalModel({
      context,
      positions: fixture.pose.positions,
      samples: fullSamples,
      objective: objectiveInput,
    }).initialSystem;
    const actualGradient = Float64Array.from(
      actual.gradient,
      (value, index) => value - baseline.gradient[index]!,
    );
    const actualHessian = Float64Array.from(
      actual.hessian,
      (value, index) => value - baseline.hessian[index]!,
    );
    const expectedGradient = new Float64Array(3);
    const expectedHessian = new Float64Array(9);
    const currentBasis = buildCurrentCoRotatedEquilibriumBasis(
      context,
      fixture.pose.positions,
    );
    for (let vertex = 0; vertex < objective.vertexCount; vertex += 1) {
      const evaluation = evaluateNonlinearPerVertexObjective(
        objective,
        vertex,
        fixture.pose.positions.subarray(vertex * 3, vertex * 3 + 3),
      );
      for (let reducedRow = 0; reducedRow < 3; reducedRow += 1) {
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          expectedGradient[reducedRow] +=
            currentBasis[vertex * 9 + coordinate * 3 + reducedRow]! *
            evaluation.gradient[coordinate]!;
        }
        for (let reducedColumn = 0; reducedColumn < 3; reducedColumn += 1) {
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            expectedHessian[reducedRow * 3 + reducedColumn] +=
              evaluation.targetStiffness *
              currentBasis[vertex * 9 + coordinate * 3 + reducedRow]! *
              currentBasis[vertex * 9 + coordinate * 3 + reducedColumn]!;
          }
        }
      }
    }

    expect(relativeError(actualGradient, expectedGradient)).toBeLessThan(1e-10);
    expect(relativeError(actualHessian, expectedHessian)).toBeLessThan(1e-10);
  });

  it("makes a released target exactly identical to an absent objective", () => {
    const fixture = buildRestrictedFixture();
    const vertexCount = fixture.context.mesh.positions.length / 3;
    const released: NonlinearPerVertexObjectiveInput = {
      externalForces: new Float64Array(vertexCount * 3),
      targetPositions: new Float64Array(vertexCount * 3).fill(
        Number.MAX_VALUE,
      ),
      targetStiffnesses: new Float64Array(vertexCount),
    };
    const baseline = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
    });
    const releasedModel = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions: fixture.pose.positions,
      samples: fixture.samples,
      objective: released,
    });
    const displacement = new Float64Array([2e-4, -1e-4, 3e-4]);

    expect(releasedModel.energy(displacement)).toBe(
      baseline.energy(displacement),
    );
    expect(releasedModel.initialSystem.gradient).toEqual(
      baseline.initialSystem.gradient,
    );
    expect(releasedModel.initialSystem.hessian).toEqual(
      baseline.initialSystem.hessian,
    );
    expect(releasedModel.initialSystem.newtonUpdate).toEqual(
      baseline.initialSystem.newtonUpdate,
    );
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
    const objective = buildPerVertexObjective(
      fixture.context,
      fixture.pose.positions,
    );
    const model = createSelectedCubatureRestrictedLocalModel({
      context: fixture.context,
      positions,
      samples,
      objective,
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
    objective.externalForces.fill(-987);
    objective.targetPositions.fill(-987);
    objective.targetStiffnesses.fill(987);
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

  it("keeps representative composite-objective updates within two-percent RMS", () => {
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
    const inverseTimestepSquared =
      1 / (definition.settings.timestep * definition.settings.timestep);
    let squaredError = 0;
    let squaredReference = 0;
    let systemCount = 0;

    for (const sourceVertex of restSystem.activeVertices) {
      const exactBasis = computeExactVertexBasis(
        definition.mesh,
        restSystem,
        sourceVertex,
      ).basis;
      const trainingContext: StableNeoHookeanCubatureContext = {
        mesh: definition.mesh,
        restData,
        materials: definition.materials,
        lumpedMasses,
        restSystem,
        timestep: definition.settings.timestep,
        predictedPositions: definition.mesh.positions,
        sourceVertex,
        exactBasis,
      };
      const selection = selectStableNeoHookeanCubatureSamples(
        trainingContext,
        corpus.training,
        definition.settings.cubatureSamples,
      );
      const packedExactBasis = Float64Array.from(Float32Array.from(exactBasis));
      const packedContext = {
        ...trainingContext,
        exactBasis: packedExactBasis,
      };
      const fullSamples: CubatureSample[] = Array.from(
        { length: definition.mesh.tetrahedra.length / 4 },
        (_unused, tetrahedron) => {
          const basisBlocks = new Float64Array(36);
          for (let localVertex = 0; localVertex < 4; localVertex += 1) {
            const vertex =
              definition.mesh.tetrahedra[tetrahedron * 4 + localVertex]!;
            basisBlocks.set(
              packedExactBasis.subarray(vertex * 9, vertex * 9 + 9),
              localVertex * 9,
            );
          }
          return { tetrahedron, weight: 1, basisBlocks };
        },
      );

      for (const pose of [...corpus.training, ...corpus.validation]) {
        const vertexCount = definition.mesh.positions.length / 3;
        const externalForces = new Float64Array(vertexCount * 3);
        const targetPositions = pose.positions.slice();
        const targetStiffnesses = new Float64Array(vertexCount);
        for (const vertex of restSystem.activeVertices) {
          const inertialScale =
            lumpedMasses[vertex]! * inverseTimestepSquared;
          targetStiffnesses[vertex] = 0.35 * inertialScale;
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            const phase =
              pose.index * 0.37 + vertex * 0.71 + coordinate * 1.13;
            externalForces[vertex * 3 + coordinate] =
              0.008 * inertialScale * Math.cos(phase);
            targetPositions[vertex * 3 + coordinate] +=
              0.03 * Math.sin(phase + 0.23);
          }
        }
        const objective = {
          externalForces,
          targetPositions,
          targetStiffnesses,
        };
        const context = {
          ...packedContext,
          predictedPositions: pose.predictedPositions,
        };
        const selectedUpdate = createSelectedCubatureRestrictedLocalModel({
          context,
          positions: pose.positions,
          samples: selection.packedSamples,
          objective,
        }).initialSystem.newtonUpdate;
        const referenceUpdate = createSelectedCubatureRestrictedLocalModel({
          context,
          positions: pose.positions,
          samples: fullSamples,
          objective,
        }).initialSystem.newtonUpdate;
        expect(selectedUpdate).toBeDefined();
        expect(referenceUpdate).toBeDefined();
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          const reference = referenceUpdate![coordinate]!;
          const error = selectedUpdate![coordinate]! - reference;
          squaredError += error * error;
          squaredReference += reference * reference;
        }
        systemCount += 1;
      }
    }

    const updateRms = Math.sqrt(squaredError / squaredReference);
    expect(systemCount).toBe(240);
    expect(squaredReference).toBeGreaterThan(0);
    expect(updateRms).toBeLessThanOrEqual(0.02);
    console.log(
      `Phase 1 composite-objective selected update RMS: ` +
        `${updateRms.toExponential(6)} over ${systemCount} systems`,
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
