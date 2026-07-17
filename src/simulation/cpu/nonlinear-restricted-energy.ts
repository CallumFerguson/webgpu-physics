import {
  assembleStableNeoHookeanJGS2LocalSystem,
  buildCurrentCoRotatedCubatureBasisBlocks,
  type StableNeoHookeanCubatureContext,
  type StableNeoHookeanJGS2LocalSystem,
} from "./nonlinear-cubature-training";
import {
  JGS2_LOCAL_MAX_NORMALIZED_SHIFT,
  PHASE1_ACCEPTED_DETERMINANT_FLOOR,
  computeJGS2LocalDescentDirection,
  lineSearchRestrictedJGS2LocalDirection,
  type JGS2LocalDirectionResult,
  type JGS2RestrictedLocalLineSearchResult,
} from "./nonlinear-globalization";
import { getTetrahedronCount, getVertexCount } from "./mesh";
import {
  evaluateStableNeoHookeanTetrahedron,
  minimumTetrahedralDeformationDeterminant,
  tetrahedralDeformationDeterminant,
} from "./stable-neo-hookean";
import { solveDenseLinearSystem } from "./math";
import {
  createNonlinearPerVertexObjective,
  evaluateNonlinearPerVertexObjective,
  nonlinearPerVertexObjectiveEnergyDelta,
  type NonlinearPerVertexObjective,
  type NonlinearPerVertexObjectiveInput,
} from "./nonlinear-objective";
import type {
  CubatureSample,
  LinearMaterial,
  RestLinearSystem,
  RestTetraData,
  TetrahedralMesh,
} from "./types";

interface FrozenRestrictedCubatureSample {
  readonly tetrahedron: number;
  readonly weight: number;
  /** Four current co-rotated row-major 3x3 basis blocks. */
  readonly currentBasisBlocks: Float64Array;
}

export interface SelectedCubatureRestrictedLocalOptions {
  readonly context: StableNeoHookeanCubatureContext;
  /** Feasible accepted pose about which q is defined. */
  readonly positions: Float64Array;
  /** Runtime-packed selected records, including their stored Ubar blocks. */
  readonly samples: readonly CubatureSample[];
  /** Frame-frozen force/target terms; excluded from Cubature training. */
  readonly objective?: NonlinearPerVertexObjectiveInput;
}

export interface SelectedCubatureRestrictedLocalModel {
  readonly sourceVertex: number;
  readonly basePositions: Float64Array;
  readonly initialSystem: StableNeoHookeanJGS2LocalSystem;
  readonly frozenSamples: readonly FrozenRestrictedCubatureSample[];
  readonly sourceIncidentTetrahedra: Uint32Array;
  readonly sourceInertiaScale: number;
  readonly baseMinimumDeformationDeterminant: number;
  /** Source-exact plus weighted complementary-Cubature scalar. */
  energy(localDisplacement: Float64Array): number;
  /** Geometry-only feasibility check; evaluates no material energy. */
  minimumDeformationDeterminant(localDisplacement: Float64Array): number;
}

export interface SelectedCubatureRestrictedGlobalizationResult {
  readonly model: SelectedCubatureRestrictedLocalModel;
  readonly direction: JGS2LocalDirectionResult;
  readonly lineSearch: JGS2RestrictedLocalLineSearchResult | undefined;
}

function validateLocalDisplacement(values: Float64Array): void {
  if (values.length !== 3) {
    throw new RangeError("Restricted local displacement must contain three values.");
  }
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new RangeError("Restricted local displacement must be finite.");
    }
  }
}

function allFinite(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) return false;
  }
  return true;
}

function buildIncidentCounts(context: StableNeoHookeanCubatureContext): Uint32Array {
  const counts = new Uint32Array(getVertexCount(context.mesh));
  for (const vertex of context.mesh.tetrahedra) {
    counts[vertex] += 1;
  }
  if (counts[context.sourceVertex] === 0) {
    throw new Error("The restricted source vertex has no incident tetrahedron.");
  }
  return counts;
}

function sourceIncidentTetrahedra(
  context: StableNeoHookeanCubatureContext,
): Uint32Array {
  const result: number[] = [];
  for (
    let tetrahedron = 0;
    tetrahedron < getTetrahedronCount(context.mesh);
    tetrahedron += 1
  ) {
    const start = tetrahedron * 4;
    if (
      context.mesh.tetrahedra[start] === context.sourceVertex ||
      context.mesh.tetrahedra[start + 1] === context.sourceVertex ||
      context.mesh.tetrahedra[start + 2] === context.sourceVertex ||
      context.mesh.tetrahedra[start + 3] === context.sourceVertex
    ) {
      result.push(tetrahedron);
    }
  }
  return Uint32Array.from(result);
}

function sourceTrialPositions(
  basePositions: Float64Array,
  sourceVertex: number,
  localDisplacement: Float64Array,
): Float64Array {
  const positions = basePositions.slice();
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    positions[sourceVertex * 3 + coordinate] += localDisplacement[coordinate]!;
  }
  return positions;
}

function projectedSamplePositions(
  context: StableNeoHookeanCubatureContext,
  basePositions: Float64Array,
  sample: FrozenRestrictedCubatureSample,
  localDisplacement: Float64Array,
): Float64Array {
  const positions = basePositions.slice();
  for (let localVertex = 0; localVertex < 4; localVertex += 1) {
    const vertex = context.mesh.tetrahedra[sample.tetrahedron * 4 + localVertex]!;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      let displacement = 0;
      for (let reduced = 0; reduced < 3; reduced += 1) {
        displacement +=
          sample.currentBasisBlocks[
            localVertex * 9 + coordinate * 3 + reduced
          ]! * localDisplacement[reduced]!;
      }
      positions[vertex * 3 + coordinate] += displacement;
    }
  }
  return positions;
}

function quadraticInertiaEnergy(
  positions: Float64Array,
  predictedPositions: Float64Array,
  vertex: number,
  inertia: number,
): number {
  let squared = 0;
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    const difference =
      positions[vertex * 3 + coordinate]! -
      predictedPositions[vertex * 3 + coordinate]!;
    squared += difference * difference;
  }
  return 0.5 * inertia * squared;
}

interface RestrictedObjectiveSystem {
  readonly sourceGradient: Float64Array;
  readonly sourceHessian: Float64Array;
  readonly remainderGradient: Float64Array;
  readonly remainderHessian: Float64Array;
}

function addScaledArray(
  destination: Float64Array,
  source: ArrayLike<number>,
  scale: number,
): void {
  for (let index = 0; index < destination.length; index += 1) {
    destination[index] += scale * source[index]!;
  }
}

function evaluateObjectiveAtVertex(
  objective: NonlinearPerVertexObjective,
  positions: Float64Array,
  vertex: number,
) {
  return evaluateNonlinearPerVertexObjective(
    objective,
    vertex,
    positions.subarray(vertex * 3, vertex * 3 + 3),
  );
}

function objectiveEnergyDeltaAtVertex(
  objective: NonlinearPerVertexObjective,
  basePositions: Float64Array,
  vertex: number,
  displacement: Float64Array,
): number {
  const base = vertex * 3;
  return nonlinearPerVertexObjectiveEnergyDelta(
    objective,
    vertex,
    basePositions.subarray(base, base + 3),
    displacement,
  );
}

function projectedVertexDisplacement(
  sample: FrozenRestrictedCubatureSample,
  localVertex: number,
  localDisplacement: Float64Array,
): Float64Array {
  const result = new Float64Array(3);
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    for (let reduced = 0; reduced < 3; reduced += 1) {
      result[coordinate] +=
        sample.currentBasisBlocks[
          localVertex * 9 + coordinate * 3 + reduced
        ]! * localDisplacement[reduced]!;
    }
  }
  return result;
}

function accumulateProjectedObjective(
  gradient: Float64Array,
  hessian: Float64Array,
  evaluation: ReturnType<typeof evaluateNonlinearPerVertexObjective>,
  basisBlocks: ArrayLike<number>,
  basisOffset: number,
  scale: number,
): void {
  for (let reducedRow = 0; reducedRow < 3; reducedRow += 1) {
    let projectedGradient = 0;
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      projectedGradient +=
        basisBlocks[basisOffset + coordinate * 3 + reducedRow]! *
        evaluation.gradient[coordinate]!;
    }
    gradient[reducedRow] += scale * projectedGradient;
    for (let reducedColumn = 0; reducedColumn < 3; reducedColumn += 1) {
      let projectedHessian = 0;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        projectedHessian +=
          basisBlocks[basisOffset + coordinate * 3 + reducedRow]! *
          basisBlocks[basisOffset + coordinate * 3 + reducedColumn]!;
      }
      hessian[reducedRow * 3 + reducedColumn] +=
        scale * evaluation.targetStiffness * projectedHessian;
    }
  }
}

/** Assemble the same source-exact/source-subtracted objective as the scalar. */
function assembleRestrictedObjectiveSystem(
  context: StableNeoHookeanCubatureContext,
  basePositions: Float64Array,
  samples: readonly FrozenRestrictedCubatureSample[],
  incidentCounts: Uint32Array,
  objective: NonlinearPerVertexObjective,
): RestrictedObjectiveSystem {
  const sourceGradient = new Float64Array(3);
  const sourceHessian = new Float64Array(9);
  const sourceEvaluation = evaluateObjectiveAtVertex(
    objective,
    basePositions,
    context.sourceVertex,
  );
  sourceGradient.set(sourceEvaluation.gradient);
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    sourceHessian[coordinate * 3 + coordinate] =
      sourceEvaluation.targetStiffness;
  }

  const remainderGradient = new Float64Array(3);
  const remainderHessian = new Float64Array(9);
  for (const sample of samples) {
    const sampleGradient = new Float64Array(3);
    const sampleHessian = new Float64Array(9);
    for (let localVertex = 0; localVertex < 4; localVertex += 1) {
      const vertex =
        context.mesh.tetrahedra[sample.tetrahedron * 4 + localVertex]!;
      const incidentShare = 1 / incidentCounts[vertex]!;
      const evaluation = evaluateObjectiveAtVertex(
        objective,
        basePositions,
        vertex,
      );
      accumulateProjectedObjective(
        sampleGradient,
        sampleHessian,
        evaluation,
        sample.currentBasisBlocks,
        localVertex * 9,
        incidentShare,
      );
      if (vertex === context.sourceVertex) {
        addScaledArray(
          sampleGradient,
          sourceEvaluation.gradient,
          -incidentShare,
        );
        for (let coordinate = 0; coordinate < 3; coordinate += 1) {
          sampleHessian[coordinate * 3 + coordinate] -=
            incidentShare * sourceEvaluation.targetStiffness;
        }
      }
    }
    addScaledArray(remainderGradient, sampleGradient, sample.weight);
    addScaledArray(remainderHessian, sampleHessian, sample.weight);
  }

  for (const values of [
    sourceGradient,
    sourceHessian,
    remainderGradient,
    remainderHessian,
  ]) {
    if (!allFinite(values)) {
      throw new RangeError(
        "Restricted per-vertex objective derivatives must be finite.",
      );
    }
  }
  return {
    sourceGradient,
    sourceHessian,
    remainderGradient,
    remainderHessian,
  };
}

function recomposeLocalSystem(
  base: StableNeoHookeanJGS2LocalSystem,
  objective: RestrictedObjectiveSystem | undefined,
): StableNeoHookeanJGS2LocalSystem {
  if (!objective) return base;
  const sourceGradient = base.sourceGradient.slice();
  const sourceHessian = base.sourceHessian.slice();
  const remainderGradient = base.remainderGradient.slice();
  const remainderHessian = base.remainderHessian.slice();
  addScaledArray(sourceGradient, objective.sourceGradient, 1);
  addScaledArray(sourceHessian, objective.sourceHessian, 1);
  addScaledArray(remainderGradient, objective.remainderGradient, 1);
  addScaledArray(remainderHessian, objective.remainderHessian, 1);
  const gradient = sourceGradient.slice();
  const hessian = sourceHessian.slice();
  addScaledArray(gradient, remainderGradient, 1);
  addScaledArray(hessian, remainderHessian, 1);
  return {
    sourceGradient,
    sourceHessian,
    remainderGradient,
    remainderHessian,
    gradient,
    hessian,
    newtonUpdate: solveDenseLinearSystem(
      hessian,
      Float64Array.from(gradient, (value) => -value),
      3,
    ),
    candidates: base.candidates,
    minimumDeformationDeterminant: base.minimumDeformationDeterminant,
  };
}

function snapshotMesh(mesh: TetrahedralMesh): TetrahedralMesh {
  return {
    positions: mesh.positions.slice(),
    tetrahedra: mesh.tetrahedra.slice(),
    materialIds: mesh.materialIds.slice(),
    fixed: mesh.fixed.slice(),
    bodyIds: mesh.bodyIds.slice(),
  };
}

function snapshotRestData(restData: RestTetraData): RestTetraData {
  return {
    volumes: restData.volumes.slice(),
    inverseRestMatrices: restData.inverseRestMatrices.slice(),
    stiffnessMatrices: restData.stiffnessMatrices.slice(),
  };
}

function snapshotRestSystem(restSystem: RestLinearSystem): RestLinearSystem {
  return {
    dimension: restSystem.dimension,
    activeVertices: restSystem.activeVertices.slice(),
    vertexToActiveDof: restSystem.vertexToActiveDof.slice(),
    hessian: restSystem.hessian.slice(),
    choleskyLower: restSystem.choleskyLower.slice(),
  };
}

function snapshotMaterials(
  materials: readonly LinearMaterial[],
): readonly LinearMaterial[] {
  return Object.freeze(
    materials.map((material) =>
      Object.freeze({
        ...material,
        color: [
          material.color[0],
          material.color[1],
          material.color[2],
          material.color[3],
        ] as const,
      }),
    ),
  );
}

function snapshotContext(
  context: StableNeoHookeanCubatureContext,
): StableNeoHookeanCubatureContext {
  return {
    mesh: snapshotMesh(context.mesh),
    restData: snapshotRestData(context.restData),
    materials: snapshotMaterials(context.materials),
    lumpedMasses: context.lumpedMasses.slice(),
    restSystem: snapshotRestSystem(context.restSystem),
    timestep: context.timestep,
    predictedPositions: context.predictedPositions.slice(),
    sourceVertex: context.sourceVertex,
    exactBasis: context.exactBasis.slice(),
  };
}

function cloneLocalSystem(
  system: StableNeoHookeanJGS2LocalSystem,
): StableNeoHookeanJGS2LocalSystem {
  return {
    sourceGradient: system.sourceGradient.slice(),
    sourceHessian: system.sourceHessian.slice(),
    remainderGradient: system.remainderGradient.slice(),
    remainderHessian: system.remainderHessian.slice(),
    gradient: system.gradient.slice(),
    hessian: system.hessian.slice(),
    newtonUpdate: system.newtonUpdate?.slice(),
    candidates: Object.freeze(
      system.candidates.map((candidate) =>
        Object.freeze({
          tetrahedron: candidate.tetrahedron,
          currentBasisBlocks: candidate.currentBasisBlocks.slice(),
          projectedGradient: candidate.projectedGradient.slice(),
          projectedHessian: candidate.projectedHessian.slice(),
          remainderGradient: candidate.remainderGradient.slice(),
          remainderHessian: candidate.remainderHessian.slice(),
          sourceGradient: candidate.sourceGradient.slice(),
          sourceHessian: candidate.sourceHessian.slice(),
          deformationDeterminant: candidate.deformationDeterminant,
        }),
      ),
    ),
    minimumDeformationDeterminant: system.minimumDeformationDeterminant,
  };
}

function cloneFrozenSamples(
  samples: readonly FrozenRestrictedCubatureSample[],
): readonly FrozenRestrictedCubatureSample[] {
  return Object.freeze(
    samples.map((sample) =>
      Object.freeze({
        tetrahedron: sample.tetrahedron,
        weight: sample.weight,
        currentBasisBlocks: sample.currentBasisBlocks.slice(),
      }),
    ),
  );
}

/**
 * Freeze the exact-source/selected-remainder objective used by one nonlinear
 * JGS2 local solve. Basis rotations are frozen at the iteration source pose.
 */
export function createSelectedCubatureRestrictedLocalModel(
  options: SelectedCubatureRestrictedLocalOptions,
): SelectedCubatureRestrictedLocalModel {
  const context = snapshotContext(options.context);
  const vertexCount = getVertexCount(context.mesh);
  if (options.positions.length !== vertexCount * 3) {
    throw new RangeError("Restricted local positions do not match the mesh.");
  }
  for (const value of options.positions) {
    if (!Number.isFinite(value)) {
      throw new RangeError("Restricted local positions must be finite.");
    }
  }
  const basePositions = options.positions.slice();
  const objective = options.objective
    ? createNonlinearPerVertexObjective(context.mesh, options.objective)
    : undefined;
  const baseMinimumDeformationDeterminant =
    minimumTetrahedralDeformationDeterminant(
      context.mesh,
      context.restData,
      basePositions,
    );
  if (
    !Number.isFinite(baseMinimumDeformationDeterminant) ||
    !(baseMinimumDeformationDeterminant >
      PHASE1_ACCEPTED_DETERMINANT_FLOOR)
  ) {
    throw new RangeError(
      `Restricted local source pose has minimum J ` +
        `${baseMinimumDeformationDeterminant}; required greater than ` +
        `${PHASE1_ACCEPTED_DETERMINANT_FLOOR}.`,
    );
  }

  const used = new Set<number>();
  const restSamples: CubatureSample[] = [];
  for (const sample of options.samples) {
    if (!(sample.weight >= 0) || !Number.isFinite(sample.weight)) {
      throw new RangeError(
        "Restricted Cubature weights must be finite and nonnegative.",
      );
    }
    // Match the GPU ABI: zero-weight records are empty padding slots. Ignore
    // them before validating the sentinel tetrahedron or absent basis payload.
    if (!(sample.weight > 0)) continue;
    if (
      !Number.isSafeInteger(sample.tetrahedron) ||
      sample.tetrahedron < 0 ||
      sample.tetrahedron >= getTetrahedronCount(context.mesh) ||
      used.has(sample.tetrahedron)
    ) {
      throw new RangeError(
        "Restricted Cubature tetrahedra must be unique and in range.",
      );
    }
    if (sample.basisBlocks.length !== 36) {
      throw new RangeError(
        "Restricted Cubature records require four 3x3 basis blocks.",
      );
    }
    used.add(sample.tetrahedron);
    restSamples.push({
      tetrahedron: sample.tetrahedron,
      weight: sample.weight,
      basisBlocks: sample.basisBlocks.slice(),
    });
  }
  const frozenSamples = restSamples.map((sample) =>
    Object.freeze({
      tetrahedron: sample.tetrahedron,
      weight: sample.weight,
      currentBasisBlocks: buildCurrentCoRotatedCubatureBasisBlocks(
        context,
        basePositions,
        sample.tetrahedron,
        sample.basisBlocks,
      ),
    }),
  );
  const incidentCounts = buildIncidentCounts(context);
  const sourceIncident = sourceIncidentTetrahedra(context);
  const inverseTimestepSquared = 1 / (context.timestep * context.timestep);
  const sourceInertiaScale =
    context.lumpedMasses[context.sourceVertex]! * inverseTimestepSquared;
  if (!(sourceInertiaScale > 0) || !Number.isFinite(sourceInertiaScale)) {
    throw new RangeError("Restricted local source inertia must be positive.");
  }
  const materialInertiaSystem = assembleStableNeoHookeanJGS2LocalSystem(
    context,
    basePositions,
    restSamples,
  );
  const initialSystem = recomposeLocalSystem(
    materialInertiaSystem,
    objective?.active
      ? assembleRestrictedObjectiveSystem(
          context,
          basePositions,
          frozenSamples,
          incidentCounts,
          objective,
        )
      : undefined,
  );

  const model: SelectedCubatureRestrictedLocalModel = {
    sourceVertex: context.sourceVertex,
    get basePositions() {
      return basePositions.slice();
    },
    get initialSystem() {
      return cloneLocalSystem(initialSystem);
    },
    get frozenSamples() {
      return cloneFrozenSamples(frozenSamples);
    },
    get sourceIncidentTetrahedra() {
      return sourceIncident.slice();
    },
    sourceInertiaScale,
    baseMinimumDeformationDeterminant,
    minimumDeformationDeterminant(localDisplacement) {
      validateLocalDisplacement(localDisplacement);
      const sourcePositions = sourceTrialPositions(
        basePositions,
        context.sourceVertex,
        localDisplacement,
      );
      if (!allFinite(sourcePositions)) return Number.NaN;
      let minimum = baseMinimumDeformationDeterminant;
      for (const tetrahedron of sourceIncident) {
        const determinant = tetrahedralDeformationDeterminant(
          context.mesh,
          context.restData,
          tetrahedron,
          sourcePositions,
        );
        if (!Number.isFinite(determinant)) return determinant;
        minimum = Math.min(minimum, determinant);
      }
      for (const sample of frozenSamples) {
        const projected = projectedSamplePositions(
          context,
          basePositions,
          sample,
          localDisplacement,
        );
        if (!allFinite(projected)) return Number.NaN;
        const determinant = tetrahedralDeformationDeterminant(
          context.mesh,
          context.restData,
          sample.tetrahedron,
          projected,
        );
        if (!Number.isFinite(determinant)) return determinant;
        minimum = Math.min(minimum, determinant);
      }
      return minimum;
    },
    energy(localDisplacement) {
      validateLocalDisplacement(localDisplacement);
      const sourcePositions = sourceTrialPositions(
        basePositions,
        context.sourceVertex,
        localDisplacement,
      );
      let energy = quadraticInertiaEnergy(
        sourcePositions,
        context.predictedPositions,
        context.sourceVertex,
        sourceInertiaScale,
      );
      const sourceObjectiveEnergyDelta = objective?.active
        ? nonlinearPerVertexObjectiveEnergyDelta(
            objective,
            context.sourceVertex,
            basePositions.subarray(
              context.sourceVertex * 3,
              context.sourceVertex * 3 + 3,
            ),
            localDisplacement,
          )
        : 0;
      energy += sourceObjectiveEnergyDelta;
      const sourceElementEnergy = new Map<number, number>();
      for (const tetrahedron of sourceIncident) {
        const elementEnergy = evaluateStableNeoHookeanTetrahedron(
          context.mesh,
          context.restData,
          context.materials,
          tetrahedron,
          sourcePositions,
        ).energy;
        sourceElementEnergy.set(tetrahedron, elementEnergy);
        energy += elementEnergy;
      }

      for (const sample of frozenSamples) {
        if (!(sample.weight > 0)) continue;
        const projected = projectedSamplePositions(
          context,
          basePositions,
          sample,
          localDisplacement,
        );
        let complementaryEnergy = evaluateStableNeoHookeanTetrahedron(
          context.mesh,
          context.restData,
          context.materials,
          sample.tetrahedron,
          projected,
        ).energy;
        let sourceIsIncident = false;
        for (let localVertex = 0; localVertex < 4; localVertex += 1) {
          const vertex =
            context.mesh.tetrahedra[sample.tetrahedron * 4 + localVertex]!;
          const distributedInertia =
            context.restSystem.vertexToActiveDof[vertex]! < 0
              ? 0
              : (context.lumpedMasses[vertex]! * inverseTimestepSquared) /
                incidentCounts[vertex]!;
          complementaryEnergy += quadraticInertiaEnergy(
            projected,
            context.predictedPositions,
            vertex,
            distributedInertia,
          );
          if (objective?.active) {
            complementaryEnergy +=
              objectiveEnergyDeltaAtVertex(
                objective,
                basePositions,
                vertex,
                projectedVertexDisplacement(
                  sample,
                  localVertex,
                  localDisplacement,
                ),
              ) /
              incidentCounts[vertex]!;
          }
          if (vertex === context.sourceVertex) {
            sourceIsIncident = true;
            complementaryEnergy -= quadraticInertiaEnergy(
              sourcePositions,
              context.predictedPositions,
              vertex,
              distributedInertia,
            );
            if (objective?.active) {
              complementaryEnergy -=
                sourceObjectiveEnergyDelta / incidentCounts[vertex]!;
            }
          }
        }
        if (sourceIsIncident) {
          complementaryEnergy -= sourceElementEnergy.get(sample.tetrahedron)!;
        }
        energy += sample.weight * complementaryEnergy;
      }
      return energy;
    },
  };
  return Object.freeze(model);
}

/** Apply the frozen shift and Armijo policy to the actual selected objective. */
export function globalizeSelectedCubatureRestrictedLocal(
  options: SelectedCubatureRestrictedLocalOptions,
): SelectedCubatureRestrictedGlobalizationResult {
  const model = createSelectedCubatureRestrictedLocalModel(options);
  const initialSystem = model.initialSystem;
  const direction = computeJGS2LocalDescentDirection(
    initialSystem.hessian,
    initialSystem.gradient,
    { inertiaScale: model.sourceInertiaScale },
  );
  if (!direction.accepted) {
    return { model, direction, lineSearch: undefined };
  }
  const lineSearch = lineSearchRestrictedJGS2LocalDirection({
    initialEnergy: model.energy(new Float64Array(3)),
    gradient: initialSystem.gradient,
    direction: direction.direction,
    minimumDeformationDeterminant: (alpha) =>
      model.minimumDeformationDeterminant(
        Float64Array.from(direction.direction, (value) => alpha * value),
      ),
    energy: (alpha) =>
      model.energy(
        Float64Array.from(direction.direction, (value) => alpha * value),
      ),
  });
  if (
    direction.normalizedShift > JGS2_LOCAL_MAX_NORMALIZED_SHIFT ||
    (lineSearch.accepted &&
      !(lineSearch.minimumDeformationDeterminant >
        PHASE1_ACCEPTED_DETERMINANT_FLOOR))
  ) {
    throw new Error("Restricted globalization violated its frozen safety policy.");
  }
  return { model, direction, lineSearch };
}
