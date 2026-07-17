import { describe, expect, it } from "vitest";

import { computeRestTetraData } from "./fem";
import {
  JGS2_TINY_REFERENCE_RESIDUAL_TOLERANCE,
  acceptOrRevertFeasibleJacobiIteration,
  evaluateJGS2Convergence,
} from "./nonlinear-iteration";
import { minimumTetrahedralDeformationDeterminant } from "./stable-neo-hookean";
import type { LinearMaterial, TetrahedralMesh } from "./types";

const MATERIAL: LinearMaterial = {
  name: "iteration-feasibility-solid",
  model: "stable-neo-hookean",
  density: 1_000,
  youngModulus: 10_000,
  poissonRatio: 0.3,
  color: [1, 1, 1, 1],
};

const MESH: TetrahedralMesh = {
  positions: new Float64Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]),
  tetrahedra: new Uint32Array([0, 1, 2, 3]),
  materialIds: new Uint16Array([0]),
  fixed: new Uint8Array(4),
  bodyIds: new Uint16Array(4),
};

const MULTI_TET_MESH: TetrahedralMesh = {
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

describe("Phase 1 assembled feasibility and convergence reference", () => {
  it("reverts the entire Jacobi pose when individually feasible updates invert together", () => {
    const restData = computeRestTetraData(MESH, [MATERIAL]);
    const moveB = MESH.positions.slice();
    moveB[4] += 1.1;
    const moveC = MESH.positions.slice();
    moveC[6] += 1.1;
    expect(
      acceptOrRevertFeasibleJacobiIteration(
        MESH,
        restData,
        MESH.positions,
        moveB,
      ).accepted,
    ).toBe(true);
    expect(
      acceptOrRevertFeasibleJacobiIteration(
        MESH,
        restData,
        MESH.positions,
        moveC,
      ).accepted,
    ).toBe(true);

    const assembled = MESH.positions.slice();
    assembled[4] += 1.1;
    assembled[6] += 1.1;
    const result = acceptOrRevertFeasibleJacobiIteration(
      MESH,
      restData,
      MESH.positions,
      assembled,
      { source: 2, candidate: 1 },
    );
    expect(result.accepted).toBe(false);
    expect(result.reverted).toBe(true);
    expect(result.revertCount).toBe(1);
    expect(result.candidateMinimumDeformationDeterminant).toBeLessThan(0);
    expect(result.acceptedMinimumDeformationDeterminant).toBe(1);
    expect(result.energyValid).toBe(true);
    expect(result.sourceEnergy).toBe(2);
    expect(result.candidateEnergy).toBe(1);
    expect(result.acceptedEnergy).toBe(2);
    expect(result.positions).toEqual(MESH.positions);
  });

  it("uses finite sentinels when assembled determinant arithmetic overflows", () => {
    const restData = computeRestTetraData(MESH, [MATERIAL]);
    const candidate = Float64Array.from([
      0, 0, 0,
      Number.MAX_VALUE, 0, 0,
      0, Number.MAX_VALUE, 0,
      0, 0, Number.MAX_VALUE,
    ]);
    const result = acceptOrRevertFeasibleJacobiIteration(
      MESH,
      restData,
      MESH.positions,
      candidate,
    );
    expect(result.accepted).toBe(false);
    expect(result.candidateMinimumDeformationDeterminantValid).toBe(false);
    expect(result.candidateMinimumDeformationDeterminant).toBe(0);
    expect(Number.isFinite(result.acceptedMinimumDeformationDeterminant)).toBe(
      true,
    );
    expect(result.energyValid).toBe(false);
    expect(result.sourceEnergy).toBe(0);
    expect(result.candidateEnergy).toBe(0);
    expect(result.acceptedEnergy).toBe(0);
  });

  it("rejects a multi-tet pose when positive determinant overflow follows a finite tet", () => {
    const restData = computeRestTetraData(MULTI_TET_MESH, [MATERIAL]);
    const candidate = MULTI_TET_MESH.positions.slice();
    candidate.set(
      [
        3, 0, 0,
        Number.MAX_VALUE, 0, 0,
        3, Number.MAX_VALUE, 0,
        3, 0, Number.MAX_VALUE,
      ],
      12,
    );
    expect(
      minimumTetrahedralDeformationDeterminant(
        MULTI_TET_MESH,
        restData,
        candidate,
      ),
    ).toBe(Number.POSITIVE_INFINITY);
    const result = acceptOrRevertFeasibleJacobiIteration(
      MULTI_TET_MESH,
      restData,
      MULTI_TET_MESH.positions,
      candidate,
    );
    expect(result.accepted).toBe(false);
    expect(result.reverted).toBe(true);
    expect(result.candidateMinimumDeformationDeterminantValid).toBe(false);
    expect(result.candidateMinimumDeformationDeterminant).toBe(0);
    expect(result.positions).toEqual(MULTI_TET_MESH.positions);
  });

  it("records assembled energy without using it as a feasibility gate", () => {
    const restData = computeRestTetraData(MESH, [MATERIAL]);
    const candidate = MESH.positions.slice();
    candidate[11] = 1.1;
    const accepted = acceptOrRevertFeasibleJacobiIteration(
      MESH,
      restData,
      MESH.positions,
      candidate,
      { source: 2, candidate: 3 },
    );
    expect(accepted.accepted).toBe(true);
    expect(accepted.energyValid).toBe(true);
    expect(accepted.candidateEnergy).toBe(3);
    expect(accepted.acceptedEnergy).toBe(3);
    expect(() =>
      acceptOrRevertFeasibleJacobiIteration(
        MESH,
        restData,
        MESH.positions,
        candidate,
        { source: 2, candidate: Number.NaN },
      ),
    ).toThrow(/assembled energies must be finite/i);
  });

  it("requires both component-aware residual and normalized update gates", () => {
    const base = {
      sceneScale: 2,
      residualTolerance: JGS2_TINY_REFERENCE_RESIDUAL_TOLERANCE,
      normalizedUpdateTolerance: 1e-4,
      feasible: true,
      gradients: {
        inertia: new Float64Array([1, 0, 0]),
        material: new Float64Array([-1, 0, 0]),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
    };
    const converged = evaluateJGS2Convergence({
      ...base,
      acceptedUpdates: new Float64Array([1e-5, 0, 0]),
    });
    expect([...converged.totalGradient]).toEqual([0, 0, 0]);
    expect(converged.residualDenominator).toBe(2);
    expect(converged.converged).toBe(true);

    const highUpdate = evaluateJGS2Convergence({
      ...base,
      acceptedUpdates: new Float64Array([1e-3, 0, 0]),
    });
    expect(highUpdate.residualSatisfied).toBe(true);
    expect(highUpdate.updateSatisfied).toBe(false);
    expect(highUpdate.converged).toBe(false);

    const highResidual = evaluateJGS2Convergence({
      ...base,
      gradients: {
        ...base.gradients,
        material: new Float64Array([-0.5, 0, 0]),
      },
      acceptedUpdates: new Float64Array(3),
    });
    expect(highResidual.residualSatisfied).toBe(false);
    expect(highResidual.updateSatisfied).toBe(true);
    expect(highResidual.converged).toBe(false);
  });

  it("keeps invalid diagnostics finite and cannot converge through failures", () => {
    const result = evaluateJGS2Convergence({
      sceneScale: 1,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-5,
      feasible: false,
      localFailureCount: 1,
      gradients: {
        inertia: new Float64Array([Number.NaN, 0, 0]),
        material: new Float64Array(3),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
      acceptedUpdates: new Float64Array(3),
    });
    expect(result.finite).toBe(false);
    expect(result.converged).toBe(false);
    expect(Number.isFinite(result.relativeResidual)).toBe(true);
    expect(Number.isFinite(result.normalizedMaximumUpdate)).toBe(true);
  });

  it("requires matching nonempty xyz gradient and update dimensions", () => {
    const gradients = {
      inertia: new Float64Array(3),
      material: new Float64Array(3),
      externalForce: new Float64Array(3),
      target: new Float64Array(3),
      contact: new Float64Array(3),
    };
    const options = {
      sceneScale: 1,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-5,
      feasible: true,
      gradients,
    };
    expect(() =>
      evaluateJGS2Convergence({
        ...options,
        acceptedUpdates: new Float64Array(0),
      }),
    ).toThrow(/match the convergence gradient dimension/i);
    expect(() =>
      evaluateJGS2Convergence({
        ...options,
        gradients: {
          ...gradients,
          inertia: new Float64Array(2),
          material: new Float64Array(2),
          externalForce: new Float64Array(2),
          target: new Float64Array(2),
          contact: new Float64Array(2),
        },
        acceptedUpdates: new Float64Array(2),
      }),
    ).toThrow(/one or more xyz triplets/i);
  });

  it("normalizes updates by max(scene scale, 1e-12)", () => {
    const result = evaluateJGS2Convergence({
      sceneScale: 0,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-3,
      feasible: true,
      gradients: {
        inertia: new Float64Array(3),
        material: new Float64Array(3),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
      acceptedUpdates: new Float64Array([5e-16, 0, 0]),
    });
    expect(result.normalizedMaximumUpdate).toBeCloseTo(5e-4, 14);
    expect(result.converged).toBe(true);
  });

  it("cannot converge through derived overflow or an assembled revert", () => {
    const overflow = evaluateJGS2Convergence({
      sceneScale: 1,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-5,
      feasible: true,
      gradients: {
        inertia: new Float64Array(3).fill(Number.MAX_VALUE),
        material: new Float64Array(3).fill(Number.MAX_VALUE),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
      acceptedUpdates: new Float64Array(3),
    });
    expect(overflow.finite).toBe(false);
    expect(overflow.converged).toBe(false);
    expect([...overflow.totalGradient].every(Number.isFinite)).toBe(true);
    expect(Number.isFinite(overflow.gradientNorm)).toBe(true);
    expect(
      Object.values(overflow.componentGradientNorms).every(Number.isFinite),
    ).toBe(true);
    expect(Number.isFinite(overflow.residualDenominator)).toBe(true);
    expect(Number.isFinite(overflow.relativeResidual)).toBe(true);
    expect(Number.isFinite(overflow.normalizedMaximumUpdate)).toBe(true);

    const updateOverflow = evaluateJGS2Convergence({
      sceneScale: 1,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-5,
      feasible: true,
      gradients: {
        inertia: new Float64Array(3),
        material: new Float64Array(3),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
      acceptedUpdates: new Float64Array(3).fill(Number.MAX_VALUE),
    });
    expect(updateOverflow.finite).toBe(false);
    expect(updateOverflow.converged).toBe(false);
    expect(Number.isFinite(updateOverflow.maximumUpdate)).toBe(true);
    expect(Number.isFinite(updateOverflow.normalizedMaximumUpdate)).toBe(true);

    const reverted = evaluateJGS2Convergence({
      sceneScale: 1,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-5,
      feasible: true,
      reverted: true,
      gradients: {
        inertia: new Float64Array(3),
        material: new Float64Array(3),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
      acceptedUpdates: new Float64Array(3),
    });
    expect(reverted.residualSatisfied).toBe(true);
    expect(reverted.updateSatisfied).toBe(true);
    expect(reverted.reverted).toBe(true);
    expect(reverted.converged).toBe(false);
  });

  it("keeps representable large norms valid without squared-norm overflow", () => {
    const result = evaluateJGS2Convergence({
      sceneScale: 1,
      residualTolerance: 1e-5,
      normalizedUpdateTolerance: 1e-5,
      feasible: true,
      gradients: {
        inertia: new Float64Array([1e200, 0, 0]),
        material: new Float64Array(3),
        externalForce: new Float64Array(3),
        target: new Float64Array(3),
        contact: new Float64Array(3),
      },
      acceptedUpdates: new Float64Array(3),
    });
    expect(result.finite).toBe(true);
    expect(result.gradientNorm).toBe(1e200);
    expect(result.componentGradientNorms.inertia).toBe(1e200);
    expect(result.residualDenominator).toBe(1e200);
    expect(result.relativeResidual).toBe(1);
    expect(result.converged).toBe(false);
  });
});
