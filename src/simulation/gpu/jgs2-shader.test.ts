import { describe, expect, it } from "vitest";

import { jgs2Shader } from "./jgs2-shader";

function functionBody(name: string): string {
  const start = jgs2Shader.indexOf(`fn ${name}(`);
  if (start < 0) throw new Error(`Missing WGSL function ${name}.`);
  const next = jgs2Shader.indexOf("\nfn ", start + 3);
  const nextEntry = jgs2Shader.indexOf("\n@compute", start + 3);
  const candidates = [next, nextEntry].filter((index) => index >= 0);
  const end = candidates.length > 0 ? Math.min(...candidates) : jgs2Shader.length;
  return jgs2Shader.slice(start, end);
}

describe("JGS2 stable globalization shader contract", () => {
  it("binds one guarded read-only composite-objective record per vertex", () => {
    expect(jgs2Shader).toContain("struct VertexObjective {");
    expect(jgs2Shader).toContain("externalForce: vec4f");
    expect(jgs2Shader).toContain("targetPositionStiffness: vec4f");
    expect(jgs2Shader).toContain(
      "@group(0) @binding(6)\nvar<storage, read> vertexObjectives: array<VertexObjective>",
    );
    expect(jgs2Shader).toContain(
      "@group(0) @binding(7)\nvar<uniform> params: SimParams",
    );
    expect(jgs2Shader).toContain("fn ipcContactBase() -> u32");
    expect(jgs2Shader).toContain(
      "return dynamicData[ipcContactBase() + 1u]",
    );
    expect(jgs2Shader).toContain(
      "const OBJECTIVE_FORCE_ENABLED_BIT: u32 = 1u",
    );
    expect(jgs2Shader).toContain(
      "const OBJECTIVE_TARGET_ENABLED_BIT: u32 = 2u",
    );

    const force = functionBody("vertexExternalForce");
    expect(force).toContain("if (objectiveForceEnabled())");
    expect(force.indexOf("if (objectiveForceEnabled())")).toBeLessThan(
      force.indexOf("vertexObjectives[vertex].externalForce.xyz"),
    );
    const target = functionBody("vertexTargetPositionStiffness");
    expect(target).toContain("if (objectiveTargetEnabled())");
    expect(target.indexOf("if (objectiveTargetEnabled())")).toBeLessThan(
      target.indexOf("vertexObjectives[vertex].targetPositionStiffness"),
    );
  });

  it("keeps the inertial target separate from the first feasible source pose", () => {
    const predict = functionBody("predict");
    expect(predict).toContain(
      "select(predicted, oldPosition, params.offsets4.w != 0u)",
    );
    expect(predict).toContain(
      "params.time.x * params.time.x * params.gravityFloor.xyz",
    );
    expect(predict).not.toContain("vertexObjectives");
    expect(predict).not.toContain("vertexExternalForce");
    expect(predict).not.toContain("vertexTargetPositionStiffness");
  });

  it("adds force and target derivatives to both exact and Cubature restrictions", () => {
    const solve = functionBody("jgs2Solve");
    expect(solve).toContain("vertexObjectiveGradient(vertex, position)");
    expect(solve).toContain(
      "hessian += vertexTargetStiffness(vertex) * identityMat3()",
    );
    expect(solve).toContain("distributedObjectiveGradient(indices[local])");
    expect(solve).toContain("cubatureObjectiveHessian(indices, basis)");
    expect(solve).toContain("distributedObjectiveGradient(vertex)");
    expect(solve.match(/distributedTargetStiffness\(vertex\)/g)).toHaveLength(1);

    const objectiveGradient = functionBody("vertexObjectiveGradient");
    expect(objectiveGradient).toContain("result -= vertexExternalForce(vertex)");
    expect(objectiveGradient).toContain("if (targetRecord.w > 0.0)");
    expect(objectiveGradient).toContain(
      "targetRecord.w * (position - targetRecord.xyz)",
    );
    expect(
      objectiveGradient.indexOf("if (targetRecord.w > 0.0)"),
    ).toBeLessThan(
      objectiveGradient.indexOf("position - targetRecord.xyz"),
    );
    const objectiveHessian = functionBody("cubatureObjectiveHessian");
    expect(objectiveHessian).toContain("distributedTargetStiffness(vertex)");
    expect(objectiveHessian).toContain(
      "transpose(basis[local]) * basis[local]",
    );
  });

  it("uses the complete objective in restricted and assembled energy", () => {
    const objectiveDelta = functionBody("vertexObjectiveEnergyDelta");
    expect(objectiveDelta).toContain(
      "result -= dot(vertexExternalForce(vertex), displacement)",
    );
    expect(objectiveDelta).toContain("quadraticEnergyDelta(");
    expect(objectiveDelta).toContain("targetRecord.w");
    expect(
      objectiveDelta.indexOf("if (targetRecord.w > 0.0)"),
    ).toBeLessThan(
      objectiveDelta.indexOf("targetRecord.xyz"),
    );

    const restricted = functionBody("restrictedEnergyDelta");
    expect(restricted).toContain("vertexObjectiveEnergyDelta(");
    expect(
      restricted.match(/distributedVertexObjectiveEnergyDelta\(/g),
    ).toHaveLength(2);

    const assembled = functionBody("vertexImplicitEnergyAt");
    expect(assembled).toContain(
      "energy -= dot(vertexExternalForce(vertex), position)",
    );
    expect(assembled).toContain(
      "0.5 * targetRecord.w * dot(targetResidual, targetResidual)",
    );
    expect(assembled.indexOf("if (targetRecord.w > 0.0)")).toBeLessThan(
      assembled.indexOf("position - targetRecord.xyz"),
    );
  });

  it("writes explicit force and target convergence components", () => {
    const convergence = functionBody("convergenceGradient");
    expect(convergence).toContain(
      "externalForce = -vertexExternalForce(vertex)",
    );
    expect(convergence).toContain("var targetGradient = vec3f(0.0)");
    expect(convergence).toContain("if (targetRecord.w > 0.0)");
    expect(convergence.indexOf("if (targetRecord.w > 0.0)")).toBeLessThan(
      convergence.indexOf("position - targetRecord.xyz"),
    );
    expect(convergence).toContain(
      "dynamicData[base + 2u] = vec4f(externalForce, 0.0)",
    );
    expect(convergence).toContain(
      "dynamicData[base + 3u] = vec4f(targetGradient, 0.0)",
    );
  });

  it("uniformly bypasses objective projection and force-only target work", () => {
    const solve = functionBody("jgs2Solve");
    const solveObjectiveGuard = solve.indexOf("if (params.offsets2.w != 0u)");
    expect(solveObjectiveGuard).toBeGreaterThanOrEqual(0);
    expect(solveObjectiveGuard).toBeLessThan(
      solve.indexOf("vertexObjectiveGradient(vertex, position)"),
    );
    const cubatureObjective = solve.indexOf(
      "cubatureObjectiveHessian(indices, basis)",
    );
    expect(
      solve.lastIndexOf("if (objectiveTargetEnabled())", cubatureObjective),
    ).toBeGreaterThanOrEqual(0);
    const sourceTarget = solve.indexOf(
      "distributedTargetStiffness(vertex)",
    );
    expect(
      solve.lastIndexOf("if (objectiveTargetEnabled())", sourceTarget),
    ).toBeGreaterThanOrEqual(0);

    const restricted = functionBody("restrictedEnergyDelta");
    expect(restricted.indexOf("if (params.offsets2.w != 0u)")).toBeLessThan(
      restricted.indexOf("vertexObjectiveEnergyDelta("),
    );

    const objectiveGradient = functionBody("vertexObjectiveGradient");
    expect(objectiveGradient.indexOf("if (objectiveTargetEnabled())")).toBeLessThan(
      objectiveGradient.indexOf("vertexTargetPositionStiffness(vertex)"),
    );
    const objectiveDelta = functionBody("vertexObjectiveEnergyDelta");
    expect(objectiveDelta.indexOf("if (objectiveTargetEnabled())")).toBeLessThan(
      objectiveDelta.indexOf("vertexTargetPositionStiffness(vertex)"),
    );

    for (const name of ["vertexImplicitEnergyAt", "convergenceGradient"]) {
      const body = functionBody(name);
      const outerGuard = body.indexOf("if (params.offsets2.w != 0u)");
      expect(outerGuard, name).toBeGreaterThanOrEqual(0);
      expect(outerGuard, name).toBeLessThan(
        body.indexOf("vertexExternalForce(vertex)"),
      );
      expect(outerGuard, name).toBeLessThan(
        body.indexOf("vertexTargetPositionStiffness(vertex)"),
      );
    }
  });

  it("uses the shared frozen shift solver only on the stable path", () => {
    const solve = functionBody("jgs2Solve");
    expect(solve).toContain("jgs2_globalization_solve(");
    expect(solve).toContain("var delta = regularizedSolve(hessian, -gradient)");
    expect(solve.indexOf("jgs2_globalization_solve(")).toBeLessThan(
      solve.indexOf("var delta = regularizedSolve"),
    );
  });

  it("checks every local trial's geometry before evaluating its energy", () => {
    const solve = functionBody("jgs2Solve");
    const geometry = solve.indexOf(
      "let trialGeometry = restrictedMinimumDeformationDeterminant(",
    );
    const energy = solve.indexOf("let trialEnergyDelta = restrictedEnergyDelta(");
    expect(geometry).toBeGreaterThanOrEqual(0);
    expect(energy).toBeGreaterThan(geometry);
    expect(solve).toContain("if (trialGeometry.valid == 0u)");
    expect(solve).toContain("localStatus = LOCAL_STATUS_NONFINITE");
    expect(solve).toContain("trialMinimum > jgs2_globalization_determinant_floor");
  });

  it("tests and stores the exact rounded f32 Armijo update", () => {
    const solve = functionBody("jgs2Solve");
    const roundedPosition = solve.indexOf(
      "let trialPosition = position + nominalTrialDisplacement",
    );
    const effectiveUpdate = solve.indexOf(
      "let trialDisplacement = trialPosition - position",
    );
    const geometry = solve.indexOf(
      "let trialGeometry = restrictedMinimumDeformationDeterminant(",
    );
    const energy = solve.indexOf("let trialEnergyDelta = restrictedEnergyDelta(");
    expect(roundedPosition).toBeGreaterThanOrEqual(0);
    expect(effectiveUpdate).toBeGreaterThan(roundedPosition);
    expect(geometry).toBeGreaterThan(effectiveUpdate);
    expect(energy).toBeGreaterThan(geometry);
    expect(solve).toContain(
      "let trialGradientDotDisplacement = dot(",
    );
    expect(solve).toContain(
      "jgs2_globalization_armijo_c1 *\n            trialGradientDotDisplacement",
    );
    expect(solve).toContain("acceptedDisplacement = trialDisplacement");
    expect(solve).toContain("acceptedPosition = trialPosition");
    expect(solve).toContain("vec4f(acceptedPosition, 1.0)");
    expect(solve).not.toContain("vec4f(position + acceptedDisplacement, 1.0)");
    expect(solve).toContain("let sourceCoordinateScale = max(");
    expect(solve).toContain("max(sourceCoordinateScale, 1.0e-12)");
    expect(solve.indexOf("sourceCoordinateScale")).toBeLessThan(
      solve.indexOf("let belowPositionResolution"),
    );
  });

  it("keeps nonfinite and no-op trial geometry distinct", () => {
    const geometry = functionBody(
      "restrictedMinimumDeformationDeterminant",
    );
    expect(geometry).toContain("RestrictedGeometryResult(0.0, 0u)");
    expect(geometry).toContain(
      "RestrictedGeometryResult(minimumDeterminant, 1u)",
    );

    const solve = functionBody("jgs2Solve");
    expect(solve).toContain("trialLength <= positionResolution");
    expect(solve).toContain("localStatus = LOCAL_STATUS_ZERO_GRADIENT");
    expect(solve).toContain("if (trialGeometry.valid == 0u)");
    expect(solve).toContain("minimumTrialDeterminant = 0.0");
  });

  it("seeds local geometry and rejects an unsafe preflight source before material work", () => {
    const geometry = functionBody(
      "restrictedMinimumDeformationDeterminant",
    );
    expect(geometry).toContain(
      "(preflightValidity & ACCEPTED_MINIMUM_VALID_BIT) != 0u",
    );
    expect(geometry).toContain(
      "minimumDeterminant = preflightGeometry.z",
    );

    const solve = functionBody("jgs2Solve");
    const preflight = solve.indexOf(
      "let preflightGeometry = dynamicData[params.offsets4.x]",
    );
    const material = solve.indexOf(
      "// Exact local restriction: inertia, user force, soft target, and every",
    );
    expect(preflight).toBeGreaterThanOrEqual(0);
    expect(material).toBeGreaterThan(preflight);
    expect(solve.slice(preflight, material)).toContain(
      "preflightGeometry.z > jgs2_globalization_determinant_floor",
    );
    expect(solve.slice(preflight, material)).toContain(
      "dynamicData[params.offsets1.w + vertex]",
    );
    expect(solve.slice(preflight, material)).toContain(
      "LOCAL_STATUS_LINE_SEARCH_FAILED",
    );
    expect(solve.slice(preflight, material)).toContain(
      "LOCAL_STATUS_NONFINITE",
    );
    expect(solve.slice(preflight, material)).toContain("return;");
  });

  it("terminates a nonfinite trial energy as a numerical failure", () => {
    const solve = functionBody("jgs2Solve");
    const energy = solve.indexOf("let trialEnergyDelta = restrictedEnergyDelta(");
    const finiteCheck = solve.indexOf(
      "if (!jgs2_globalization_finite_scalar(trialEnergyDelta))",
      energy,
    );
    const nextBacktrack = solve.indexOf(
      "trialAlpha *= jgs2_globalization_backtrack_factor",
      energy,
    );
    expect(energy).toBeGreaterThanOrEqual(0);
    expect(finiteCheck).toBeGreaterThan(energy);
    expect(nextBacktrack).toBeGreaterThan(finiteCheck);
    expect(solve.slice(finiteCheck, nextBacktrack)).toContain(
      "localStatus = LOCAL_STATUS_NONFINITE",
    );
    expect(solve.slice(finiteCheck, nextBacktrack)).not.toContain(
      "minimumTrialDeterminant = 0.0",
    );
    expect(solve.slice(finiteCheck, nextBacktrack)).toContain("break;");
  });

  it("does not evaluate assembled material energy through an infeasible pose", () => {
    const candidate = functionBody("candidateTetrahedron");
    const feasibility = candidate.indexOf(
      "candidateJ > jgs2_globalization_determinant_floor",
    );
    const energy = candidate.indexOf(
      "candidateEnergy = attributes.x * snh_energy_density(",
    );
    expect(feasibility).toBeGreaterThanOrEqual(0);
    expect(energy).toBeGreaterThan(feasibility);
  });

  it("uses deterministic one-workgroup strided candidate reduction", () => {
    const reduction = functionBody("reduceCandidate");
    expect(reduction).toContain("@builtin(local_invocation_index) lane: u32");
    expect(jgs2Shader).toContain(
      "@compute @workgroup_size(WORKGROUP_SIZE)\nfn reduceCandidate",
    );
    expect(reduction).toContain("var tet = lane");
    expect(reduction).toContain("tet += WORKGROUP_SIZE");
    expect(reduction).toContain("var vertex = lane");
    expect(reduction).toContain("vertex += WORKGROUP_SIZE");
    expect(reduction).toContain("var stride = WORKGROUP_SIZE / 2u");
    expect(reduction).toContain("if (lane < stride)");
    expect(reduction).toContain("stride /= 2u");
    expect(reduction.indexOf("workgroupBarrier();")).toBeLessThan(
      reduction.indexOf("var stride = WORKGROUP_SIZE / 2u"),
    );
    expect(reduction).toContain(
      "left.validityBits &= right.validityBits",
    );
    expect(reduction).not.toContain("@workgroup_size(1)");
  });

  it("counts prior trials without treating solver-level zero as a trial", () => {
    const reduction = functionBody("reduceCandidate");
    expect(reduction).toContain(
      "status == LOCAL_STATUS_ZERO_GRADIENT && backtracks > 0u",
    );
    expect(reduction).toContain("evaluatedTrials = backtracks");
    expect(reduction).toContain("status == LOCAL_STATUS_NONFINITE");
    expect(reduction).toContain(
      "local3.x > jgs2_globalization_determinant_floor",
    );
    expect(reduction).toContain("evaluatedTrials += 1u");
  });

  it("uses exactly two strided scans and deterministic trees for convergence", () => {
    const reduction = functionBody("reduceConvergence");
    expect(reduction).toContain("@builtin(local_invocation_index) lane: u32");
    expect(jgs2Shader).toContain(
      "@compute @workgroup_size(WORKGROUP_SIZE)\nfn reduceConvergence",
    );
    expect(reduction.match(/var vertex = lane/g)).toHaveLength(2);
    expect(reduction.match(/vertex \+= WORKGROUP_SIZE/g)).toHaveLength(2);
    expect(reduction).toContain("var scaleStride = WORKGROUP_SIZE / 2u");
    expect(reduction).toContain("var sumStride = WORKGROUP_SIZE / 2u");
    expect(reduction).toContain("scaleStride /= 2u");
    expect(reduction).toContain("sumStride /= 2u");
    expect(reduction).toContain(
      "jgs2_globalization_finite_vec3(rawValue)",
    );
    expect(reduction).toContain(
      "jgs2_globalization_finite_scalar(update)",
    );
    expect(reduction).not.toContain("convergenceComponentNorm(");
    expect(reduction).not.toContain("convergenceTotalGradientNorm(");
    expect(reduction).not.toContain("@workgroup_size(1)");
  });

  it("packs independent geometry and local-numeric validity bits", () => {
    const candidate = functionBody("reduceCandidate");
    expect(candidate).toContain(
      "validityBits |= ACCEPTED_MINIMUM_VALID_BIT",
    );
    expect(candidate).toContain(
      "let acceptedMinimumValid = select(",
    );
    expect(candidate).toContain("f32(controlValidityBits)");

    const convergence = functionBody("reduceConvergence");
    expect(convergence).toContain(
      "(controlValidityBits & SOURCE_GEOMETRY_VALID_BIT) != 0u",
    );
    expect(convergence).toContain(
      "(controlValidityBits & CANDIDATE_GEOMETRY_VALID_BIT) != 0u",
    );
    expect(convergence).toContain(
      "(controlValidityBits & ACCEPTED_MINIMUM_VALID_BIT) != 0u",
    );
    expect(convergence).toContain(
      "(controlValidityBits & LOCAL_NUMERICS_VALID_BIT) != 0u",
    );
    expect(convergence).toContain(
      "sourceGeometryValid && candidateGeometryValid",
    );
  });

  it("masks the control-only nonlinear stop latch out of validity decoding", () => {
    expect(jgs2Shader).toContain("const NONLINEAR_STOP_LATCH_BIT: u32 = 16u");
    const controlBits = functionBody("globalizationControlBits");
    expect(controlBits).toContain("GLOBALIZATION_CONTROL_BITS_MASK");
    const validityBits = functionBody("globalizationValidityBits");
    expect(validityBits).toContain(
      "globalizationControlBits(packed) & VALIDITY_BITS_MASK",
    );
  });

  it("short-circuits every remaining stable solve/globalization pass", () => {
    const guardedWork = [
      ["tetPolarRotation", "if (usesStableNeoHookean(tet))"],
      ["vertexPolarRotation", "let info = vertexData[vertex].info"],
      ["jgs2Solve", "let vertexItem = vertexData[vertex]"],
      ["candidateTetrahedron", "let attributes = tetData[tet].attributes"],
      ["applyCandidate", "let accepted = dynamicData[params.offsets4.x + 1u].x"],
      ["convergenceGradient", "let base = params.offsets3.w"],
    ] as const;
    for (const [name, firstWork] of guardedWork) {
      const body = functionBody(name);
      const guard = body.indexOf("if (nonlinearStopLatched())");
      expect(guard, name).toBeGreaterThanOrEqual(0);
      expect(guard, name).toBeLessThan(body.indexOf(firstWork));
    }

    // Workgroup reductions must keep every barrier in uniform control flow.
    // Only their lane-zero storage side effects are suppressed after latching.
    for (const name of ["reduceCandidate", "reduceConvergence"] as const) {
      const body = functionBody(name);
      expect(body).not.toContain("if (nonlinearStopLatched()) {\n    return;");
      const lastBarrier = body.lastIndexOf("workgroupBarrier()");
      const gatedWrite = body.indexOf(
        "if (lane == 0u && !nonlinearStopLatched())",
      );
      expect(lastBarrier, name).toBeGreaterThanOrEqual(0);
      expect(gatedWrite, name).toBeGreaterThan(lastBarrier);
    }
  });

  it("propagates odd- and even-iteration convergence through ping-pong", () => {
    const solve = functionBody("jgs2Solve");
    const guard = solve.indexOf("if (nonlinearStopLatched())");
    const copy = solve.indexOf(
      "dynamicData[params.offsets2.x + vertex] =\n      dynamicData[params.offsets1.w + vertex]",
      guard,
    );
    expect(copy).toBeGreaterThan(guard);
    expect(copy).toBeLessThan(solve.indexOf("let vertexItem"));
    expect(functionBody("copyPosition")).not.toContain(
      "nonlinearStopLatched",
    );

    const finalCanonicalPose = (executedIterations: number): string => {
      let a = "initial-a";
      let b = "initial-b";
      const configuredIterations = 5;
      for (let iteration = 0; iteration < configuredIterations; iteration += 1) {
        const source = iteration % 2 === 0 ? b : a;
        const target =
          iteration < executedIterations ? `accepted-${iteration}` : source;
        if (iteration % 2 === 0) a = target;
        else b = target;
      }
      return a;
    };
    expect(finalCanonicalPose(1)).toBe("accepted-0");
    expect(finalCanonicalPose(2)).toBe("accepted-1");
  });

  it("latches only after writing the first converged history record", () => {
    const stop = functionBody("stopAfterConvergenceEnabled");
    expect(stop).toContain("params.convergence.w == 1.0");

    const reduction = functionBody("reduceConvergence");
    const historyWrite = reduction.indexOf(
      "dynamicData[history] = dynamicData[control]",
    );
    const historyCount = reduction.indexOf(
      "dynamicData[control + 1u].w = f32(historyIndex + 1u)",
    );
    const latch = reduction.indexOf(
      "if (converged && stopAfterConvergenceEnabled())",
    );
    expect(historyWrite).toBeGreaterThanOrEqual(0);
    expect(historyCount).toBeGreaterThan(historyWrite);
    expect(latch).toBeGreaterThan(historyCount);
    expect(reduction.slice(latch)).toContain(
      "controlValidityBits | NONLINEAR_STOP_LATCH_BIT",
    );
  });

  it("keeps portable workgroup scratch below 16 KiB", () => {
    expect(jgs2Shader).toContain(
      "40 bytes/lane * 128 lanes = 5,120 bytes",
    );
    expect(jgs2Shader).toContain(
      "32 bytes/lane * 128 lanes = 4,096 bytes",
    );
    expect(jgs2Shader).toContain("consume 9,216 bytes");
  });

  it("reverts a rejected candidate by copying the complete source vec4", () => {
    const apply = functionBody("applyCandidate");
    expect(apply).toContain(
      "dynamicData[params.offsets2.x + vertex] =\n      dynamicData[params.offsets1.w + vertex]",
    );
    const convergence = functionBody("reduceConvergence");
    expect(convergence).toContain(
      "assembledAccepted && !assembledReverted && localFailureCount == 0u",
    );
    expect(convergence).toContain("residualSatisfied && updateSatisfied");
  });

  it("does not reapply a rejected stable pinned target after the assembled gate", () => {
    const finalize = functionBody("finalize");
    const stableGuard = finalize.indexOf("if (params.offsets4.w == 0u)");
    const restTarget = finalize.indexOf(
      "vec4f(vertexData[vertex].restMass.xyz, 1.0)",
    );
    expect(stableGuard).toBeGreaterThanOrEqual(0);
    expect(restTarget).toBeGreaterThan(stableGuard);
    expect(finalize).toContain(
      "dynamicData[params.offsets0.w + vertex] = vec4f(0.0)",
    );
    expect(finalize).not.toContain("vertexObjectives");
    expect(finalize).not.toContain("vertexTargetPositionStiffness");
  });
});
