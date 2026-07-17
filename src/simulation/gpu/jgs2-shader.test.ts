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
  it("keeps the inertial target separate from the first feasible source pose", () => {
    const predict = functionBody("predict");
    expect(predict).toContain(
      "select(predicted, oldPosition, params.offsets4.w != 0u)",
    );
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
      "// Exact local restriction: inertia plus every incident tetrahedron.",
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
  });
});
