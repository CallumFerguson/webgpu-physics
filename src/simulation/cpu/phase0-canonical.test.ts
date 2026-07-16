import { describe, expect, it } from "vitest";

import {
  PHASE0_ORACLE_CORPUS_CASE_COUNT,
  PHASE0_ORACLE_CORPUS_GENERATOR_VERSION,
  PHASE0_ORACLE_CORPUS_ID,
  PHASE0_ORACLE_CORPUS_SEED,
  PHASE0_ORACLE_FIXTURE_ID,
  PHASE0_ORACLE_MINIMUM_DETERMINANT,
  PHASE0_ORACLE_REST_POSITIONS,
  buildPhase0OracleFixture,
  evaluatePhase0OraclePose,
  generatePhase0OraclePoseCorpus,
  measurePhase0OraclePoseErrors,
} from "./phase0-canonical";

describe("frozen Phase 0 single-tetrahedron corpus", () => {
  it("uses the manifest's explicit positive tetrahedron and generator", () => {
    const fixture = buildPhase0OracleFixture();
    const first = generatePhase0OraclePoseCorpus();
    const second = generatePhase0OraclePoseCorpus();

    expect(fixture.definition.id).toBe(PHASE0_ORACLE_FIXTURE_ID);
    expect(fixture.mesh.positions).toEqual(PHASE0_ORACLE_REST_POSITIONS);
    expect([...fixture.mesh.tetrahedra]).toEqual([0, 1, 2, 3]);
    expect([...fixture.mesh.fixed]).toEqual([1, 0, 0, 0]);
    expect(fixture.definition.materials[0]).toMatchObject({
      density: 1_000,
      youngModulus: 80_000,
      poissonRatio: 0.3,
    });
    expect(fixture.definition.settings.timestep).toBe(1 / 60);
    expect(PHASE0_ORACLE_CORPUS_SEED).toBe(65_537);
    expect(PHASE0_ORACLE_CORPUS_GENERATOR_VERSION).toBe("1");
    expect(first).toEqual(second);
    expect(first).toHaveLength(PHASE0_ORACLE_CORPUS_CASE_COUNT);
    expect(new Set(first.map((pose) => pose.id)).size).toBe(
      PHASE0_ORACLE_CORPUS_CASE_COUNT,
    );
    expect(first[0]).toMatchObject({
      id: `${PHASE0_ORACLE_CORPUS_ID}/00`,
      kind: "rest",
      determinant: 1,
    });
    expect(first.filter((pose) => pose.kind === "rigid")).toHaveLength(3);
    expect(first.filter((pose) => pose.kind === "deformed")).toHaveLength(60);
    expect([...first[4]!.positions]).toEqual([
      0, 0, 0,
      0.9659412819892168, 0.09770480934530497, -0.03810854216571897,
      0.25658654181752355, 0.8192284137010575, 0.11348439222201705,
      0.028169284132309264, 0.32534589504357425, 0.8768623472424224,
    ]);
    expect(first[4]!.determinant).toBe(0.8290824329547426);
    expect([...first[63]!.positions]).toEqual([
      0, 0, 0,
      0.8653492582030594, 0.08063826877623796, 0.13988581504672765,
      0.15422659039031714, 0.9303950880188495, -0.0047778727021068335,
      0.005720784794539219, 0.10270756869576873, 0.9733727779239416,
    ]);
    expect(first[63]!.determinant).toBe(1.0110679439101526);

    for (const pose of first) {
      expect(pose.positions.subarray(0, 3)).toEqual(
        PHASE0_ORACLE_REST_POSITIONS.subarray(0, 3),
      );
      expect(pose.determinant).toBeGreaterThanOrEqual(
        PHASE0_ORACLE_MINIMUM_DETERMINANT,
      );
      expect([...pose.positions].every(Number.isFinite)).toBe(true);
    }
  });

  it("meets every derivative, Newton, and equilibrium-basis gate on all 64 poses", () => {
    const start = performance.now();
    const fixture = buildPhase0OracleFixture();
    const corpus = generatePhase0OraclePoseCorpus();
    const worst = {
      gradient: { error: 0, id: "" },
      hessian: { error: 0, id: "" },
      newton: { error: 0, id: "" },
      block: { error: 0, id: "" },
      basis: { error: 0, id: "" },
    };

    for (const pose of corpus) {
      const errors = measurePhase0OraclePoseErrors(fixture, pose);
      const candidates = [
        ["gradient", errors.gradientRelativeError],
        ["hessian", errors.hessianRelativeError],
        ["newton", errors.newtonResidualRelativeError],
        ["block", errors.maximumNewtonBlockRelativeError],
        ["basis", errors.maximumEquilibriumBasisRelativeError],
      ] as const;
      for (const [name, error] of candidates) {
        if (error > worst[name].error) {
          worst[name] = { error, id: pose.id };
        }
      }

      expect(errors.gradientRelativeError, `${pose.id} gradient`).toBeLessThanOrEqual(
        1e-5,
      );
      expect(errors.hessianRelativeError, `${pose.id} Hessian`).toBeLessThanOrEqual(
        1e-4,
      );
      expect(errors.newtonResidualRelativeError, `${pose.id} Newton residual`).toBeLessThanOrEqual(
        1e-8,
      );
      expect(errors.maximumNewtonBlockRelativeError, `${pose.id} Newton block`).toBeLessThanOrEqual(
        1e-8,
      );
      expect(errors.maximumEquilibriumBasisRelativeError, `${pose.id} equilibrium basis`).toBeLessThanOrEqual(
        1e-8,
      );
    }

    for (const pose of corpus.filter((entry) => entry.kind !== "deformed")) {
      const { evaluation } = evaluatePhase0OraclePose(fixture, pose);
      expect(Math.abs(evaluation.components.elasticity), `${pose.id} energy`).toBeLessThan(
        1e-8,
      );
      expect(
        Math.hypot(...evaluation.gradient),
        `${pose.id} material gradient`,
      ).toBeLessThan(1e-7);
    }

    console.log(
      `Phase 0 oracle corpus (${corpus.length} poses, ` +
        `${(performance.now() - start).toFixed(1)} ms): ` +
        `gradient=${worst.gradient.error.toExponential(3)} (${worst.gradient.id}), ` +
        `Hessian=${worst.hessian.error.toExponential(3)} (${worst.hessian.id}), ` +
        `Newton=${worst.newton.error.toExponential(3)} (${worst.newton.id}), ` +
        `block=${worst.block.error.toExponential(3)} (${worst.block.id}), ` +
        `basis=${worst.basis.error.toExponential(3)} (${worst.basis.id})`,
    );
  }, 30_000);
});
