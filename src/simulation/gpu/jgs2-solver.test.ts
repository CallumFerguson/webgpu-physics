import { describe, expect, it } from "vitest";

import {
  DEFAULT_JGS2_STEP_SETTINGS,
  resolveJGS2StepSettings,
} from "./jgs2-solver";

describe("JGS2 runtime settings", () => {
  it("preserves the existing demo stabilizers by default", () => {
    const settings = resolveJGS2StepSettings(
      DEFAULT_JGS2_STEP_SETTINGS,
      {},
    );

    expect(settings.parityMode).toBe(false);
    expect(settings.velocityDamping).toBeLessThan(1);
    expect(settings.contactTangentialDamping).toBeGreaterThan(0);
    expect(settings.horizontalBodyCorrection).toBe(true);
  });

  it("enforces parity-safe settings even when conflicting values are supplied", () => {
    const settings = resolveJGS2StepSettings(
      DEFAULT_JGS2_STEP_SETTINGS,
      {
        parityMode: true,
        velocityDamping: 0.25,
        contactTangentialDamping: 99,
        horizontalBodyCorrection: true,
      },
    );

    expect(settings.parityMode).toBe(true);
    expect(settings.velocityDamping).toBe(1);
    expect(settings.contactTangentialDamping).toBe(0);
    expect(settings.horizontalBodyCorrection).toBe(false);
  });
});
