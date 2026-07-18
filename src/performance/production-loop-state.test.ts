import { describe, expect, it } from "vitest";

import {
  PeriodicDiagnosticTracker,
  QueueSubmissionTracker,
} from "./production-loop-state";

describe("production loop state", () => {
  it("retires only submissions covered by each queue fence", () => {
    const tracker = new QueueSubmissionTracker();
    tracker.recordSubmission();
    const firstBatch = tracker.recordSubmission();
    tracker.recordSubmission();
    const secondBatch = tracker.recordSubmission();

    expect(tracker.currentOutstanding).toBe(4);
    tracker.recordDrainedThrough(firstBatch);
    expect(tracker.currentOutstanding).toBe(2);

    tracker.recordSubmission();
    const thirdBatch = tracker.recordSubmission();
    tracker.recordDrainedThrough(secondBatch);
    expect(tracker.currentOutstanding).toBe(2);
    expect(tracker.maximumOutstanding).toBe(4);

    tracker.recordDrainedThrough(thirdBatch);
    expect(tracker.currentOutstanding).toBe(0);
  });

  it("keeps drain updates monotonic when callbacks settle out of order", () => {
    const tracker = new QueueSubmissionTracker();
    tracker.recordSubmission();
    const firstBatch = tracker.recordSubmission();
    tracker.recordSubmission();
    const secondBatch = tracker.recordSubmission();

    tracker.recordDrainedThrough(secondBatch);
    tracker.recordDrainedThrough(firstBatch);
    expect(tracker.currentOutstanding).toBe(0);
  });

  it("retains a crossed diagnostic boundary for a later idle batch", () => {
    const tracker = new PeriodicDiagnosticTracker(600);
    tracker.recordCompletedRange(599, 600);

    // The crossing batch may finish while a newer batch is still in flight.
    tracker.recordCompletedRange(600, 602);
    expect(tracker.takePendingBoundary()).toBe(600);
    expect(tracker.takePendingBoundary()).toBeNull();
  });

  it("does not erase a new boundary recorded after a prior one is consumed", () => {
    const tracker = new PeriodicDiagnosticTracker(600);
    tracker.recordCompletedRange(599, 600);
    expect(tracker.takePendingBoundary()).toBe(600);

    tracker.recordCompletedRange(1199, 1200);
    expect(tracker.takePendingBoundary()).toBe(1200);
  });
});
