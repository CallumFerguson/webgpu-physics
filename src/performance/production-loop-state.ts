/**
 * Tracks queue submissions with monotonic watermarks. A completed queue fence
 * only retires submissions that existed when that fence was created, so an
 * older fence cannot accidentally erase newer in-flight work.
 */
export class QueueSubmissionTracker {
  private submitted = 0;
  private drained = 0;
  private maximum = 0;

  get submittedWatermark(): number {
    return this.submitted;
  }

  get currentOutstanding(): number {
    return this.submitted - this.drained;
  }

  get maximumOutstanding(): number {
    return this.maximum;
  }

  recordSubmission(): number {
    if (this.submitted === Number.MAX_SAFE_INTEGER) {
      throw new RangeError("GPU submission watermark exhausted.");
    }
    this.submitted += 1;
    this.maximum = Math.max(this.maximum, this.currentOutstanding);
    return this.submitted;
  }

  recordDrainedThrough(watermark: number): void {
    if (
      !Number.isSafeInteger(watermark) ||
      watermark < 0 ||
      watermark > this.submitted
    ) {
      throw new RangeError(
        "GPU drain watermark must identify an existing submission.",
      );
    }
    this.drained = Math.max(this.drained, watermark);
  }
}

/** Coalesces crossed periodic boundaries until a safe diagnostic drain. */
export class PeriodicDiagnosticTracker {
  private pending: number | null = null;

  constructor(private readonly interval: number) {
    if (!Number.isSafeInteger(interval) || interval < 1) {
      throw new RangeError("Diagnostic interval must be a positive safe integer.");
    }
  }

  recordCompletedRange(previousFrame: number, completedFrame: number): void {
    if (
      !Number.isSafeInteger(previousFrame) ||
      !Number.isSafeInteger(completedFrame) ||
      previousFrame < 0 ||
      completedFrame < previousFrame
    ) {
      throw new RangeError(
        "Diagnostic frame ranges must be ordered nonnegative safe integers.",
      );
    }
    const previousBoundary = Math.floor(previousFrame / this.interval);
    const completedBoundary = Math.floor(completedFrame / this.interval);
    if (completedBoundary > previousBoundary) {
      const crossedFrame = completedBoundary * this.interval;
      this.pending = Math.max(this.pending ?? 0, crossedFrame);
    }
  }

  takePendingBoundary(): number | null {
    const boundary = this.pending;
    this.pending = null;
    return boundary;
  }
}
