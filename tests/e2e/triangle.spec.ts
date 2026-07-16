import { expect, test } from "@playwright/test";

test("renders a WebGPU image without errors", async ({ page }, testInfo) => {
  const browserErrors: string[] = [];

  page.on("pageerror", (error) => {
    browserErrors.push(`Page error: ${error.message}`);
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`Console error: ${message.text()}`);
    }
  });

  await page.goto("/");

  const support = await page.evaluate(() => ({
    available: typeof navigator.gpu !== "undefined",
    secureContext: window.isSecureContext,
  }));

  if (!support.available) {
    throw new Error(
      `WebGPU is unavailable in headless Chromium (secure context: ${support.secureContext}). ` +
        "Install Playwright Chromium and keep the configured WebGPU SwiftShader flags enabled.",
    );
  }

  await page.waitForFunction(() => "__webgpuRenderDone" in window);

  const renderResult = await page.evaluate(async () => {
    try {
      await window.__webgpuRenderDone;
      return { ok: true, error: "" };
    } catch (reason: unknown) {
      const error =
        reason instanceof Error
          ? `${reason.name}: ${reason.message}`
          : String(reason);
      return { ok: false, error };
    }
  });

  if (!renderResult.ok) {
    throw new Error(
      `WebGPU rendering failed in headless Chromium: ${renderResult.error}`,
    );
  }

  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }),
  );

  expect(browserErrors, "The page reported errors while rendering").toEqual([]);

  const screenshotPath = testInfo.outputPath("triangle.png");
  const screenshot = await page.getByTestId("gpu-canvas").screenshot({
    path: screenshotPath,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });

  expect(screenshot.byteLength, "The canvas screenshot was empty").toBeGreaterThan(
    0,
  );
  console.log(`WebGPU screenshot saved to: ${screenshotPath}`);
});
