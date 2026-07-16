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
      `WebGPU is unavailable in headless Google Chrome (secure context: ${support.secureContext}). ` +
        "A working hardware GPU, compatible driver, and normal Chrome WebGPU support are required; " +
        "software fallback is intentionally disabled.",
    );
  }

  const adapterInfo = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });

    if (!adapter) {
      return null;
    }

    return {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    };
  });

  if (!adapterInfo) {
    throw new Error(
      "Google Chrome could not create a hardware WebGPU adapter. " +
        "SwiftShader and other software fallbacks are intentionally disabled.",
    );
  }

  const adapterDescription = [
    adapterInfo.description,
    adapterInfo.vendor,
    adapterInfo.architecture,
    adapterInfo.device,
  ]
    .filter((detail, index, all) => detail && all.indexOf(detail) === index)
    .join(" / ");

  if (
    adapterInfo.isFallbackAdapter ||
    /swiftshader/i.test(adapterDescription)
  ) {
    throw new Error(
      `Chrome selected a software WebGPU adapter (${adapterDescription || "unknown adapter"}). ` +
        "This test requires a hardware GPU and will never use SwiftShader.",
    );
  }

  console.log(
    `Hardware WebGPU adapter: ${adapterDescription || "hardware adapter (details unavailable)"}`,
  );

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
      `Hardware WebGPU rendering failed in headless Google Chrome: ${renderResult.error}`,
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
