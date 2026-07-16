import { defineConfig, devices } from "@playwright/test";

const port = 4173;
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests/e2e",
  globalSetup: "./tests/e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: "list",
  preserveOutput: "always",
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL,
    viewport: { width: 800, height: 600 },
    deviceScaleFactor: 1,
    colorScheme: "dark",
    locale: "en-US",
    timezoneId: "UTC",
    contextOptions: {
      reducedMotion: "reduce",
    },
    navigationTimeout: 15_000,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chrome-hardware",
      use: {
        ...devices["Desktop Chrome"],
        browserName: "chromium",
        channel: "chrome",
        headless: true,
        launchOptions: {
          args: ["--enable-gpu", "--disable-software-rasterizer"],
        },
      },
    },
  ],
});
