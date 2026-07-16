import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = path.join(root, "manifests", "baseline-tests.v1.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const cliModules = {
  vitest: path.join(root, "node_modules", "vitest", "vitest.mjs"),
  playwright: path.join(root, "node_modules", "@playwright", "test", "cli.js"),
};

function run(name, args) {
  const cliModule = cliModules[name];
  if (!cliModule) {
    throw new Error(`Unknown test collector ${name}.`);
  }
  const result = spawnSync(process.execPath, [cliModule, ...args], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    shell: false,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(
      `${name} ${args.join(" ")} failed with status ${result.status}.\n` +
        `${result.stdout}\n${result.stderr}`,
    );
  }
  return result.stdout;
}

function normalizedRelative(file) {
  return path.relative(root, file).split(path.sep).join("/");
}

const unitCollected = JSON.parse(run("vitest", ["list", "--json"]));
const unitKeys = new Set(
  unitCollected.map(({ file, name }) => `${normalizedRelative(file)}\0${name}`),
);

const e2eOutput = run("playwright", ["test", "--list"]);
const e2eKeys = new Set();
for (const line of e2eOutput.split(/\r?\n/u)) {
  const match = line.match(/^\s*\[[^\]]+\]\s+›\s+([^:]+):\d+:\d+\s+›\s+(.+)$/u);
  if (match) {
    const selector = match[2].replaceAll(" › ", " > ");
    e2eKeys.add(`tests/e2e/${match[1].replaceAll("\\", "/")}\0${selector}`);
  }
}

const collectedBySuite = { unit: unitKeys, e2e: e2eKeys };
const failures = [];
for (const suite of manifest.suites) {
  const frozenCount = manifest.expectedTests.filter(
    (test) => test.suite === suite.id,
  ).length;
  if (frozenCount !== suite.expectedTestCount) {
    failures.push(
      `${suite.id}: manifest declares ${suite.expectedTestCount} frozen tests but lists ${frozenCount}.`,
    );
  }
}

for (const expected of manifest.expectedTests) {
  const collected = collectedBySuite[expected.suite];
  if (!collected) {
    failures.push(`${expected.id}: unknown suite ${expected.suite}.`);
    continue;
  }
  const key = `${expected.source}\0${expected.selector}`;
  if (!collected.has(key)) {
    failures.push(
      `${expected.id}: missing collected test ${expected.source} > ${expected.selector}`,
    );
  }
}

if (failures.length > 0) {
  throw new Error(`Frozen baseline verification failed:\n- ${failures.join("\n- ")}`);
}

const unitFrozen = manifest.expectedTests.filter(({ suite }) => suite === "unit").length;
const e2eFrozen = manifest.expectedTests.filter(({ suite }) => suite === "e2e").length;
console.log(
  `Frozen baseline verified: ${unitFrozen}/${unitKeys.size} unit and ` +
    `${e2eFrozen}/${e2eKeys.size} E2E selectors are present.`,
);
