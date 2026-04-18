#!/usr/bin/env node
/**
 * WCAG 2.1 AA audit runner.
 *
 * Invokes `@axe-core/cli` against every page route in the app, aggregates
 * findings into a single markdown report at `docs/a11y-audit-results.md`.
 *
 * Prerequisites:
 *   1. `npm install` has been run so `@axe-core/cli` is available.
 *   2. A dev server is serving the app — start one in another terminal:
 *        npm start                       # defaults to http://localhost:4200
 *
 * Usage:
 *   npm run a11y                         # uses default base URL
 *   A11Y_BASE_URL=http://host:port npm run a11y
 *
 * Exit code:
 *   0 if every route had zero critical+serious violations.
 *   1 otherwise (for CI).
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const execFileP = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const OUT_PATH = path.resolve(REPO_ROOT, 'docs', 'a11y-audit-results.md');

const BASE_URL = process.env.A11Y_BASE_URL ?? 'http://localhost:4200';

// Every page route (matches frontend/src/app/pages/) — keep in sync with routes.ts.
const ROUTES = [
  { slug: '/', label: 'dashboard' },
  { slug: '/portfolio-builder', label: 'portfolio-builder' },
  { slug: '/portfolio-detail', label: 'portfolio-detail' },
  { slug: '/optimization-studio', label: 'optimization-studio' },
  { slug: '/backtesting', label: 'backtesting' },
  { slug: '/risk-center', label: 'risk-center' },
  { slug: '/factor-research', label: 'factor-research' },
  { slug: '/rebalancing', label: 'rebalancing' },
  { slug: '/attribution', label: 'attribution' },
  { slug: '/macro-intelligence', label: 'macro-intelligence' },
  { slug: '/ai-control-room', label: 'ai-control-room' },
  { slug: '/instrument-detail', label: 'instrument-detail' },
  { slug: '/settings', label: 'settings' },
];

const BLOCKING_IMPACTS = new Set(['critical', 'serious']);

async function runAxe(url) {
  const { stdout } = await execFileP(
    'npx',
    ['--yes', '@axe-core/cli', url, '--tags', 'wcag2a,wcag2aa,wcag21a,wcag21aa', '--stdout'],
    { maxBuffer: 8 * 1024 * 1024 },
  );
  return JSON.parse(stdout);
}

function filterBlocking(violations) {
  return violations.filter((v) => BLOCKING_IMPACTS.has(v.impact));
}

function formatViolation(v) {
  const nodeCount = v.nodes?.length ?? 0;
  return `| ${v.id} | ${v.impact} | ${nodeCount} | ${v.help} |`;
}

function buildRouteSection(label, url, violations) {
  const rows = violations.map(formatViolation).join('\n');
  const body = violations.length
    ? `| Rule | Impact | Nodes | Help |\n|------|--------|-------|------|\n${rows}`
    : 'No critical or serious violations.';
  return `### ${label} (\`${url}\`)\n\n${body}\n`;
}

function buildSummary(results) {
  const rows = results
    .map(({ label, violations }) => `| ${label} | ${violations.length} |`)
    .join('\n');
  return `| Route | Blocking violations |\n|-------|---------------------|\n${rows}`;
}

async function main() {
  const results = [];
  for (const route of ROUTES) {
    const url = BASE_URL + route.slug;
    try {
      const payload = await runAxe(url);
      const blocking = filterBlocking(payload?.[0]?.violations ?? payload?.violations ?? []);
      results.push({ label: route.label, url, violations: blocking });
    } catch (err) {
      results.push({
        label: route.label,
        url,
        violations: [
          {
            id: 'runner-error',
            impact: 'critical',
            nodes: [],
            help: `axe runner failed: ${err.message.split('\n')[0]}`,
          },
        ],
      });
    }
  }

  const sections = results.map(({ label, url, violations }) =>
    buildRouteSection(label, url, violations),
  );
  const md = [
    '# A11y audit results',
    '',
    `_Generated from \`${BASE_URL}\` at ${new Date().toISOString()}._`,
    '',
    '## Summary',
    '',
    buildSummary(results),
    '',
    '## Per-route findings',
    '',
    ...sections,
  ].join('\n');

  await mkdir(path.dirname(OUT_PATH), { recursive: true });
  await writeFile(OUT_PATH, md, 'utf8');
  const total = results.reduce((acc, r) => acc + r.violations.length, 0);
  console.log(`Wrote ${OUT_PATH} (${total} blocking violation(s) across ${results.length} routes)`);
  process.exit(total === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
