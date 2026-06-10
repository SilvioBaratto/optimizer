#!/usr/bin/env node
// Coverage gate for the frontend. Parses the istanbul text-summary that
// `ng test --code-coverage` prints (the @angular/build:karma builder emits an
// HTML report + this console summary; no lcov/json-summary by default), and
// fails if any metric falls below the regression FLOOR.
//
// Cycle 1 (render-coverage), Cycle 2 (service-error), and Cycle 3 (#916–#920,
// function/statement edge branches) have all landed. Measured coverage is now
// ~76.7% statements / 55.8% branches / 72.9% functions / 78.9% lines (2184 specs).
// The 80/80/80/80 final target is NOT yet reachable this cycle — Branches is the
// binding constraint (~56%, ~24pp short): template @if/@for branches in untested
// page components (optimization-studio, backtest-results-panel, canvas-pane
// @switch, several shared components) remain uncovered and need a dedicated
// render-coverage effort (Cycle 4) before an 80 branch floor can go green. See
// the surfaced note on #921.
//
// The FLOOR is a strict regression ratchet: each value sits just below the
// current measured actual to lock in the Cycle 1–3 gains and block backsliding
// while keeping CI green. NEVER lower a value; raise each toward 80 only as real
// coverage lands. Raising any floor above the actual measured value (e.g. to 80
// before the suite clears it) will hard-fail CI — that is by design.
import { readFileSync } from 'node:fs';

const FLOOR = { Statements: 76, Branches: 55, Functions: 72, Lines: 78 };

const file = process.argv[2];
if (!file) {
  console.error('usage: check-coverage.mjs <ng-test-output.txt>');
  process.exit(2);
}

const text = readFileSync(file, 'utf8');
const failures = [];

for (const [metric, min] of Object.entries(FLOOR)) {
  const match = text.match(new RegExp(`${metric}\\s*:\\s*([\\d.]+)%`));
  if (!match) {
    failures.push(`${metric}: not found in coverage output`);
    continue;
  }
  const actual = Number(match[1]);
  const status = actual >= min ? 'OK' : 'FAIL';
  console.log(`${metric.padEnd(11)} ${actual.toFixed(2)}% (floor ${min}%) ${status}`);
  if (actual < min) failures.push(`${metric} ${actual}% < floor ${min}%`);
}

if (failures.length > 0) {
  console.error('\nCoverage gate failed:\n  ' + failures.join('\n  '));
  process.exit(1);
}
console.log('\nCoverage gate passed.');
