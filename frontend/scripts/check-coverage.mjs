#!/usr/bin/env node
// Coverage gate for the frontend. Parses the istanbul text-summary that
// `ng test --code-coverage` prints (the @angular/build:karma builder emits an
// HTML report + this console summary; no lcov/json-summary by default), and
// fails if any metric falls below the regression FLOOR.
//
// The FLOOR is a fixed 80/80/80/80 hard gate (Cycle 5, "Coverage Gates"): the
// build goes red if statements, branches, functions, or lines falls below 80.
// The suite already clears 80 on every metric, so this gate only blocks
// backsliding below the locked floor.
//
// The floor is a one-way ratchet: raise a value above 80 only as real coverage
// lands and holds, NEVER lower one to make CI green.
import { readFileSync } from 'node:fs';

const FLOOR = { Statements: 80, Branches: 80, Functions: 80, Lines: 80 };

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
