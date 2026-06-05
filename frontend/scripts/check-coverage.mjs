#!/usr/bin/env node
// Coverage gate for the frontend. Parses the istanbul text-summary that
// `ng test --code-coverage` prints (the @angular/build:karma builder emits an
// HTML report + this console summary; no lcov/json-summary by default), and
// fails if any metric falls below the regression FLOOR.
//
// The floor is intentionally below the Cycle-3 ≥90% aspiration: current
// coverage is ~75% statements / ~51% branches because most component specs
// assert computed/output logic at the instance level without rendering
// templates, so template @if/@for branches stay uncovered. The floor blocks
// backsliding while keeping CI green; raise it as render-coverage specs land.
import { readFileSync } from 'node:fs';

const FLOOR = { Statements: 72, Branches: 48, Functions: 67, Lines: 74 };

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
