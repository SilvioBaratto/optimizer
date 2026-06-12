// Request-side contract assertions over an Angular `TestRequest` (issue #998).
//
// The sibling `contract-field-parity.ts` asserts RESPONSE DTO shapes against
// committed JSON-schema snapshots; this module is the missing REQUEST half:
// per-domain `*-contracts.spec.ts` specs feed each captured `TestRequest`
// (from `HttpTestingController.expectOne`) to these helpers to assert the HTTP
// method, the exact interpolated URL, query-parameter placement, and the body
// — field by field, without duplicating assertion logic across specs.
//
// All helpers are jasmine-free (they throw plain `Error`) so this file compiles
// under the production type-check like its siblings. `TestRequest` is a
// type-only import, erased at build time — no testing module reaches the bundle.

import type { TestRequest } from '@angular/common/http/testing';

/** JS runtime type of a value, distinguishing arrays and null from objects. */
function jsTypeOf(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return jsTypeOf(value) === 'object';
}

function asObject(value: unknown): Record<string, unknown> {
  return isPlainObject(value) ? value : {};
}

/** Assert the request used the expected HTTP method. */
export function assertMethod(req: TestRequest, expected: string): void {
  const actual = req.request.method;
  if (actual !== expected) {
    throw new Error(`method mismatch — expected ${expected}, got ${actual}`);
  }
}

/**
 * Assert the request path equals `expected` and carries no un-interpolated
 * `{token}` placeholder. Callers build `expected` with `encodeURIComponent`, so
 * exact equality also proves path parameters were properly encoded.
 */
export function assertUrl(req: TestRequest, expected: string): void {
  const actual = req.request.url;
  if (/\{[^}]*\}/.test(actual)) {
    throw new Error(`raw path-param token left in URL — ${actual}`);
  }
  if (actual !== expected) {
    throw new Error(`URL mismatch — expected ${expected}, got ${actual}`);
  }
}

/**
 * Assert every expected query parameter is carried in `req.request.params`
 * (and NOT in the body). Values compare as strings, matching `HttpParams`.
 */
export function assertQueryParams(
  req: TestRequest,
  expected: Record<string, string | number | boolean>,
): void {
  for (const [key, value] of Object.entries(expected)) {
    assertParamPlacement(req, key, value);
  }
}

function assertParamPlacement(
  req: TestRequest,
  key: string,
  value: string | number | boolean,
): void {
  const body = req.request.body;
  if (isPlainObject(body) && key in body) {
    throw new Error(`query param '${key}' placed in body, not params`);
  }
  const actual = req.request.params.get(key);
  if (actual !== String(value)) {
    throw new Error(`query param '${key}' mismatch — expected ${String(value)}, got ${String(actual)}`);
  }
}

/** Assert the request body's key-set is exactly `expected` (no extra/missing). */
export function assertBodyKeys(req: TestRequest, expected: string[]): void {
  const actual = Object.keys(asObject(req.request.body)).sort();
  const want = [...expected].sort();
  if (actual.length !== want.length || actual.some((k, i) => k !== want[i])) {
    throw new Error(`body key-set mismatch — expected [${want}], got [${actual}]`);
  }
}

/** Assert each named body field's JS type matches `expected` (key → JS type). */
export function assertBodyFieldTypes(
  req: TestRequest,
  expected: Record<string, string>,
): void {
  const body = asObject(req.request.body);
  const bad = Object.entries(expected)
    .filter(([k, t]) => jsTypeOf(body[k]) !== t)
    .map(([k]) => k);
  if (bad.length) {
    throw new Error(`body field type mismatch — ${bad.map((k) => `[${k}]`).join(', ')}`);
  }
}
