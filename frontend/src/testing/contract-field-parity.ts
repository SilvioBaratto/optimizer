// Field-level type assertions layered on top of the key-set-only `assertParity`.
// All helpers are jasmine-free (they throw) so this file compiles in the production
// build, which type-checks every non-spec file under src/.
//
// Cycle-4 usage pattern:
//   import portfolioSnapshot from './contract-snapshots/portfolio.json';
//   assertFieldParity(schemaOf(portfolioSnapshot, 'PortfolioResponse'), makePortfolioDto(), portfolioSnapshot);
//
// Pass the domain snapshot as the third argument whenever the schema uses $ref so
// that referenced types are resolved within the same snapshot document.

import type { JsonSchema } from './contract-parity';

export function resolveRef(ref: string, root: JsonSchema): JsonSchema {
  const path = ref.replace(/^#\//, '').split('/');
  let node: unknown = root;
  for (const seg of path) node = (node as Record<string, unknown>)?.[seg];
  if (node == null) throw new Error(`Cannot resolve $ref '${ref}'`);
  return node as JsonSchema;
}

export function requiredKeys(schema: JsonSchema): Set<string> {
  const req = schema['required'];
  return new Set<string>(Array.isArray(req) ? (req as string[]) : []);
}

function jsTypeOf(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function matchesSchemaType(value: unknown, type: string): boolean {
  if (type === 'integer' || type === 'number') return typeof value === 'number';
  if (type === 'object') return jsTypeOf(value) === 'object';
  return jsTypeOf(value) === type;
}

function matchesProp(value: unknown, prop: JsonSchema, root?: JsonSchema): boolean {
  if ('anyOf' in prop) {
    return (prop['anyOf'] as JsonSchema[]).some((b) => matchesProp(value, b, root));
  }
  if ('$ref' in prop && root) {
    try { assertFieldParity(resolveRef(prop['$ref'] as string, root), value as object); return true; }
    catch { return false; }
  }
  return !('type' in prop) || matchesSchemaType(value, prop['type'] as string);
}

export function assertFieldParity(schema: JsonSchema, fixture: object, root?: JsonSchema): void {
  const map = fixture as Record<string, unknown>;
  const bad = Object.entries(schema.properties ?? {})
    .filter(([k, p]) => !matchesProp(map[k], p as JsonSchema, root))
    .map(([k]) => k);
  if (bad.length) throw new Error(`field type mismatch — ${bad.map((k) => `[${k}]`).join(', ')}`);
}
