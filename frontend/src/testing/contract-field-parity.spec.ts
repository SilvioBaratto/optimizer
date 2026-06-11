/**
 * Source-blind tests for the contract-field-parity harness (issue #947).
 *
 * Every test is derived solely from the acceptance criteria:
 *   - resolveRef(ref, root)  dereferences a '#/$defs/Name' pointer against root
 *   - requiredKeys(schema)   returns the Set of schema.required keys (empty when absent)
 *   - assertFieldParity(schema, fixture, root?) throws on JS-type/schema-type mismatch;
 *     handles string/integer/number/boolean/array/object/null, anyOf (incl. null branch),
 *     $ref (one-level resolve); skips untyped properties; jasmine-free (throws, not expect)
 *
 * Assumption: these symbols are exported from './contract-field-parity' and re-exported
 * via './index'. Tests import directly from the module rather than the barrel so they
 * fail to compile (RED) if the module does not exist yet.
 */

import { resolveRef, requiredKeys, assertFieldParity } from './contract-field-parity';
import type { JsonSchema } from './contract-parity';

// ---------------------------------------------------------------------------
// Shared fixtures — built from AC only, not from production code
// ---------------------------------------------------------------------------

const PORTFOLIO_SCHEMA: JsonSchema = {
  properties: {
    id: { type: 'string' },
    name: { type: 'string' },
    description: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    currency: { type: 'string' },
    benchmark_ticker: { type: 'string' },
    is_active: { type: 'boolean' },
    created_at: { type: 'string' },
    updated_at: { type: 'string' },
  },
  required: ['id', 'name', 'currency', 'benchmark_ticker', 'is_active', 'created_at', 'updated_at'],
};

const PORTFOLIO_FIXTURE = {
  id: 'pf-1',
  name: 'Test Portfolio',
  description: null,
  currency: 'USD',
  benchmark_ticker: 'SPY',
  is_active: true,
  created_at: '2026-01-01T00:00:00.000Z',
  updated_at: '2026-01-01T00:00:00.000Z',
};

const SCHEMA_WITH_DEFS: JsonSchema = {
  $defs: {
    NestedType: {
      properties: {
        value: { type: 'number' },
        label: { type: 'string' },
      },
    },
  },
  properties: {
    nested: { $ref: '#/$defs/NestedType' },
    count: { type: 'integer' },
  },
};

// ---------------------------------------------------------------------------
// resolveRef
// ---------------------------------------------------------------------------

describe('resolveRef', () => {
  it('when ref is "#/$defs/NestedType", returns the NestedType schema from the root', () => {
    const result = resolveRef('#/$defs/NestedType', SCHEMA_WITH_DEFS);
    expect(result).toEqual(SCHEMA_WITH_DEFS.$defs!['NestedType']);
  });

  it('when ref is "#/$defs/Missing", throws an error indicating the ref was not found', () => {
    expect(() => resolveRef('#/$defs/Missing', SCHEMA_WITH_DEFS)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// requiredKeys
// ---------------------------------------------------------------------------

describe('requiredKeys', () => {
  it('when schema has a required array, returns a Set containing exactly those keys', () => {
    const result = requiredKeys(PORTFOLIO_SCHEMA);
    expect(result instanceof Set).toBe(true);
    expect(result.has('id')).toBe(true);
    expect(result.has('name')).toBe(true);
    expect(result.has('description')).toBe(false);
  });

  it('when schema has no required array, returns an empty Set', () => {
    const schema: JsonSchema = { properties: { x: { type: 'string' } } };
    const result = requiredKeys(schema);
    expect(result instanceof Set).toBe(true);
    expect(result.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// assertFieldParity — passing cases
// ---------------------------------------------------------------------------

describe('assertFieldParity — passing', () => {
  it('when every fixture field matches its schema type, no error is thrown', () => {
    expect(() => assertFieldParity(PORTFOLIO_SCHEMA, PORTFOLIO_FIXTURE)).not.toThrow();
  });

  it('when description is null and schema uses anyOf with null branch, no error is thrown', () => {
    const fixture = { ...PORTFOLIO_FIXTURE, description: null };
    expect(() => assertFieldParity(PORTFOLIO_SCHEMA, fixture)).not.toThrow();
  });

  it('when description is a string and schema uses anyOf with string branch, no error is thrown', () => {
    const fixture = { ...PORTFOLIO_FIXTURE, description: 'a description' };
    expect(() => assertFieldParity(PORTFOLIO_SCHEMA, fixture)).not.toThrow();
  });

  it('when a schema property has no type (untyped), that field is skipped and no error is thrown', () => {
    const schemaWithUntypedField: JsonSchema = {
      properties: {
        id: { type: 'string' },
        meta: {}, // untyped — must be skipped
      },
    };
    const fixture = { id: 'x', meta: { anything: true } };
    expect(() => assertFieldParity(schemaWithUntypedField, fixture)).not.toThrow();
  });

  it('when a property is a $ref and the referenced schema matches the fixture field, no error is thrown', () => {
    const fixture = { nested: { value: 1.5, label: 'ok' }, count: 3 };
    expect(() => assertFieldParity(SCHEMA_WITH_DEFS, fixture, SCHEMA_WITH_DEFS)).not.toThrow();
  });

  it('when fixture field is an array and schema type is "array", no error is thrown', () => {
    const schema: JsonSchema = { properties: { items: { type: 'array' } } };
    const fixture = { items: [1, 2, 3] };
    expect(() => assertFieldParity(schema, fixture)).not.toThrow();
  });

  it('when fixture field is a plain object and schema type is "object", no error is thrown', () => {
    const schema: JsonSchema = { properties: { data: { type: 'object' } } };
    const fixture = { data: { key: 'value' } };
    expect(() => assertFieldParity(schema, fixture)).not.toThrow();
  });

  it('when fixture field is a boolean and schema type is "boolean", no error is thrown', () => {
    const schema: JsonSchema = { properties: { flag: { type: 'boolean' } } };
    expect(() => assertFieldParity(schema, { flag: false })).not.toThrow();
  });

  it('when fixture field is a number and schema type is "number", no error is thrown', () => {
    const schema: JsonSchema = { properties: { ratio: { type: 'number' } } };
    expect(() => assertFieldParity(schema, { ratio: 3.14 })).not.toThrow();
  });

  it('when fixture field is an integer and schema type is "integer", no error is thrown', () => {
    const schema: JsonSchema = { properties: { count: { type: 'integer' } } };
    expect(() => assertFieldParity(schema, { count: 7 })).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// assertFieldParity — failure cases (must throw)
// ---------------------------------------------------------------------------

describe('assertFieldParity — failures', () => {
  it('when a string field has a number value, throws a type-mismatch error naming the field', () => {
    const mutated = { ...PORTFOLIO_FIXTURE, name: 42 as unknown as string };
    let caughtMessage = '';
    try {
      assertFieldParity(PORTFOLIO_SCHEMA, mutated);
    } catch (err) {
      caughtMessage = (err as Error).message;
    }
    expect(caughtMessage).toContain('name');
  });

  it('when a boolean field has a string value, throws a type-mismatch error naming the field', () => {
    const mutated = { ...PORTFOLIO_FIXTURE, is_active: 'yes' as unknown as boolean };
    expect(() => assertFieldParity(PORTFOLIO_SCHEMA, mutated)).toThrow();
  });

  it('when anyOf has no matching branch, throws a type-mismatch error', () => {
    // description must be string | null — giving a number should throw
    const mutated = { ...PORTFOLIO_FIXTURE, description: 999 as unknown as string };
    expect(() => assertFieldParity(PORTFOLIO_SCHEMA, mutated)).toThrow();
  });

  it('error message contains the field name that caused the mismatch', () => {
    const schema: JsonSchema = {
      properties: {
        score: { type: 'number' },
      },
    };
    const fixture = { score: 'not-a-number' };
    let message = '';
    try {
      assertFieldParity(schema, fixture);
    } catch (err) {
      message = (err as Error).message;
    }
    expect(message).toContain('score');
  });
});

// ---------------------------------------------------------------------------
// assertFieldParity is jasmine-free (throw contract)
// ---------------------------------------------------------------------------

describe('assertFieldParity — jasmine-free throw contract', () => {
  it('when called outside a Jasmine context on a mismatched fixture, it throws (not calls expect)', () => {
    // Verify the function itself throws — it does not rely on Jasmine's expect().
    // This test passes if and only if the function uses `throw new Error(...)`.
    const schema: JsonSchema = { properties: { x: { type: 'string' } } };
    const fixture = { x: 123 };
    let threw = false;
    try {
      assertFieldParity(schema, fixture);
    } catch {
      threw = true;
    }
    expect(threw).toBe(true);
  });
});
