// Contract-parity assertions over the committed Pydantic JSON-schema snapshots
// (`contract-snapshots/<domain>.json`, emitted by the API in issue #845).
//
// A snapshot file is a map `{ ClassName: jsonSchema }`; each schema may nest the
// models it references under `$defs`. `schemaOf` resolves a class by name from
// the top level or any `$defs`. `assertParity` then requires the wire property
// set to equal the fixture's key set — so a backend rename (a wire key the
// fixture lacks) and a phantom frontend field (a fixture key the wire lacks)
// both fail the parity check. Helper is jasmine-free (it throws) so it compiles
// in the production build, which type-checks every non-spec file under src/.

export interface JsonSchema {
  properties?: Record<string, unknown>;
  $defs?: Record<string, JsonSchema>;
  [key: string]: unknown;
}

export function schemaOf(snapshot: unknown, className: string): JsonSchema {
  const map = snapshot as Record<string, JsonSchema>;
  const direct = map[className];
  if (direct) {
    return direct;
  }
  for (const top of Object.values(map)) {
    const nested = top.$defs?.[className];
    if (nested) {
      return nested;
    }
  }
  throw new Error(`Schema '${className}' not found in snapshot`);
}

export function assertParity(schema: JsonSchema, fixture: object): void {
  const wireKeys = Object.keys(schema.properties ?? {});
  if (wireKeys.length === 0) {
    throw new Error('Snapshot schema exposes no properties to compare');
  }
  const wireSet = new Set(wireKeys);
  const fixtureSet = new Set(Object.keys(fixture));
  const missing = wireKeys.filter((key) => !fixtureSet.has(key));
  const extra = Object.keys(fixture).filter((key) => !wireSet.has(key));
  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      `parity mismatch — missing in fixture: [${missing}]; extra in fixture: [${extra}]`,
    );
  }
}
