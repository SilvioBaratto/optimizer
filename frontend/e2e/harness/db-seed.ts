import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// frontend/e2e/harness → repo root is three levels up.
const REPO_ROOT = resolve(__dirname, '../../..');
const PRICES_SQL = 'api/tests/fixtures/smoke_prices.sql';

// One deterministic Portfolio + PortfolioSnapshot. Guarded by ON CONFLICT so
// the seed is idempotent across retried runs (mirrors smoke_prices.sql).
const PORTFOLIO_SQL = `
BEGIN;
INSERT INTO portfolios
  (id, name, currency, benchmark_ticker, is_active, created_at, updated_at)
VALUES
  ('22222222-2222-2222-2222-222222222222', 'e2e-portfolio', 'EUR', 'SPY',
   true, now(), now())
ON CONFLICT (name) DO NOTHING;
INSERT INTO portfolio_snapshots
  (id, portfolio_id, snapshot_date, snapshot_type, holding_count,
   created_at, updated_at)
VALUES
  ('33333333-3333-3333-3333-333333333333',
   '22222222-2222-2222-2222-222222222222',
   DATE '2023-12-29', 'optimization', 5, now(), now())
ON CONFLICT (portfolio_id, snapshot_date, snapshot_type) DO NOTHING;
COMMIT;
`;

function runPsql(sql: string): void {
  execFileSync(
    'docker',
    ['compose', 'exec', '-T', 'db', 'psql', '-U', 'postgres', '-d', 'optimizer_db'],
    { cwd: REPO_ROOT, input: sql, stdio: ['pipe', 'inherit', 'inherit'] },
  );
}

/** Seed the deterministic price fixture + one portfolio/snapshot into the DB. */
export function seedDatabase(): void {
  runPsql(readFileSync(resolve(REPO_ROOT, PRICES_SQL), 'utf8'));
  runPsql(PORTFOLIO_SQL);
}
