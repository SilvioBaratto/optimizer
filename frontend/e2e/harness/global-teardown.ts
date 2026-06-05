import type { FullConfig } from '@playwright/test';

// Teardown deliberately does NOT run `docker compose down` — wiping the stack
// (and its volume) would destroy data and is forbidden. Locally the stack is
// left running for fast re-runs (`reuseExistingServer`); in CI the ephemeral
// runner is discarded wholesale. Set E2E_STOP_STACK=1 to stop (not remove)
// containers without touching the volume.
export default async function globalTeardown(_config: FullConfig): Promise<void> {
  if (process.env.E2E_STOP_STACK !== '1') return;
  const { execFileSync } = await import('node:child_process');
  const { resolve } = await import('node:path');
  execFileSync('docker', ['compose', 'stop', 'db', 'api', 'frontend'], {
    cwd: resolve(__dirname, '../../..'),
    stdio: 'inherit',
  });
}
