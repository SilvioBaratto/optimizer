import { defineConfig, devices } from '@playwright/test';

// Journeys exercise the DOCKER stack (frontend :4300 → nginx → api :8000),
// NOT the dev servers (4200/8000). The nginx `/api/` proxy is part of the
// system under test.
const BASE_URL = process.env.E2E_BASE_URL ?? 'http://localhost:4300';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  globalSetup: './e2e/harness/global-setup.ts',
  globalTeardown: './e2e/harness/global-teardown.ts',
  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
  },
  // Boots the real compose stack from the repo root. globalSetup then polls
  // api :8005/health and seeds the DB. `reuseExistingServer` keeps a locally
  // running stack alive between runs; CI always boots fresh.
  webServer: {
    command: 'docker compose up -d db api frontend',
    cwd: '..',
    url: BASE_URL,
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
