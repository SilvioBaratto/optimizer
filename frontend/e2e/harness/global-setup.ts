import type { FullConfig } from '@playwright/test';

import { seedDatabase } from './db-seed';

const HEALTH_URL = process.env.E2E_API_HEALTH ?? 'http://localhost:8005/health';
const HEALTH_TIMEOUT_MS = 60_000;
const POLL_INTERVAL_MS = 2_000;

function delay(ms: number): Promise<void> {
  return new Promise((done) => setTimeout(done, ms));
}

async function isHealthy(url: string): Promise<boolean> {
  try {
    return (await fetch(url)).ok;
  } catch {
    return false;
  }
}

async function waitForHealth(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await isHealthy(url)) return;
    await delay(POLL_INTERVAL_MS);
  }
  throw new Error(`api never became healthy at ${url} within ${timeoutMs}ms`);
}

export default async function globalSetup(_config: FullConfig): Promise<void> {
  await waitForHealth(HEALTH_URL, HEALTH_TIMEOUT_MS);
  seedDatabase();
}
