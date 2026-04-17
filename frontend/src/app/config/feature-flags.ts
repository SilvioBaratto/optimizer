import { InjectionToken } from '@angular/core';

/**
 * Compile-time feature flags.
 *
 * Flags are exposed both as module-level constants and as an Angular
 * `InjectionToken` so components can `inject(FEATURE_AI_DECISION_LOG_TOKEN)`
 * and tests can override the value via `TestBed.configureTestingModule`.
 *
 * Default values are flipped only by editing this file and redeploying.
 * There is no runtime toggle — YAGNI until a real decision-log API lands.
 */

/**
 * Gates the `overview`, `history`, and `agents` tabs of the AI Control Room.
 * Off until a real decision-log backend exists. When off, only the `pipeline`
 * tab is visible.
 */
export const FEATURE_AI_DECISION_LOG = false;

export const FEATURE_AI_DECISION_LOG_TOKEN = new InjectionToken<boolean>(
  'FEATURE_AI_DECISION_LOG',
  {
    providedIn: 'root',
    factory: () => FEATURE_AI_DECISION_LOG,
  },
);
