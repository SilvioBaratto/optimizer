import type {
  BuilderTrade,
  DriftDiagnostics,
  DriftRow,
  DriftTotals,
} from '../../core/models/drift.model';
import type { EfficientFrontierPoint } from '../../core/models/optimization.model';

export type BuilderResultStatus =
  | 'idle'
  | 'running'
  | 'ok'
  | 'error'
  | 'stale';

export interface BuilderResult {
  readonly runId: string;
  readonly weights: Record<string, number>;
  readonly metrics: Record<string, unknown>;
  readonly riskContributions?: Record<string, number>;
  readonly optimizerType: string;
  readonly createdAt: string;
}

export interface BuilderFrontier {
  readonly points: EfficientFrontierPoint[];
}

export interface BuilderBacktest {
  readonly runId: string;
  readonly summaryStats: Record<string, number | null>;
  readonly equityCurve: Record<string, number | null>;
  readonly createdAt: string;
}

export type BuilderDriftStatus = BuilderResultStatus;

export interface BuilderDrift {
  readonly drift: readonly DriftRow[];
  readonly totals: DriftTotals;
  readonly portfolioName: string;
}

export interface BuilderTrades {
  readonly trades: readonly BuilderTrade[];
}

export interface BuilderDriftDiagnostics {
  readonly diagnostics: DriftDiagnostics;
  readonly requestId: number;
}
