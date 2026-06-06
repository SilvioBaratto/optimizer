import {
  DestroyRef,
  InjectionToken,
  Injectable,
  Injector,
  effect,
  inject,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { type Observable, Subject, defer, merge, of } from 'rxjs';
import {
  catchError,
  debounceTime,
  filter,
  last,
  map,
  switchMap,
  takeWhile,
  tap,
} from 'rxjs/operators';

export const BUILDER_RESULT_DEBOUNCE_MS = new InjectionToken<number>(
  'BUILDER_RESULT_DEBOUNCE_MS',
  { providedIn: 'root', factory: () => 500 },
);

import { JOB_POLL_TICK } from '../shared/job-progress-tracker/job-progress-tracker';
// TEMPORARY CROSS-FEATURE EDGE (Cycle 5 debt)
import type {
  BacktestApiRequest,
  BacktestProgressResponse,
  BacktestRunResponse,
} from '../backtesting/backtest.model';
import type {
  OptimizationRunResponse,
  OptimizeRequest,
} from '../models/optimization.model';
import type {
  PipelineStepId,
  RunLevelConfig,
} from '../models/pipeline-builder.model';
// TEMPORARY CROSS-FEATURE EDGE (Cycle 5 debt)
import { BacktestService } from '../backtesting/backtest.service';
import { OptimizationService } from '../optimization-studio/optimization.service';
import { BuilderStore } from './state/builder.store';
import type {
  BuilderBacktest,
  BuilderFrontier,
  BuilderResult,
} from './models/builder-result.model';

type StepResultEntry = Record<string, unknown> | null;

interface ResultPacket {
  readonly seq: number;
  readonly opt: OptimizationRunResponse | null;
  readonly bt: BacktestRunResponse | null;
  readonly error: Error | null;
}

function extractTickers(
  stepResults: Map<PipelineStepId, StepResultEntry>,
): string[] {
  const fromScreen = readTickers(stepResults.get('screen'));
  if (fromScreen) return fromScreen;
  const fromLoad = readTickers(stepResults.get('load'));
  if (fromLoad) return fromLoad;
  throw new Error('Tickers unavailable: run "load" or "screen" step first.');
}

function readTickers(entry: StepResultEntry | undefined): string[] | null {
  if (!entry) return null;
  const raw = (entry as { tickers?: unknown }).tickers;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  return raw.filter((t): t is string => typeof t === 'string');
}

function buildOptRequest(
  config: RunLevelConfig,
  tickers: string[],
): OptimizeRequest {
  return {
    tickers,
    start_date: config.start_date ?? '',
    end_date: config.end_date ?? '',
    optimizer_type: 'mean_risk',
    config: {},
    constraints: null,
  };
}

function buildBacktestRequest(
  config: RunLevelConfig,
  tickers: string[],
): BacktestApiRequest {
  return {
    tickers,
    start_date: config.start_date ?? '',
    end_date: config.end_date ?? '',
    pipeline_config: {},
  };
}

function mapToBuilderResult(r: OptimizationRunResponse): BuilderResult {
  return {
    runId: r.id,
    weights: r.weights,
    metrics: r.metrics,
    riskContributions: r.riskContributions,
    optimizerType: r.optimizerType,
    createdAt: r.createdAt,
  };
}

function mapToBuilderFrontier(
  r: OptimizationRunResponse,
): BuilderFrontier | null {
  return r.efficientFrontier ? { points: r.efficientFrontier } : null;
}

function mapToBuilderBacktest(b: BacktestRunResponse): BuilderBacktest {
  return {
    runId: b.id,
    summaryStats: b.summaryStats,
    equityCurve: b.equityCurve,
    createdAt: b.createdAt,
  };
}

function isTerminal<T extends { status: string }>(r: T): boolean {
  return r.status === 'completed' || r.status === 'failed';
}

@Injectable()
export class BuilderResultService {
  private readonly store = inject(BuilderStore);
  private readonly destroyRef = inject(DestroyRef);
  private readonly injector = inject(Injector);
  private readonly optSvc = inject(OptimizationService);
  private readonly btSvc = inject(BacktestService);
  private readonly tick$ = inject(JOB_POLL_TICK) as Observable<unknown>;
  private readonly debounceMs = inject(BUILDER_RESULT_DEBOUNCE_MS);
  private readonly _trigger$ = new Subject<'explicit'>();
  private readonly _configChange$ = new Subject<RunLevelConfig | null>();
  private _initialized = false;
  private _seq = 0;

  init(): void {
    if (this._initialized) return;
    this._initialized = true;
    this._registerConfigEffect();
    const auto$ = this._configChange$.pipe(
      filter(() => this.store.sessionId() !== null),
      tap(() => this._markStaleIfOk()),
      debounceTime(this.debounceMs),
      map(() => 'auto' as const),
    );
    merge(this._trigger$, auto$)
      .pipe(
        filter(() => this._guardsPass()),
        map(() => ++this._seq),
        tap(() => this.store.setResultStatus('running')),
        switchMap((seq) => this._runChain(seq)),
        filter((packet) => packet.seq === this._seq),
        takeUntilDestroyed(this.destroyRef),
      )
      // subscribe-no-error: _runChain() has its own catchError, so every
      // emission is a packet (success or error-encoded); the stream never errors.
      .subscribe((packet) => this._applyPacket(packet));
  }

  private _registerConfigEffect(): void {
    let previous = this.store.config();
    effect(
      () => {
        const curr = this.store.config();
        if (curr === previous) return;
        previous = curr;
        this._configChange$.next(curr);
      },
      { injector: this.injector },
    );
  }

  private _markStaleIfOk(): void {
    if (this.store.resultStatus() === 'ok') {
      this.store.setResultStatus('stale');
    }
  }

  runExplicit(): void {
    this._trigger$.next('explicit');
  }

  private _guardsPass(): boolean {
    return this.store.sessionId() !== null && this.store.config() !== null;
  }

  private _runChain(seq: number): Observable<ResultPacket> {
    return this._runOptimize().pipe(
      switchMap((opt) =>
        this._runBacktest().pipe(
          map((bt): ResultPacket => ({ seq, opt, bt, error: null })),
        ),
      ),
      catchError((err: unknown) =>
        of<ResultPacket>({
          seq,
          opt: null,
          bt: null,
          error: err instanceof Error ? err : new Error(String(err)),
        }),
      ),
    );
  }

  private _applyPacket(packet: ResultPacket): void {
    if (packet.error || !packet.opt || !packet.bt) {
      this.store.setResultStatus('error');
      return;
    }
    this.store.setResult(mapToBuilderResult(packet.opt));
    this.store.setFrontier(mapToBuilderFrontier(packet.opt));
    this.store.setBacktest(mapToBuilderBacktest(packet.bt));
    this.store.setResultStatus('ok');
  }

  private _runOptimize(): Observable<OptimizationRunResponse> {
    return defer(() => this.optSvc.optimize(this._buildOptRequest())).pipe(
      switchMap((resp) =>
        OptimizationService.isAsyncResponse(resp)
          ? this._pollOptimize(resp.run_id)
          : of(resp),
      ),
    );
  }

  private _pollOptimize(runId: string): Observable<OptimizationRunResponse> {
    return this.tick$.pipe(
      switchMap(() => this.optSvc.getOptimizationRun(runId)),
      takeWhile((r) => !isTerminal(r), true),
      last(),
      map((r) => {
        if (r.status === 'failed') {
          throw new Error(r.errorMessage ?? 'optimize failed');
        }
        return r;
      }),
    );
  }

  private _runBacktest(): Observable<BacktestRunResponse> {
    return defer(() => this.btSvc.runBacktest(this._buildBacktestRequest())).pipe(
      switchMap((accept) =>
        this._pollBacktest(accept.jobId).pipe(
          switchMap(() => this.btSvc.getBacktestRun(accept.runId)),
        ),
      ),
    );
  }

  private _pollBacktest(jobId: string): Observable<BacktestProgressResponse> {
    return this.tick$.pipe(
      switchMap(() => this.btSvc.pollBacktest(jobId)),
      takeWhile((r) => !isTerminal(r), true),
      last(),
      map((r) => {
        if (r.status === 'failed') {
          throw new Error(r.error ?? 'backtest failed');
        }
        return r;
      }),
    );
  }

  private _buildOptRequest(): OptimizeRequest {
    return buildOptRequest(
      this.store.config() as RunLevelConfig,
      extractTickers(this.store.stepResults()),
    );
  }

  private _buildBacktestRequest(): BacktestApiRequest {
    return buildBacktestRequest(
      this.store.config() as RunLevelConfig,
      extractTickers(this.store.stepResults()),
    );
  }
}
