import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  input,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import {
  DataTableComponent,
  type TableColumn,
} from '../../shared/data-table/data-table';
import { JobProgressTrackerComponent } from '../../shared/job-progress-tracker/job-progress-tracker';
import { BacktestService } from '../../services/backtest.service';
import type {
  CvType,
  ValidateFoldResult,
  ValidateProgressResponse,
} from '../../models/backtest.model';

interface FoldRow extends Record<string, unknown> {
  fold: string;
  sharpe: string;
  annualizedReturn: string;
  volatility: string;
  maxDrawdown: string;
}

interface AggregateRow extends Record<string, unknown> {
  metric: string;
  value: string;
}

const AGGREGATE_METRICS = ['sharpe', 'annualized_return', 'volatility', 'max_drawdown'];

function fmt(value: number | undefined, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function fmtPct(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : '—';
}

@Component({
  selector: 'app-walk-forward-panel',
  imports: [DataTableComponent, JobProgressTrackerComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="space-y-4">
      <div class="bg-surface-raised border border-border rounded-lg p-4 flex items-center gap-3">
        <div class="flex-1">
          <h4 class="text-xs font-medium text-text-secondary uppercase tracking-wide">
            Walk-forward cross-validation
          </h4>
          <p class="text-xs text-text-tertiary mt-1">
            Runs a background CV job via POST /validate/cross-validation with cv_type=walk_forward.
          </p>
        </div>
        <button type="button" (click)="onRun()" [disabled]="isRunning()"
          class="px-3 py-1.5 text-xs font-medium rounded-lg bg-accent text-white hover:bg-accent/90 disabled:opacity-50 transition-colors">
          @if (isRunning()) { Running… } @else { Run walk-forward }
        </button>
      </div>

      @if (error(); as err) {
        <p class="text-xs text-loss">{{ err }}</p>
      }

      @if (jobId(); as id) {
        <div class="bg-surface-raised border border-border rounded-lg p-3">
          <app-job-progress-tracker [jobId]="id"
            (completed)="onJobCompleted()"
            (failed)="onJobFailed($event.error ?? 'Job failed')" />
        </div>
      }

      @if (foldRows().length > 0) {
        <div class="bg-surface-raised border border-border rounded-lg p-4">
          <h4 class="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">
            Per-fold metrics
          </h4>
          <app-data-table [columns]="foldColumns"
            [rows]="foldRows()"
            [total]="foldRows().length"
            [dense]="true" />
        </div>
      }

      @if (aggregateRows().length > 0) {
        <div class="bg-surface-raised border border-border rounded-lg p-4">
          <h4 class="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">
            Aggregate score
          </h4>
          <app-data-table [columns]="aggregateColumns"
            [rows]="aggregateRows()"
            [total]="aggregateRows().length"
            [dense]="true" />
        </div>
      }

      @if (!isRunning() && foldRows().length === 0 && !error()) {
        <p class="text-xs text-text-tertiary">No walk-forward run yet. Click the button to start one.</p>
      }
    </div>
  `,
})
export class WalkForwardPanelComponent {
  private readonly backtest = inject(BacktestService);
  private readonly destroyRef = inject(DestroyRef);

  readonly tickers = input<string[]>([]);
  readonly startDate = input<string>('');
  readonly endDate = input<string>('');
  readonly optimizerType = input<string>('hrp');
  readonly cvType = input<CvType>('walk_forward');

  readonly jobId = signal<string | null>(null);
  readonly folds = signal<ValidateFoldResult[]>([]);
  readonly aggregate = signal<Record<string, number>>({});
  readonly error = signal<string | null>(null);

  readonly isRunning = computed(() => this.jobId() !== null);

  readonly foldColumns: TableColumn[] = [
    { key: 'fold', label: 'Fold' },
    { key: 'sharpe', label: 'Sharpe', align: 'right' },
    { key: 'annualizedReturn', label: 'Ann. return', align: 'right' },
    { key: 'volatility', label: 'Volatility', align: 'right' },
    { key: 'maxDrawdown', label: 'Max DD', align: 'right' },
  ];

  readonly aggregateColumns: TableColumn[] = [
    { key: 'metric', label: 'Metric' },
    { key: 'value', label: 'Value', align: 'right' },
  ];

  readonly foldRows = computed<FoldRow[]>(() =>
    this.folds().map((fold, i) => ({
      fold: `#${i + 1}`,
      sharpe: fmt(fold.measures['sharpe']),
      annualizedReturn: fmtPct(fold.measures['annualized_return']),
      volatility: fmtPct(fold.measures['volatility']),
      maxDrawdown: fmtPct(fold.measures['max_drawdown']),
    })),
  );

  readonly aggregateRows = computed<AggregateRow[]>(() => {
    const agg = this.aggregate();
    return AGGREGATE_METRICS.filter((k) => typeof agg[k] === 'number').map((k) => ({
      metric: k,
      value: k.includes('drawdown') || k.includes('return') || k === 'volatility'
        ? fmtPct(agg[k])
        : fmt(agg[k]),
    }));
  });

  onRun(): void {
    if (this.isRunning()) return;
    const tickers = this.tickers();
    if (tickers.length === 0) {
      this.error.set('No tickers provided');
      return;
    }
    this.error.set(null);
    this.backtest
      .runCrossValidation({
        tickers,
        start_date: this.startDate(),
        end_date: this.endDate(),
        cv_type: this.cvType(),
        optimizer_type: this.optimizerType(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.jobId.set(res.job_id),
        error: (err: Error) => this.error.set(err.message ?? 'CV failed'),
      });
  }

  onJobCompleted(): void {
    const id = this.jobId();
    this.jobId.set(null);
    if (!id) return;
    this.backtest
      .pollCrossValidation(id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (progress) => this.applyResult(progress),
        error: (err: Error) => this.error.set(err.message ?? 'Failed to load CV result'),
      });
  }

  onJobFailed(message: string): void {
    this.jobId.set(null);
    this.error.set(message);
  }

  private applyResult(progress: ValidateProgressResponse): void {
    this.folds.set(progress.result?.folds ?? []);
    this.aggregate.set(progress.result?.aggregate ?? {});
  }
}
