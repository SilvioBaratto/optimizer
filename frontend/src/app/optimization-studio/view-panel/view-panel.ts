import {
  Component,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
  DestroyRef,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HelpTooltipComponent } from '../../shared/help-tooltip/help-tooltip';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { OptimizationService } from '../optimization.service';
import { FormatService } from '../../core/services/format.service';
import type {
  GenerateViewsResponse,
  EntropyPoolingResponse,
  OpinionPoolResponse,
} from '../../core/models/optimization.model';

export type ViewFramework = 'black_litterman' | 'opinion_pool' | 'entropy_pooling';

const VIEW_FRAMEWORK_LABELS: Record<ViewFramework, string> = {
  black_litterman: 'Black-Litterman (LLM-generated views)',
  opinion_pool: 'Opinion Pool (multi-persona)',
  entropy_pooling: 'Entropy Pooling (distribution tilt)',
};

interface ViewRow extends Record<string, unknown> {
  index: string;
  assetExposure: string;
  targetReturn: string;
  confidence: string;
}

function parseTickers(raw: string): string[] {
  return raw
    .split(',')
    .map((s) => s.trim().toUpperCase())
    .filter((s) => s.length > 0);
}

function parseLines(raw: string): string[] {
  return raw
    .split('\n')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function formatMatrixRow(row: number[], tickers: string[]): string {
  return row
    .map((w, i) => (Math.abs(w) > 1e-9 ? `${w > 0 ? '+' : ''}${w.toFixed(2)}·${tickers[i]}` : null))
    .filter((x): x is string => x !== null)
    .join(' ');
}

@Component({
  selector: 'app-view-panel',
  imports: [FormsModule, HelpTooltipComponent, DataTableComponent],
  templateUrl: './view-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ViewPanelComponent {
  private readonly optimization = inject(OptimizationService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly fmt = inject(FormatService);

  readonly viewFrameworkOptions: ViewFramework[] = [
    'black_litterman',
    'opinion_pool',
    'entropy_pooling',
  ];
  readonly viewFrameworkLabels = VIEW_FRAMEWORK_LABELS;

  readonly selectedViewFramework = signal<ViewFramework>('black_litterman');

  // Shared state
  readonly tickersRaw = signal<string>('AAPL, MSFT, GOOGL, AMZN, NVDA');
  readonly startDate = signal<string>(this.fmt.defaultStartIso());
  readonly endDate = signal<string>(this.fmt.todayIso());

  // BL views
  readonly blLoading = signal<boolean>(false);
  readonly blError = signal<string | null>(null);
  readonly blResponse = signal<GenerateViewsResponse | null>(null);

  // Opinion pool
  readonly personasRaw = signal<string>('VALUE_INVESTOR, MOMENTUM_TRADER, MACRO_ANALYST');
  readonly opinionLoading = signal<boolean>(false);
  readonly opinionError = signal<string | null>(null);
  readonly opinionResponse = signal<OpinionPoolResponse | null>(null);

  // Entropy pooling
  readonly meanViewsRaw = signal<string>('');
  readonly varianceViewsRaw = signal<string>('');
  readonly cvarViewsRaw = signal<string>('');
  readonly entropyLoading = signal<boolean>(false);
  readonly entropyError = signal<string | null>(null);
  readonly entropyResponse = signal<EntropyPoolingResponse | null>(null);

  readonly entropyHasViews = computed<boolean>(
    () =>
      this.meanViewsRaw().trim().length > 0 ||
      this.varianceViewsRaw().trim().length > 0 ||
      this.cvarViewsRaw().trim().length > 0,
  );

  readonly viewTableColumns: TableColumn[] = [
    { key: 'index', label: '#' },
    { key: 'assetExposure', label: 'Exposure' },
    { key: 'targetReturn', label: 'Target return', align: 'right' },
    { key: 'confidence', label: 'Confidence', align: 'right' },
  ];

  readonly opinionTableColumns: TableColumn[] = [
    { key: 'persona', label: 'Persona' },
    { key: 'nViews', label: 'Views', align: 'right' },
    { key: 'icWeight', label: 'IC weight', align: 'right' },
  ];

  readonly blViewRows = computed<ViewRow[]>(() => {
    const res = this.blResponse();
    if (!res) return [];
    return res.p.map((row, i) => ({
      index: `${i + 1}`,
      assetExposure: formatMatrixRow(row, res.tickersWithData),
      targetReturn: `${(res.q[i] * 100).toFixed(2)}%`,
      confidence: `${(res.viewConfidences[i] * 100).toFixed(0)}%`,
    }));
  });

  readonly opinionViewRows = computed<Record<string, unknown>[]>(() => {
    const res = this.opinionResponse();
    if (!res) return [];
    return res.experts.map((e) => ({
      persona: e.persona,
      nViews: `${e.nViews}`,
      icWeight: `${(e.icWeight * 100).toFixed(0)}%`,
    }));
  });

  readonly personaWeightsEntries = computed<[string, number][]>(() => {
    const res = this.opinionResponse();
    return res ? res.experts.map((e) => [e.persona, e.icWeight]) : [];
  });

  readonly entropyMuEntries = computed(() => {
    const res = this.entropyResponse();
    if (!res) return [];
    return res.tickers.map((ticker, i) => ({ ticker, mu: res.mu[i] }));
  });

  setViewFramework(value: ViewFramework): void {
    this.selectedViewFramework.set(value);
  }

  runBlackLitterman(): void {
    const tickers = parseTickers(this.tickersRaw());
    if (tickers.length === 0) return;
    this.blLoading.set(true);
    this.blError.set(null);
    this.optimization
      .generateViews({ tickers })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.blResponse.set(res);
          this.blLoading.set(false);
        },
        error: (err: Error) => {
          this.blError.set(err.message ?? '/views/generate failed');
          this.blLoading.set(false);
        },
      });
  }

  runOpinionPool(): void {
    const tickers = parseTickers(this.tickersRaw());
    const personas = parseTickers(this.personasRaw());
    if (tickers.length === 0 || personas.length < 2) return;
    this.opinionLoading.set(true);
    this.opinionError.set(null);
    this.optimization
      .opinionPool({ tickers, personas })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.opinionResponse.set(res);
          this.opinionLoading.set(false);
        },
        error: (err: Error) => {
          this.opinionError.set(err.message ?? '/views/opinion-pool failed');
          this.opinionLoading.set(false);
        },
      });
  }

  runEntropyPooling(): void {
    const tickers = parseTickers(this.tickersRaw());
    if (tickers.length === 0) return;
    this.entropyLoading.set(true);
    this.entropyError.set(null);
    this.optimization
      .entropyPooling({
        tickers,
        start_date: this.startDate(),
        end_date: this.endDate(),
        mean_views: parseLines(this.meanViewsRaw()),
        variance_views: parseLines(this.varianceViewsRaw()),
        cvar_views: parseLines(this.cvarViewsRaw()),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.entropyResponse.set(res);
          this.entropyLoading.set(false);
        },
        error: (err: Error) => {
          this.entropyError.set(err.message ?? '/views/entropy-pooling failed');
          this.entropyLoading.set(false);
        },
      });
  }

}
