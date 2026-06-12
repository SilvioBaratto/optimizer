import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed, toObservable } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';

import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../shared/stat-card/stat-card';
import { JobProgressTrackerComponent } from '../shared/job-progress-tracker/job-progress-tracker';
import { FormatService } from '../core/services/format.service';
import { FactorsService } from './factors.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { TickerSeedingService } from '../core/services/ticker-seeding.service';

import { TaaPanelComponent } from './taa-panel/taa-panel';
import { FactorAnalysisPanelComponent } from './factor-analysis-panel/factor-analysis-panel';
import { CmaBuilderPanelComponent } from './cma-builder-panel/cma-builder-panel';
import { AssetScreenerPanelComponent } from './asset-screener-panel/asset-screener-panel';
import { ScorePanelComponent } from './score-panel/score-panel';
import { SelectPanelComponent } from './select-panel/select-panel';
import { ExposureConstraintsPanelComponent } from './exposure-constraints-panel/exposure-constraints-panel';

import type {
  CMASet,
  FactorExposureConstraintsApiResponse,
  FactorExposureConstraintsRequest,
  FactorICReport,
  FactorQuintileSpreadApiResponse,
  FactorReturnSeries,
  FactorScoreApiResponse,
  FactorScoreRequest,
  FactorSelectApiResponse,
  FactorSelectRequest,
  FactorValidateResponse,
  TAASignal,
  TradingEconomicsObservation,
} from './factor.model';
import type { MacroCalibrationResponse } from '../core/models/macro-intelligence.model';

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V'];
const DEFAULT_FACTOR = 'momentum_12_1';

/** Order-insensitive equality check for two string arrays. */
function arraysEqualUnordered(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const aSorted = [...a].sort();
  const bSorted = [...b].sort();
  return aSorted.every((v, i) => v === bSorted[i]);
}

@Component({
  selector: 'app-factor-research',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    JobProgressTrackerComponent,
    TaaPanelComponent,
    FactorAnalysisPanelComponent,
    CmaBuilderPanelComponent,
    AssetScreenerPanelComponent,
    ScorePanelComponent,
    SelectPanelComponent,
    ExposureConstraintsPanelComponent,
  ],
  templateUrl: './factor-research.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FactorResearchComponent {
  private readonly fmt = inject(FormatService);
  private readonly factors = inject(FactorsService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly portfolioContext = inject(PortfolioContextService);
  private readonly tickerSeeding = inject(TickerSeedingService);

  /** Tracks the last array written by the seeding pipeline for the re-seed guard. */
  private readonly lastSeed = signal<string[] | null>(null);

  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  readonly activeTab = signal<string>('regime');

  // Form state
  readonly tickers = signal<string[]>(DEFAULT_TICKERS);
  readonly selectedFactor = signal<string>(DEFAULT_FACTOR);
  readonly startDate = signal<string>(this.fmt.defaultStartIso());
  readonly endDate = signal<string>(this.fmt.todayIso());
  readonly country = signal<string>('USA');

  // Compute (async) state
  readonly computeJobId = signal<string | null>(null);
  readonly computeError = signal<string | null>(null);

  // Compute result signals (populated from pollCompute after job completes)
  readonly taaSignals = signal<TAASignal[]>([]);
  readonly factorReturns = signal<FactorReturnSeries[]>([]);
  readonly icReports = signal<FactorICReport[]>([]);
  readonly cmaSets = signal<CMASet[]>([]);

  // Response signals
  readonly validateReport = signal<FactorValidateResponse | null>(null);
  readonly quintileSpread = signal<FactorQuintileSpreadApiResponse | null>(null);
  readonly macroCalibration = signal<MacroCalibrationResponse | null>(null);
  readonly teObservations = signal<TradingEconomicsObservation[]>([]);
  readonly scoreResult = signal<FactorScoreApiResponse | null>(null);
  readonly scoreLoading = signal<boolean>(false);
  readonly selectResult = signal<FactorSelectApiResponse | null>(null);
  readonly selectLoading = signal<boolean>(false);
  readonly exposureConstraintsResult = signal<FactorExposureConstraintsApiResponse | null>(null);
  readonly exposureConstraintsLoading = signal<boolean>(false);

  // Per-panel error state
  readonly panelErrors = signal<Record<string, string | null>>({});

  readonly tabs: Tab[] = [
    { id: 'regime', label: 'Regime Detection' },
    { id: 'taa', label: 'TAA Signals' },
    { id: 'factor-analysis', label: 'Factor Analysis' },
    { id: 'cma-builder', label: 'CMA Builder' },
    { id: 'screener', label: 'Asset Screener' },
    { id: 'score', label: 'Scoring' },
    { id: 'select', label: 'Selection' },
    { id: 'exposure-constraints', label: 'Exposure Constraints' },
  ];

  /** Active portfolio name from the global context, shown in the page header. */
  readonly portfolioName = this.portfolioContext.currentPortfolioName;

  readonly macroPhase = computed(() => this.macroCalibration()?.phase ?? '—');

  readonly macroConfidence = computed(() => {
    const cal = this.macroCalibration();
    return cal ? `${(cal.confidence * 100).toFixed(0)}%` : '—';
  });

  readonly macroDelta = computed(() => {
    const cal = this.macroCalibration();
    return cal ? cal.delta.toFixed(2) : '—';
  });

  constructor() {
    this.initTickerSeeding();
  }

  ngOnInit(): void {
    this.fetchMacroCalibration();
  }

  retry(): void {
    this.hasError.set(false);
    this.errorMessage.set('');
  }

  onRunCompute(): void {
    if (this.computeJobId()) return;
    this.computeError.set(null);
    this.factors
      .compute({
        tickers: this.tickers(),
        start_date: this.startDate(),
        end_date: this.endDate(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.computeJobId.set(res.job_id),
        error: (err: Error) => this.computeError.set(err.message ?? 'Compute failed'),
      });
  }

  onComputeCompleted(): void {
    const id = this.computeJobId();
    this.computeJobId.set(null);
    if (!id) return;
    this.factors
      .pollCompute(id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (progress) => this.applyComputeResult(progress.result),
        error: (err: Error) => this.computeError.set(err.message ?? 'Failed to load compute result'),
      });
  }

  private applyComputeResult(result: Record<string, unknown> | null): void {
    if (!result) return;
    const taaSignals = result['taa_signals'];
    if (Array.isArray(taaSignals)) this.taaSignals.set(taaSignals as TAASignal[]);
    const factorReturns = result['factor_returns'];
    if (Array.isArray(factorReturns)) this.factorReturns.set(factorReturns as FactorReturnSeries[]);
    const icReports = result['ic_reports'];
    if (Array.isArray(icReports)) this.icReports.set(icReports as FactorICReport[]);
    const cmaSets = result['cma_sets'];
    if (Array.isArray(cmaSets)) this.cmaSets.set(cmaSets as CMASet[]);
  }

  onComputeFailed(message: string): void {
    this.computeJobId.set(null);
    this.computeError.set(message || 'Compute job failed');
  }

  onValidate(factorType: string): void {
    this.factors
      .validate({
        tickers: this.tickers(),
        start_date: this.startDate(),
        end_date: this.endDate(),
        factor_type: factorType,
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.validateReport.set(res),
        error: (err: Error) => this.setPanelError('validate', err.message ?? 'Validate failed'),
      });
  }

  onQuintileSpread(factorName: string): void {
    this.factors
      .quintileSpread({
        tickers: this.tickers(),
        factor_name: factorName,
        start_date: this.startDate(),
        end_date: this.endDate(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.quintileSpread.set(res),
        error: (err: Error) =>
          this.setPanelError('quintile', err.message ?? 'Quintile spread failed'),
      });
  }

  onRunScore(payload: FactorScoreRequest): void {
    this.scoreLoading.set(true);
    this.setPanelError('score', null);
    this.factors
      .score(payload)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.scoreResult.set(res);
          this.scoreLoading.set(false);
        },
        error: (err: Error) => {
          this.scoreLoading.set(false);
          this.setPanelError('score', err.message ?? 'Score run failed');
        },
      });
  }

  onRunSelect(payload: FactorSelectRequest): void {
    this.selectLoading.set(true);
    this.setPanelError('select', null);
    this.factors
      .select(payload)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.selectResult.set(res);
          this.selectLoading.set(false);
        },
        error: (err: Error) => {
          this.selectLoading.set(false);
          this.setPanelError('select', err.message ?? 'Select run failed');
        },
      });
  }

  onRunExposureConstraints(payload: FactorExposureConstraintsRequest): void {
    this.exposureConstraintsLoading.set(true);
    this.setPanelError('exposure-constraints', null);
    this.factors
      .exposureConstraints(payload)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.exposureConstraintsResult.set(res);
          this.exposureConstraintsLoading.set(false);
        },
        error: (err: Error) => {
          this.exposureConstraintsLoading.set(false);
          this.setPanelError(
            'exposure-constraints',
            err.message ?? 'Exposure constraints run failed',
          );
        },
      });
  }

  onFetchTe(): void {
    this.factors
      .teObservations({ country: this.country(), limit: 500 })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (rows) => this.teObservations.set(rows),
        error: (err: Error) => this.setPanelError('te', err.message ?? 'TE fetch failed'),
      });
  }

  private initTickerSeeding(): void {
    toObservable(this.portfolioContext.currentPortfolioName)
      .pipe(
        switchMap((name) =>
          this.tickerSeeding.seedFromPortfolio(name, [...DEFAULT_TICKERS]),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((seeded) => this.applySeed(seeded));
  }

  private applySeed(seeded: string[]): void {
    const prior = this.lastSeed();
    const current = this.tickers();
    const isUnseeded = prior === null;
    const matchesPriorSeed = prior !== null && arraysEqualUnordered(current, prior);
    if (isUnseeded || matchesPriorSeed) {
      this.lastSeed.set(seeded);
      this.tickers.set(seeded);
    }
  }

  private fetchMacroCalibration(): void {
    this.factors
      .macroCalibration({ country: this.country() })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.macroCalibration.set(res),
        error: (err: Error) =>
          this.setPanelError('macro-calibration', err.message ?? 'Macro calibration failed'),
      });
  }

  private setPanelError(key: string, message: string | null): void {
    this.panelErrors.update((prev) => ({ ...prev, [key]: message }));
  }

}
