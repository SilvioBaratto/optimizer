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
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { JobProgressTrackerComponent } from '../../shared/job-progress-tracker/job-progress-tracker';
import { FormatService } from '../../services/format.service';
import { FactorsService } from '../../services/factors.service';

import { RegimePanelComponent } from './regime-panel';
import { TaaPanelComponent } from './taa-panel';
import { FactorAnalysisPanelComponent } from './factor-analysis-panel';
import { CmaBuilderPanelComponent } from './cma-builder-panel';
import { AssetScreenerPanelComponent } from './asset-screener-panel';

import type {
  FactorQuintileSpreadApiResponse,
  FactorRegimeTiltApiResponse,
  FactorValidateResponse,
  MacroCalibrationResponse,
  RegimeHistoryApiResponse,
  TradingEconomicsObservation,
} from '../../models/factor.model';

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V'];
const DEFAULT_FACTOR = 'momentum_12_1';

@Component({
  selector: 'app-factor-research',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    JobProgressTrackerComponent,
    RegimePanelComponent,
    TaaPanelComponent,
    FactorAnalysisPanelComponent,
    CmaBuilderPanelComponent,
    AssetScreenerPanelComponent,
  ],
  templateUrl: './factor-research.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FactorResearchComponent {
  private readonly fmt = inject(FormatService);
  private readonly factors = inject(FactorsService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  readonly activeTab = signal<string>('regime');

  // Form state
  readonly tickers = signal<string[]>(DEFAULT_TICKERS);
  readonly selectedFactor = signal<string>(DEFAULT_FACTOR);
  readonly startDate = signal<string>(this.defaultStart());
  readonly endDate = signal<string>(this.todayIso());
  readonly country = signal<string>('USA');

  // Compute (async) state
  readonly computeJobId = signal<string | null>(null);
  readonly computeError = signal<string | null>(null);

  // Response signals
  readonly validateReport = signal<FactorValidateResponse | null>(null);
  readonly quintileSpread = signal<FactorQuintileSpreadApiResponse | null>(null);
  readonly regimeTiltResult = signal<FactorRegimeTiltApiResponse | null>(null);
  readonly regimeHistory = signal<RegimeHistoryApiResponse | null>(null);
  readonly macroCalibration = signal<MacroCalibrationResponse | null>(null);
  readonly teObservations = signal<TradingEconomicsObservation[]>([]);

  // Per-panel error state
  readonly panelErrors = signal<Record<string, string | null>>({});

  readonly tabs: Tab[] = [
    { id: 'regime', label: 'Regime Detection' },
    { id: 'taa', label: 'TAA Signals' },
    { id: 'factor-analysis', label: 'Factor Analysis' },
    { id: 'cma-builder', label: 'CMA Builder' },
    { id: 'screener', label: 'Asset Screener' },
  ];

  readonly currentRegime = computed(() => {
    const pts = this.regimeHistory()?.points ?? [];
    const last = pts[pts.length - 1];
    return last?.regime ?? '—';
  });

  readonly regimeConfidence = computed(() => {
    const pts = this.regimeHistory()?.points ?? [];
    const last = pts[pts.length - 1];
    if (!last) return '—';
    const max = Math.max(last.bullProb, last.bearProb, last.sidewaysProb, last.volatileProb);
    return this.fmt.formatPercent(max);
  });

  readonly tiltedGroupsCount = computed(() => {
    const tilts = this.regimeTiltResult()?.tilt_multipliers ?? {};
    return Object.values(tilts).filter((m) => Math.abs(m - 1) > 1e-6).length;
  });

  readonly macroDelta = computed(() => {
    const cal = this.macroCalibration();
    return cal ? cal.delta.toFixed(2) : '—';
  });

  ngOnInit(): void {
    this.fetchRegimeHistory();
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
    this.computeJobId.set(null);
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

  onRunRegimeTilt(groupWeights: Record<string, number>): void {
    this.factors
      .regimeTilt({ group_weights: groupWeights })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.regimeTiltResult.set(res),
        error: (err: Error) =>
          this.setPanelError('regime-tilt', err.message ?? 'Regime tilt failed'),
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

  private fetchRegimeHistory(): void {
    this.factors
      .regimeHistory({})
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.regimeHistory.set(res),
        error: (err: Error) =>
          this.setPanelError('regime-history', err.message ?? 'Regime history failed'),
      });
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

  private setPanelError(key: string, message: string): void {
    this.panelErrors.update((prev) => ({ ...prev, [key]: message }));
  }

  private defaultStart(): string {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 3);
    return d.toISOString().slice(0, 10);
  }

  private todayIso(): string {
    return new Date().toISOString().slice(0, 10);
  }
}
