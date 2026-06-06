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
import { OptimizationService } from '../optimization.service';
import type { CovEstimatorType, MuEstimatorType } from '../../models/optimization.model';

const MU_ESTIMATOR_LABELS: Record<MuEstimatorType, string> = {
  empirical: 'Empirical Mean',
  shrunk: 'Shrinkage Estimator',
  ew: 'Exponentially Weighted',
  equilibrium: 'CAPM Equilibrium',
};

const COV_ESTIMATOR_LABELS: Record<CovEstimatorType, string> = {
  empirical: 'Empirical Covariance',
  ledoit_wolf: 'Ledoit-Wolf Shrinkage',
  oas: 'OAS Shrinkage',
  shrunk: 'Shrunk Covariance',
  ew: 'Exponentially Weighted',
  gerber: 'Gerber Statistic',
  graphical_lasso_cv: 'Graphical Lasso CV',
  denoise: 'Random Matrix Denoised',
  detone: 'Detoned Covariance',
  implied: 'Implied Covariance',
};

@Component({
  selector: 'app-moment-panel',
  imports: [FormsModule, HelpTooltipComponent],
  templateUrl: './moment-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MomentPanelComponent {
  private readonly optimization = inject(OptimizationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly muEstimatorOptions: MuEstimatorType[] = [
    'empirical',
    'shrunk',
    'ew',
    'equilibrium',
  ];

  readonly covEstimatorOptions: CovEstimatorType[] = [
    'empirical',
    'ledoit_wolf',
    'oas',
    'shrunk',
    'ew',
    'gerber',
    'graphical_lasso_cv',
    'denoise',
    'detone',
    'implied',
  ];

  readonly muEstimatorLabels = MU_ESTIMATOR_LABELS;
  readonly covEstimatorLabels = COV_ESTIMATOR_LABELS;

  // ── Form state ──
  readonly muEstimator = signal<MuEstimatorType>('empirical');
  readonly covEstimator = signal<CovEstimatorType>('ledoit_wolf');

  // ── Calibrate delta (/llm-moments/calibrate-delta) ──
  readonly macroText = signal<string>('');
  readonly calibrateDeltaLoading = signal<boolean>(false);
  readonly calibrateDeltaError = signal<string | null>(null);
  readonly delta = signal<number | null>(null);
  readonly deltaRationale = signal<string | null>(null);

  // ── Adapt factor weights (/llm-moments/adapt-factor-weights) ──
  readonly factorGroupsRaw = signal<string>('value, momentum, quality');
  readonly macroIndicators = signal<string>('');
  readonly adaptLoading = signal<boolean>(false);
  readonly adaptError = signal<string | null>(null);
  readonly factorPhase = signal<string | null>(null);
  readonly factorWeights = signal<Record<string, number> | null>(null);

  // ── Select cov regime (/llm-moments/select-cov-regime) ──
  readonly newsHeadlinesRaw = signal<string>('');
  readonly avgSentiment = signal<number>(0);
  readonly realizedVol = signal<number>(0.15);
  readonly regimeLoading = signal<boolean>(false);
  readonly regimeError = signal<string | null>(null);
  readonly regimeEstimator = signal<CovEstimatorType | null>(null);
  readonly regimeRationale = signal<string | null>(null);

  readonly factorWeightsEntries = computed(() => {
    const weights = this.factorWeights();
    return weights ? Object.entries(weights) : [];
  });

  setMuEstimator(value: string): void {
    this.muEstimator.set(value as MuEstimatorType);
  }

  setCovEstimator(value: string): void {
    this.covEstimator.set(value as CovEstimatorType);
  }

  runCalibrateDelta(): void {
    const text = this.macroText().trim();
    if (!text) return;
    this.calibrateDeltaLoading.set(true);
    this.calibrateDeltaError.set(null);
    this.optimization
      .calibrateDelta({ macro_text: text })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.delta.set(res.delta);
          this.deltaRationale.set(res.rationale);
          this.calibrateDeltaLoading.set(false);
        },
        error: (err: Error) => {
          this.calibrateDeltaError.set(err.message ?? 'calibrate-delta failed');
          this.calibrateDeltaLoading.set(false);
        },
      });
  }

  runAdaptFactorWeights(): void {
    const groups = this.parseGroups(this.factorGroupsRaw());
    const indicators = this.macroIndicators().trim();
    if (groups.length === 0) return;
    if (indicators.length < 20) {
      this.adaptError.set('Macro indicators description must be at least 20 characters.');
      return;
    }
    this.adaptLoading.set(true);
    this.adaptError.set(null);
    this.optimization
      .adaptFactorWeights({ macro_indicators: indicators, factor_groups: groups })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.factorPhase.set(res.phase);
          this.factorWeights.set(res.weights);
          this.adaptLoading.set(false);
        },
        error: (err: Error) => {
          this.adaptError.set(err.message ?? 'adapt-factor-weights failed');
          this.adaptLoading.set(false);
        },
      });
  }

  runSelectCovRegime(): void {
    const headlines = this.parseLines(this.newsHeadlinesRaw());
    this.regimeLoading.set(true);
    this.regimeError.set(null);
    this.optimization
      .selectCovRegime({
        news_headlines: headlines,
        avg_sentiment_score: this.avgSentiment(),
        realized_vol_30d: this.realizedVol(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.regimeEstimator.set(res.estimator_type);
          this.regimeRationale.set(res.rationale);
          this.covEstimator.set(res.estimator_type);
          this.regimeLoading.set(false);
        },
        error: (err: Error) => {
          this.regimeError.set(err.message ?? 'select-cov-regime failed');
          this.regimeLoading.set(false);
        },
      });
  }

  private parseGroups(raw: string): string[] {
    return raw
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  private parseLines(raw: string): string[] {
    return raw
      .split('\n')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }
}
