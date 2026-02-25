import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HelpTooltipComponent } from '../../shared/help-tooltip/help-tooltip';
import { EchartsHeatmapComponent } from '../../shared/echarts-heatmap/echarts-heatmap';
import { EchartsBarComponent } from '../../shared/echarts-bar/echarts-bar';
import { BarData } from '../../shared/bar-chart/bar-chart';
import { MuEstimatorType, CovEstimatorType } from '../../models/optimization.model';
import { MOCK_MOMENT_OUTPUT } from '../../mocks/optimization-mocks';

const MU_ESTIMATOR_LABELS: Record<MuEstimatorType, string> = {
  empirical: 'Empirical Mean',
  shrunk: 'Shrinkage Estimator',
  ew: 'Exponentially Weighted',
  equilibrium: 'CAPM Equilibrium',
  hmm_blended: 'HMM Regime Blended',
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
  hmm_blended: 'HMM Regime Blended',
};

@Component({
  selector: 'app-moment-panel',
  imports: [FormsModule, HelpTooltipComponent, EchartsHeatmapComponent, EchartsBarComponent],
  templateUrl: './moment-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MomentPanelComponent {
  readonly muEstimatorOptions: MuEstimatorType[] = ['empirical', 'shrunk', 'ew', 'equilibrium', 'hmm_blended'];
  readonly covEstimatorOptions: CovEstimatorType[] = [
    'empirical', 'ledoit_wolf', 'oas', 'shrunk', 'ew',
    'gerber', 'graphical_lasso_cv', 'denoise', 'detone', 'implied', 'hmm_blended',
  ];
  readonly muEstimatorLabels = MU_ESTIMATOR_LABELS;
  readonly covEstimatorLabels = COV_ESTIMATOR_LABELS;

  muEstimator = signal<MuEstimatorType>(MOCK_MOMENT_OUTPUT.muEstimator);
  covEstimator = signal<CovEstimatorType>(MOCK_MOMENT_OUTPUT.covEstimator);

  readonly heatmapAssets = computed(() =>
    MOCK_MOMENT_OUTPUT.expectedReturns.slice(0, 10).map(e => e.ticker),
  );

  readonly heatmapMatrix = computed(() => {
    const assets = this.heatmapAssets();
    const n = assets.length;
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (__, j) => {
        if (i === j) return 1;
        const seed = (i * 31 + j * 17) % 100;
        return +((seed / 100) * 0.8 - 0.2).toFixed(2);
      }),
    );
  });

  readonly expectedReturnsBarData = computed<BarData[]>(() =>
    MOCK_MOMENT_OUTPUT.expectedReturns.map(e => ({ label: e.ticker, value: e.value })),
  );

  setMuEstimator(value: string): void {
    this.muEstimator.set(value as MuEstimatorType);
  }

  setCovEstimator(value: string): void {
    this.covEstimator.set(value as CovEstimatorType);
  }
}
