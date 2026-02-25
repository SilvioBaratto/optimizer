import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../services/format.service';
import { MockFetchService } from '../../services/mock-fetch.service';
import { RegimePanelComponent } from './regime-panel';
import { TaaPanelComponent } from './taa-panel';
import { FactorAnalysisPanelComponent } from './factor-analysis-panel';
import { CmaBuilderPanelComponent } from './cma-builder-panel';
import { AssetScreenerPanelComponent } from './asset-screener-panel';
import {
  MOCK_REGIME_HISTORY,
  MOCK_TAA_SIGNALS,
  MOCK_FACTOR_RETURN_SERIES,
  MOCK_CMA_SETS,
  MOCK_FACTOR_IC_REPORTS,
} from '../../mocks/factor-mocks';

@Component({
  selector: 'app-factor-research',
  imports: [
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
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
  private fmt = inject(FormatService);
  private readonly mockFetch = inject(MockFetchService);

  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  activeTab = signal('regime');

  readonly regimeHistory = MOCK_REGIME_HISTORY;
  readonly taaSignals = MOCK_TAA_SIGNALS;
  readonly factorReturns = MOCK_FACTOR_RETURN_SERIES;
  readonly cmaSets = MOCK_CMA_SETS;
  readonly icReports = MOCK_FACTOR_IC_REPORTS;

  readonly tabs: Tab[] = [
    { id: 'regime', label: 'Regime Detection' },
    { id: 'taa', label: 'TAA Signals' },
    { id: 'factor-analysis', label: 'Factor Analysis' },
    { id: 'cma-builder', label: 'CMA Builder' },
    { id: 'screener', label: 'Asset Screener' },
  ];

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({
        regimeHistory: MOCK_REGIME_HISTORY,
        taaSignals: MOCK_TAA_SIGNALS,
        factorReturns: MOCK_FACTOR_RETURN_SERIES,
        cmaSets: MOCK_CMA_SETS,
        icReports: MOCK_FACTOR_IC_REPORTS,
      })
      .then(() => {
        this.isLoading.set(false);
      })
      .catch((err: Error) => {
        this.hasError.set(true);
        this.errorMessage.set(err.message);
        this.isLoading.set(false);
      });
  }

  retry(): void {
    this.loadData();
  }

  currentRegime = computed(() => {
    const last = this.regimeHistory.at(-1);
    if (!last) return '--';
    return last.state.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
  });

  regimeConfidence = computed(() => {
    const last = this.regimeHistory.at(-1);
    if (!last) return '--';
    const max = Math.max(...Object.values(last.probabilities));
    return this.fmt.formatPercent(max);
  });

  activeSignalsCount = computed(() => {
    return this.taaSignals.filter(s => Math.abs(s.tiltedWeight - s.currentWeight) > 0.0001).length;
  });

  avgTiltMagnitude = computed(() => {
    const signals = this.taaSignals;
    if (signals.length === 0) return '--';
    const totalBps = signals.reduce(
      (sum, s) => sum + Math.abs(s.tiltedWeight - s.currentWeight) * 10000,
      0,
    );
    return `${(totalBps / signals.length).toFixed(1)} bps`;
  });
}
