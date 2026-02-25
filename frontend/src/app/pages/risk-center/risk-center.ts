import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsGaugeComponent } from '../../shared/echarts-gauge/echarts-gauge';
import { FormatService } from '../../services/format.service';
import { MockFetchService } from '../../services/mock-fetch.service';
import { VarPanelComponent } from './var-panel';
import { StressPanelComponent } from './stress-panel';
import { CorrelationPanelComponent } from './correlation-panel';
import { FactorPanelComponent } from './factor-panel';
import { ConcentrationPanelComponent } from './concentration-panel';
import { LiquidityPanelComponent } from './liquidity-panel';
import { LimitsPanelComponent } from './limits-panel';
import {
  MOCK_VAR_RESULTS,
  MOCK_STRESS_SCENARIOS,
  MOCK_CORRELATION_DATA,
  MOCK_FACTOR_EXPOSURES,
  MOCK_CONCENTRATION,
  MOCK_LIQUIDITY,
  MOCK_RISK_LIMITS,
  MOCK_RISK_ALERTS,
} from '../../mocks/risk-mocks';
import type { VaRMethod, RiskAlert, RiskLimit } from '../../models/risk.model';

@Component({
  selector: 'app-risk-center',
  imports: [
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    EchartsGaugeComponent,
    VarPanelComponent,
    StressPanelComponent,
    CorrelationPanelComponent,
    FactorPanelComponent,
    ConcentrationPanelComponent,
    LiquidityPanelComponent,
    LimitsPanelComponent,
  ],
  templateUrl: './risk-center.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RiskCenterComponent {
  private fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly mockFetch = inject(MockFetchService);

  readonly isLoading = signal(true);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  activeTab = signal('var');
  varMethod = signal<VaRMethod>('historical');
  varConfidence = signal(0.95);
  correlationMethod = signal<'pearson' | 'spearman' | 'kendall'>('pearson');
  alerts = signal<RiskAlert[]>(MOCK_RISK_ALERTS);
  limits = signal<RiskLimit[]>(MOCK_RISK_LIMITS);

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);

    this.mockFetch.fetch({
      alerts: MOCK_RISK_ALERTS,
      limits: MOCK_RISK_LIMITS,
    }).then(data => {
      this.alerts.set(data.alerts);
      this.limits.set(data.limits);
      this.isLoading.set(false);
    }).catch((err: Error) => {
      this.hasError.set(true);
      this.errorMessage.set(err.message);
      this.isLoading.set(false);
    });
  }

  retry(): void {
    this.loadData();
  }

  readonly stressScenarios = MOCK_STRESS_SCENARIOS;
  readonly correlationData = MOCK_CORRELATION_DATA;
  readonly factorExposures = MOCK_FACTOR_EXPOSURES;
  readonly concentration = MOCK_CONCENTRATION;
  readonly liquidity = MOCK_LIQUIDITY;

  activeVarResult = computed(() => {
    const method = this.varMethod();
    const confidence = this.varConfidence();
    return MOCK_VAR_RESULTS.find(r => r.method === method && r.confidence === confidence)
      ?? MOCK_VAR_RESULTS[0];
  });

  var99Result = computed(() =>
    MOCK_VAR_RESULTS.find(r => r.method === this.varMethod() && r.confidence === 0.99)
      ?? MOCK_VAR_RESULTS[1],
  );

  varResults = computed(() =>
    MOCK_VAR_RESULTS.filter(r => r.method === this.varMethod()),
  );

  kpiVar95 = computed(() => this.fmt.formatCurrency(this.activeVarResult().varDollar));
  kpiVar99 = computed(() => this.fmt.formatCurrency(this.var99Result().varDollar));
  kpiCvar = computed(() => this.fmt.formatCurrency(this.activeVarResult().cvarDollar));
  kpiVol = computed(() => this.fmt.formatPercent(0.142));

  riskBudgetUsed = computed(() => {
    const current = this.activeVarResult().var;
    const limit = 0.025;
    return Math.round((Math.abs(current) / limit) * 100);
  });

  gaugeThresholds = [
    { value: 60, color: 'var(--color-gain)' },
    { value: 80, color: 'var(--color-chart-4)' },
    { value: 100, color: 'var(--color-loss)' },
  ];

  alertBadge = computed(() => {
    const unacked = this.alerts().filter(a => !a.acknowledged).length;
    const breached = this.limits().filter(l => l.status === 'breached' || l.status === 'warning').length;
    return unacked + breached;
  });

  tabs = computed<Tab[]>(() => {
    const badge = this.alertBadge();
    return [
      { id: 'var', label: 'VaR / CVaR' },
      { id: 'stress', label: 'Stress Testing' },
      { id: 'correlation', label: 'Correlation' },
      { id: 'factor', label: 'Factor Exposure' },
      { id: 'concentration', label: 'Concentration' },
      { id: 'liquidity', label: 'Liquidity' },
      { id: 'limits', label: 'Risk Limits', badge: badge > 0 ? badge : undefined },
    ];
  });

  openReportModal(): void {
    this.modalService.open({ component: ExportReportModalComponent, title: 'Export Report', size: 'lg' });
  }

  onAcknowledgeAlert(alertId: string) {
    this.alerts.update(list =>
      list.map(a => a.id === alertId ? { ...a, acknowledged: true } : a),
    );
  }

  onAddLimit(limit: RiskLimit) {
    this.limits.update(list => [...list, limit]);
  }

  onEditLimit(limit: RiskLimit) {
    this.limits.update(list =>
      list.map(l => l.id === limit.id ? limit : l),
    );
  }
}
