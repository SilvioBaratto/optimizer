import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

import { FormatService } from '../../services/format.service';
import { PortfolioApiService } from '../../services/portfolio-api.service';
import { RiskService } from '../../services/risk.service';
import { VarPanelComponent } from './var-panel';
import { StressPanelComponent } from './stress-panel';
import { CorrelationPanelComponent } from './correlation-panel';
import { FactorPanelComponent } from './factor-panel';
import { ConcentrationPanelComponent } from './concentration-panel';
import { LiquidityPanelComponent } from './liquidity-panel';
import { LimitsPanelComponent } from './limits-panel';

import type {
  ConcentrationApiResponse,
  ConcentrationMetric,
  CorrelationApiResponse,
  CorrelationData,
  FactorExposure,
  FactorExposureApiResponse,
  LiquidityApiResponse,
  LiquidityMetric,
  RiskLimit,
  RiskLimitCreatePayload,
  RiskLimitDto,
  RiskLimitListResponse,
  StressScenario,
  StressScenarioApiResponse,
  StressScenarioRequest,
  VaRMethod,
  VaRResult,
  VarApiResponse,
} from '../../models/risk.model';
import type { PortfolioDto } from '../../models/portfolio-api.model';

const DEFAULT_PORTFOLIO_VALUE = 10_000_000;

type ApiVarMethod = 'historical' | 'parametric';

function toVarResults(
  response: VarApiResponse | null,
  portfolioValue: number,
): VaRResult[] {
  if (!response) return [];
  const method = (response.method as VaRMethod) ?? 'historical';
  return Object.keys(response.var).map((conf) => {
    const varPct = response.var[conf] ?? 0;
    const cvarPct = response.cvar[conf] ?? 0;
    return {
      method,
      confidence: Number(conf) / 100,
      horizon: 1,
      var: varPct,
      cvar: cvarPct,
      portfolioValue,
      varDollar: varPct * portfolioValue,
      cvarDollar: cvarPct * portfolioValue,
    };
  });
}

function toCorrelationData(res: CorrelationApiResponse | null): CorrelationData {
  return res ? { assets: res.assets, matrix: res.matrix } : { assets: [], matrix: [] };
}

function toFactorExposures(res: FactorExposureApiResponse | null): FactorExposure[] {
  if (!res) return [];
  return Object.entries(res.exposures).map(([factor, exposure]) => ({
    factor,
    exposure,
    contribution: exposure,
    marginalContribution: exposure,
  }));
}

function toConcentrationMetrics(
  res: ConcentrationApiResponse | null,
): ConcentrationMetric[] {
  if (!res) return [];
  return res.assets.map((a) => ({
    ticker: a.ticker,
    name: a.name,
    weight: a.weight,
    riskContribution: a.weight,
    componentVar: 0,
  }));
}

function toLiquidityMetrics(res: LiquidityApiResponse | null): LiquidityMetric[] {
  if (!res) return [];
  return res.assets.map((a) => ({
    ticker: a.ticker,
    name: a.name,
    weight: a.weight,
    avgDailyVolume: a.avgDailyVolume ?? 0,
    daysToLiquidate: a.daysToLiquidate ?? 0,
    liquidityCost: a.liquidityCost ?? 0,
  }));
}

function toStressScenarios(res: StressScenarioApiResponse | null): StressScenario[] {
  if (!res) return [];
  return res.scenarios.map((s, i) => {
    const shocks = Object.entries(s.shocks);
    const [worstTicker, worstShock] = shocks.reduce(
      (acc, curr) => (curr[1] < acc[1] ? curr : acc),
      shocks[0] ?? ['', 0],
    );
    const meanShock = shocks.length > 0
      ? shocks.reduce((sum, [, v]) => sum + v, 0) / shocks.length
      : 0;
    return {
      id: `s-${i}`,
      name: s.name,
      description: s.description,
      portfolioImpact: meanShock,
      benchmarkImpact: 0,
      worstAsset: worstTicker,
      worstAssetImpact: worstShock,
    };
  });
}

function toRiskLimitDisplay(dto: RiskLimitDto): RiskLimit {
  const current = dto.currentValue ?? 0;
  const status: RiskLimit['status'] = dto.isBreached
    ? 'breached'
    : Math.abs(current) >= dto.threshold * 0.8
      ? 'warning'
      : 'ok';
  return {
    id: dto.id,
    name: dto.metric,
    metric: dto.metric,
    limit: dto.threshold,
    current,
    status,
    lastChecked: dto.lastCheckedAt ?? dto.updatedAt,
  };
}

@Component({
  selector: 'app-risk-center',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
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
  private readonly fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly risk = inject(RiskService);
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly destroyRef = inject(DestroyRef);

  // ── Loading state ────────────────────────────────────────────────────────
  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  // ── Portfolio selection ──────────────────────────────────────────────────
  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly selectedPortfolio = signal<string>('');
  readonly portfolioValue = signal<number>(DEFAULT_PORTFOLIO_VALUE);

  // ── Panel state ──────────────────────────────────────────────────────────
  readonly activeTab = signal('var');
  readonly varMethod = signal<ApiVarMethod>('historical');
  readonly varConfidence = signal(0.95);

  // ── Raw API responses ────────────────────────────────────────────────────
  readonly varResponse = signal<VarApiResponse | null>(null);
  readonly correlationResponse = signal<CorrelationApiResponse | null>(null);
  readonly factorExposureResponse = signal<FactorExposureApiResponse | null>(null);
  readonly concentrationResponse = signal<ConcentrationApiResponse | null>(null);
  readonly liquidityResponse = signal<LiquidityApiResponse | null>(null);
  readonly stressResponse = signal<StressScenarioApiResponse | null>(null);
  readonly limitsResponse = signal<RiskLimitListResponse>({ items: [], breachCount: 0 });

  // ── Loading / error per panel ────────────────────────────────────────────
  readonly panelErrors = signal<Record<string, string | null>>({});
  readonly stressLoading = signal<boolean>(false);

  // ── Transformed views (plug into existing panel inputs) ──────────────────
  readonly varResults = computed<VaRResult[]>(() =>
    toVarResults(this.varResponse(), this.portfolioValue()),
  );
  readonly activeVarResult = computed<VaRResult | undefined>(() => {
    const conf = this.varConfidence();
    return this.varResults().find((r) => r.confidence === conf) ?? this.varResults()[0];
  });
  readonly var99Result = computed<VaRResult | undefined>(() =>
    this.varResults().find((r) => r.confidence === 0.99),
  );

  readonly correlationData = computed<CorrelationData>(() =>
    toCorrelationData(this.correlationResponse()),
  );
  readonly factorExposures = computed<FactorExposure[]>(() =>
    toFactorExposures(this.factorExposureResponse()),
  );
  readonly concentration = computed<ConcentrationMetric[]>(() =>
    toConcentrationMetrics(this.concentrationResponse()),
  );
  readonly liquidity = computed<LiquidityMetric[]>(() =>
    toLiquidityMetrics(this.liquidityResponse()),
  );
  readonly stressScenarios = computed<StressScenario[]>(() =>
    toStressScenarios(this.stressResponse()),
  );

  readonly limits = computed<RiskLimit[]>(() =>
    this.limitsResponse().items.map(toRiskLimitDisplay),
  );
  readonly breachCount = computed(() => this.limitsResponse().breachCount);

  // ── KPI cards (header stats) ─────────────────────────────────────────────
  readonly kpiVar95 = computed(() =>
    this.activeVarResult()
      ? this.fmt.formatCurrencyCompact(this.activeVarResult()!.varDollar)
      : '—',
  );
  readonly kpiVar99 = computed(() =>
    this.var99Result() ? this.fmt.formatCurrencyCompact(this.var99Result()!.varDollar) : '—',
  );
  readonly kpiCvar = computed(() =>
    this.activeVarResult()
      ? this.fmt.formatCurrencyCompact(this.activeVarResult()!.cvarDollar)
      : '—',
  );
  readonly kpiVol = computed(() => this.fmt.formatPercent(0));

  readonly tabs = computed<Tab[]>(() => {
    const badge = this.breachCount();
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

  readonly alerts = signal([]);

  constructor() {
    this.loadPortfolios();
    effect(() => this.onPortfolioChange());
    effect(() => this.onVarMethodChange());
  }

  retry(): void {
    this.hasError.set(false);
    this.loadPortfolios();
  }

  openReportModal(): void {
    this.modalService.open({
      component: ExportReportModalComponent,
      title: 'Export Report',
      size: 'lg',
    });
  }

  onMethodChange(method: VaRMethod): void {
    if (method === 'monte_carlo') return; // Not supported by backend
    this.varMethod.set(method as ApiVarMethod);
  }

  onConfidenceChange(confidence: number): void {
    this.varConfidence.set(confidence);
  }

  onGenerateStress(payload: { macroContext: string; nScenarios: number }): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.stressLoading.set(true);
    const weights = this.concentration().reduce<Record<string, number>>(
      (acc, a) => ({ ...acc, [a.ticker]: a.weight }),
      {},
    );
    const body: StressScenarioRequest = {
      current_portfolio: Object.keys(weights).length > 0 ? weights : { AAPL: 1 },
      macro_context: payload.macroContext,
      n_scenarios: payload.nScenarios,
    };
    this.risk
      .generateStressScenarios(body)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.stressResponse.set(res);
          this.stressLoading.set(false);
        },
        error: (err: Error) => {
          this.setPanelError('stress', err.message ?? 'Stress failed');
          this.stressLoading.set(false);
        },
      });
  }

  onCreateLimit(body: RiskLimitCreatePayload): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.risk
      .createLimit(name, body)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.reloadLimits(),
        error: (err: Error) => this.setPanelError('limits', err.message ?? 'Create failed'),
      });
  }

  onUpdateLimit(update: { id: string; current: number; breached: boolean }): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.risk
      .updateLimit(name, update.id, {
        current_value: update.current,
        is_breached: update.breached,
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.reloadLimits(),
        error: (err: Error) => this.setPanelError('limits', err.message ?? 'Update failed'),
      });
  }

  onDeleteLimit(id: string): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.risk
      .deleteLimit(name, id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.reloadLimits(),
        error: (err: Error) => this.setPanelError('limits', err.message ?? 'Delete failed'),
      });
  }

  private loadPortfolios(): void {
    this.isLoading.set(true);
    this.portfolioApi
      .list()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => this.applyPortfolios(list.items),
        error: (err: Error) => {
          this.errorMessage.set(err.message ?? 'Failed to load portfolios');
          this.hasError.set(true);
          this.isLoading.set(false);
        },
      });
  }

  private applyPortfolios(items: PortfolioDto[]): void {
    this.portfolios.set(items);
    if (items.length > 0 && !this.selectedPortfolio()) {
      this.selectedPortfolio.set(items[0].name);
    }
    this.isLoading.set(false);
  }

  private onPortfolioChange(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.fetchAnalytics(name);
    this.reloadLimits();
  }

  private onVarMethodChange(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.fetchVar(name);
  }

  private fetchAnalytics(name: string): void {
    this.fetchVar(name);
    this.fetchFor('correlation', () => this.risk.getCorrelation(name), this.correlationResponse);
    this.fetchFor('factor', () => this.risk.getFactorExposure(name), this.factorExposureResponse);
    this.fetchFor('concentration', () => this.risk.getConcentration(name), this.concentrationResponse);
    this.fetchFor('liquidity', () => this.risk.getLiquidity(name), this.liquidityResponse);
  }

  private fetchVar(name: string): void {
    this.fetchFor(
      'var',
      () => this.risk.getVar(name, { method: this.varMethod() }),
      this.varResponse,
    );
  }

  private reloadLimits(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.risk
      .listLimits(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.limitsResponse.set(res),
        error: (err: Error) => this.setPanelError('limits', err.message ?? 'Load failed'),
      });
  }

  private fetchFor<T>(
    key: string,
    supplier: () => import('rxjs').Observable<T>,
    target: { set: (v: T | null) => void },
  ): void {
    this.clearPanelError(key);
    supplier()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => target.set(res),
        error: (err: Error) => {
          target.set(null);
          this.setPanelError(key, err.message ?? `Failed to load ${key}`);
        },
      });
  }

  private setPanelError(key: string, message: string): void {
    this.panelErrors.update((prev) => ({ ...prev, [key]: message }));
  }

  private clearPanelError(key: string): void {
    this.panelErrors.update((prev) => ({ ...prev, [key]: null }));
  }
}
