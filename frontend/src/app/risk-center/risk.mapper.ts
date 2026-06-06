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
  RiskLimitDto,
  StressScenario,
  StressScenarioApiResponse,
  VaRMethod,
  VaRResult,
  VarApiResponse,
} from './risk.model';

export function toVarResults(
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

export function toCorrelationData(res: CorrelationApiResponse | null): CorrelationData {
  return res ? { assets: res.assets, matrix: res.matrix } : { assets: [], matrix: [] };
}

export function toFactorExposures(res: FactorExposureApiResponse | null): FactorExposure[] {
  if (!res) return [];
  return Object.entries(res.exposures).map(([factor, exposure]) => ({
    factor,
    exposure,
    contribution: exposure,
    marginalContribution: exposure,
  }));
}

export function toConcentrationMetrics(
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

export function toLiquidityMetrics(res: LiquidityApiResponse | null): LiquidityMetric[] {
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

export function toStressScenarios(res: StressScenarioApiResponse | null): StressScenario[] {
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

export function toRiskLimitDisplay(dto: RiskLimitDto): RiskLimit {
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
