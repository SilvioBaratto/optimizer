import type { DrawdownPoint } from '../../shared/echarts-drawdown/echarts-drawdown';

export interface EquityPoint {
  date: string;
  value: number;
}

/**
 * Compute the running drawdown (value − rolling-peak) / rolling-peak
 * for a daily equity curve. Returns the same length as the input.
 *
 * Pure function; isolated from Angular DI so it can be unit-tested.
 */
export function computeDrawdownSeries(equity: EquityPoint[]): DrawdownPoint[] {
  if (equity.length === 0) return [];
  let peak = equity[0].value;
  return equity.map((p) => {
    if (p.value > peak) peak = p.value;
    const drawdown = peak > 0 ? (p.value - peak) / peak : 0;
    return { date: p.date, drawdown };
  });
}
