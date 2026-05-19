import { readCssVar } from './echarts-theme';

const PALETTE_SIZE = 8;

export function tickerColorMap(tickers: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  tickers.forEach((ticker, index) => {
    const slot = (index % PALETTE_SIZE) + 1;
    map[ticker] = readCssVar(`--color-chart-${slot}`);
  });
  return map;
}
