import { ScatterPoint } from '../shared/echarts-scatter/echarts-scatter';
import { UniverseStats, Exchange, Instrument } from '../models/universe.model';

// ── Universe ──

export const MOCK_STATS: UniverseStats = {
  total_exchanges: 12,
  total_instruments: 2_402,
  last_updated: '2026-02-19T08:00:00Z',
};

export const MOCK_EXCHANGES: Exchange[] = [
  { id: '1', name: 'New York Stock Exchange', mic: 'XNYS', country: 'US', instrument_count: 820 },
  { id: '2', name: 'NASDAQ', mic: 'XNAS', country: 'US', instrument_count: 645 },
  { id: '3', name: 'London Stock Exchange', mic: 'XLON', country: 'GB', instrument_count: 310 },
  { id: '4', name: 'Tokyo Stock Exchange', mic: 'XJPX', country: 'JP', instrument_count: 180 },
  { id: '5', name: 'Euronext Paris', mic: 'XPAR', country: 'FR', instrument_count: 125 },
  { id: '6', name: 'Deutsche Boerse', mic: 'XETR', country: 'DE', instrument_count: 98 },
  { id: '7', name: 'Hong Kong Exchange', mic: 'XHKG', country: 'HK', instrument_count: 72 },
  { id: '8', name: 'Toronto Stock Exchange', mic: 'XTSE', country: 'CA', instrument_count: 58 },
  { id: '9', name: 'SIX Swiss Exchange', mic: 'XSWX', country: 'CH', instrument_count: 34 },
  { id: '10', name: 'Australian Securities Exchange', mic: 'XASX', country: 'AU', instrument_count: 28 },
  { id: '11', name: 'Borsa Italiana', mic: 'XMIL', country: 'IT', instrument_count: 18 },
  { id: '12', name: 'Bolsa de Madrid', mic: 'XMAD', country: 'ES', instrument_count: 14 },
];

export const MOCK_INSTRUMENTS: Instrument[] = [
  { id: '1', ticker: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US0378331005' },
  { id: '2', ticker: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US5949181045' },
  { id: '3', ticker: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US02079K3059' },
  { id: '4', ticker: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US0231351067' },
  { id: '5', ticker: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US67066G1040' },
  { id: '6', ticker: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US30303M1027' },
  { id: '7', ticker: 'JPM', name: 'JPMorgan Chase & Co.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US46625H1005' },
  { id: '8', ticker: 'V', name: 'Visa Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US92826C8394' },
  { id: '9', ticker: 'JNJ', name: 'Johnson & Johnson', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US4781601046' },
  { id: '10', ticker: 'WMT', name: 'Walmart Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US9311421039' },
  { id: '11', ticker: 'PG', name: 'Procter & Gamble Co.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US7427181091' },
  { id: '12', ticker: 'UNH', name: 'UnitedHealth Group Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US91324P1021' },
  { id: '13', ticker: 'HD', name: 'Home Depot Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US4370761029' },
  { id: '14', ticker: 'BAC', name: 'Bank of America Corp.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US0605051046' },
  { id: '15', ticker: 'XOM', name: 'Exxon Mobil Corporation', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US30231G1022' },
  { id: '16', ticker: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US88160R1014' },
  { id: '17', ticker: 'AVGO', name: 'Broadcom Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US11135F1012' },
  { id: '18', ticker: 'KO', name: 'Coca-Cola Company', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US1912161007' },
  { id: '19', ticker: 'PEP', name: 'PepsiCo Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US7134481081' },
  { id: '20', ticker: 'COST', name: 'Costco Wholesale Corp.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US22160K1051' },
  { id: '21', ticker: 'LLY', name: 'Eli Lilly and Company', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US5324571083' },
  { id: '22', ticker: 'MRK', name: 'Merck & Co. Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US58933Y1055' },
  { id: '23', ticker: 'ABBV', name: 'AbbVie Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US00287Y1091' },
  { id: '24', ticker: 'CRM', name: 'Salesforce Inc.', exchange: 'NYSE', type: 'EQUITY', currency: 'USD', isin: 'US79466L3024' },
  { id: '25', ticker: 'ADBE', name: 'Adobe Inc.', exchange: 'NASDAQ', type: 'EQUITY', currency: 'USD', isin: 'US00724F1012' },
  { id: '26', ticker: 'SHEL', name: 'Shell plc', exchange: 'LSE', type: 'EQUITY', currency: 'GBP', isin: 'GB00BP6MXD84' },
  { id: '27', ticker: '7203.T', name: 'Toyota Motor Corp.', exchange: 'TSE', type: 'EQUITY', currency: 'JPY', isin: 'JP3633400001' },
  { id: '28', ticker: 'NESN.SW', name: 'Nestle S.A.', exchange: 'SIX', type: 'EQUITY', currency: 'CHF', isin: 'CH0038863350' },
  { id: '29', ticker: 'MC.PA', name: 'LVMH Moet Hennessy', exchange: 'Euronext', type: 'EQUITY', currency: 'EUR', isin: 'FR0000121014' },
  { id: '30', ticker: 'SAP.DE', name: 'SAP SE', exchange: 'XETRA', type: 'EQUITY', currency: 'EUR', isin: 'DE0007164600' },
];

// ── Portfolio Weights (used by rebalancing/risk/attribution/portfolio-builder mocks) ──

export const WEIGHT_DATA: Array<{ ticker: string; name: string; sector: string; weight: number }> = [
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Technology', weight: 0.042 },
  { ticker: 'MSFT', name: 'Microsoft Corp.', sector: 'Technology', weight: 0.038 },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology', weight: 0.031 },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', sector: 'Consumer Cyclical', weight: 0.028 },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', sector: 'Technology', weight: 0.025 },
  { ticker: 'META', name: 'Meta Platforms', sector: 'Technology', weight: 0.022 },
  { ticker: 'JPM', name: 'JPMorgan Chase', sector: 'Financial Services', weight: 0.020 },
  { ticker: 'V', name: 'Visa Inc.', sector: 'Financial Services', weight: 0.019 },
  { ticker: 'JNJ', name: 'Johnson & Johnson', sector: 'Healthcare', weight: 0.018 },
  { ticker: 'WMT', name: 'Walmart Inc.', sector: 'Consumer Defensive', weight: 0.017 },
  { ticker: 'PG', name: 'Procter & Gamble', sector: 'Consumer Defensive', weight: 0.016 },
  { ticker: 'UNH', name: 'UnitedHealth Group', sector: 'Healthcare', weight: 0.015 },
  { ticker: 'HD', name: 'Home Depot', sector: 'Consumer Cyclical', weight: 0.014 },
  { ticker: 'BAC', name: 'Bank of America', sector: 'Financial Services', weight: 0.014 },
  { ticker: 'XOM', name: 'Exxon Mobil', sector: 'Energy', weight: 0.013 },
  { ticker: 'TSLA', name: 'Tesla Inc.', sector: 'Consumer Cyclical', weight: 0.013 },
  { ticker: 'AVGO', name: 'Broadcom Inc.', sector: 'Technology', weight: 0.012 },
  { ticker: 'KO', name: 'Coca-Cola Co.', sector: 'Consumer Defensive', weight: 0.012 },
  { ticker: 'PEP', name: 'PepsiCo Inc.', sector: 'Consumer Defensive', weight: 0.011 },
  { ticker: 'COST', name: 'Costco Wholesale', sector: 'Consumer Defensive', weight: 0.011 },
  { ticker: 'LLY', name: 'Eli Lilly', sector: 'Healthcare', weight: 0.011 },
  { ticker: 'MRK', name: 'Merck & Co.', sector: 'Healthcare', weight: 0.010 },
  { ticker: 'ABBV', name: 'AbbVie Inc.', sector: 'Healthcare', weight: 0.010 },
  { ticker: 'CRM', name: 'Salesforce Inc.', sector: 'Technology', weight: 0.010 },
  { ticker: 'ADBE', name: 'Adobe Inc.', sector: 'Technology', weight: 0.009 },
  { ticker: 'TMO', name: 'Thermo Fisher', sector: 'Healthcare', weight: 0.009 },
  { ticker: 'ACN', name: 'Accenture plc', sector: 'Technology', weight: 0.009 },
  { ticker: 'DHR', name: 'Danaher Corp.', sector: 'Healthcare', weight: 0.009 },
  { ticker: 'ABT', name: 'Abbott Labs', sector: 'Healthcare', weight: 0.008 },
  { ticker: 'CMCSA', name: 'Comcast Corp.', sector: 'Communication', weight: 0.008 },
  { ticker: 'NFLX', name: 'Netflix Inc.', sector: 'Communication', weight: 0.008 },
  { ticker: 'DIS', name: 'Walt Disney Co.', sector: 'Communication', weight: 0.008 },
  { ticker: 'ORCL', name: 'Oracle Corp.', sector: 'Technology', weight: 0.008 },
  { ticker: 'CSCO', name: 'Cisco Systems', sector: 'Technology', weight: 0.007 },
  { ticker: 'INTC', name: 'Intel Corp.', sector: 'Technology', weight: 0.007 },
  { ticker: 'IBM', name: 'IBM Corp.', sector: 'Technology', weight: 0.007 },
  { ticker: 'QCOM', name: 'Qualcomm Inc.', sector: 'Technology', weight: 0.007 },
  { ticker: 'INTU', name: 'Intuit Inc.', sector: 'Technology', weight: 0.006 },
  { ticker: 'TXN', name: 'Texas Instruments', sector: 'Technology', weight: 0.006 },
  { ticker: 'AMD', name: 'AMD Inc.', sector: 'Technology', weight: 0.006 },
  { ticker: 'NOW', name: 'ServiceNow', sector: 'Technology', weight: 0.006 },
  { ticker: 'NEE', name: 'NextEra Energy', sector: 'Utilities', weight: 0.006 },
  { ticker: 'LOW', name: 'Lowe\'s Companies', sector: 'Consumer Cyclical', weight: 0.006 },
  { ticker: 'SPGI', name: 'S&P Global', sector: 'Financial Services', weight: 0.006 },
  { ticker: 'GS', name: 'Goldman Sachs', sector: 'Financial Services', weight: 0.005 },
  { ticker: 'BLK', name: 'BlackRock Inc.', sector: 'Financial Services', weight: 0.005 },
  { ticker: 'MDT', name: 'Medtronic plc', sector: 'Healthcare', weight: 0.005 },
  { ticker: 'ADP', name: 'ADP Inc.', sector: 'Industrials', weight: 0.005 },
  { ticker: 'ISRG', name: 'Intuitive Surgical', sector: 'Healthcare', weight: 0.005 },
  { ticker: 'DE', name: 'Deere & Company', sector: 'Industrials', weight: 0.005 },
  { ticker: 'VRTX', name: 'Vertex Pharma', sector: 'Healthcare', weight: 0.005 },
  { ticker: 'BKNG', name: 'Booking Holdings', sector: 'Consumer Cyclical', weight: 0.005 },
  { ticker: 'MMC', name: 'Marsh McLennan', sector: 'Financial Services', weight: 0.005 },
  { ticker: 'CB', name: 'Chubb Limited', sector: 'Financial Services', weight: 0.004 },
  { ticker: 'SYK', name: 'Stryker Corp.', sector: 'Healthcare', weight: 0.004 },
  { ticker: 'SCHW', name: 'Charles Schwab', sector: 'Financial Services', weight: 0.004 },
  { ticker: 'AMT', name: 'American Tower', sector: 'Real Estate', weight: 0.004 },
  { ticker: 'PLD', name: 'Prologis Inc.', sector: 'Real Estate', weight: 0.004 },
  { ticker: 'MO', name: 'Altria Group', sector: 'Consumer Defensive', weight: 0.004 },
  { ticker: 'SO', name: 'Southern Company', sector: 'Utilities', weight: 0.004 },
  { ticker: 'DUK', name: 'Duke Energy', sector: 'Utilities', weight: 0.004 },
  { ticker: 'CL', name: 'Colgate-Palmolive', sector: 'Consumer Defensive', weight: 0.004 },
  { ticker: 'ITW', name: 'Illinois Tool Works', sector: 'Industrials', weight: 0.004 },
  { ticker: 'ETN', name: 'Eaton Corp.', sector: 'Industrials', weight: 0.004 },
  { ticker: 'CVX', name: 'Chevron Corp.', sector: 'Energy', weight: 0.004 },
  { ticker: 'SLB', name: 'Schlumberger Ltd.', sector: 'Energy', weight: 0.003 },
  { ticker: 'EOG', name: 'EOG Resources', sector: 'Energy', weight: 0.003 },
  { ticker: 'FDX', name: 'FedEx Corp.', sector: 'Industrials', weight: 0.003 },
  { ticker: 'GE', name: 'GE Aerospace', sector: 'Industrials', weight: 0.003 },
  { ticker: 'CAT', name: 'Caterpillar Inc.', sector: 'Industrials', weight: 0.003 },
  { ticker: 'HON', name: 'Honeywell Intl', sector: 'Industrials', weight: 0.003 },
  { ticker: 'UPS', name: 'United Parcel Service', sector: 'Industrials', weight: 0.003 },
  { ticker: 'RTX', name: 'RTX Corporation', sector: 'Industrials', weight: 0.003 },
  { ticker: 'MMM', name: '3M Company', sector: 'Industrials', weight: 0.003 },
  { ticker: 'BA', name: 'Boeing Company', sector: 'Industrials', weight: 0.003 },
  { ticker: 'COP', name: 'ConocoPhillips', sector: 'Energy', weight: 0.003 },
  { ticker: 'PSX', name: 'Phillips 66', sector: 'Energy', weight: 0.002 },
  { ticker: 'VLO', name: 'Valero Energy', sector: 'Energy', weight: 0.002 },
  { ticker: 'WFC', name: 'Wells Fargo', sector: 'Financial Services', weight: 0.002 },
  { ticker: 'USB', name: 'U.S. Bancorp', sector: 'Financial Services', weight: 0.002 },
  { ticker: 'PNC', name: 'PNC Financial', sector: 'Financial Services', weight: 0.002 },
  { ticker: 'TFC', name: 'Truist Financial', sector: 'Financial Services', weight: 0.002 },
  { ticker: 'SPG', name: 'Simon Property', sector: 'Real Estate', weight: 0.002 },
  { ticker: 'O', name: 'Realty Income', sector: 'Real Estate', weight: 0.002 },
  { ticker: 'WEC', name: 'WEC Energy', sector: 'Utilities', weight: 0.002 },
  { ticker: 'AEP', name: 'American Electric', sector: 'Utilities', weight: 0.002 },
  { ticker: 'D', name: 'Dominion Energy', sector: 'Utilities', weight: 0.002 },
  { ticker: 'EXC', name: 'Exelon Corp.', sector: 'Utilities', weight: 0.002 },
  { ticker: 'GILD', name: 'Gilead Sciences', sector: 'Healthcare', weight: 0.002 },
  { ticker: 'BMY', name: 'Bristol-Myers Squibb', sector: 'Healthcare', weight: 0.002 },
  { ticker: 'AMGN', name: 'Amgen Inc.', sector: 'Healthcare', weight: 0.002 },
  { ticker: 'REGN', name: 'Regeneron Pharma', sector: 'Healthcare', weight: 0.002 },
  { ticker: 'ZTS', name: 'Zoetis Inc.', sector: 'Healthcare', weight: 0.002 },
  { ticker: 'TMUS', name: 'T-Mobile US', sector: 'Communication', weight: 0.002 },
  { ticker: 'VZ', name: 'Verizon Comms', sector: 'Communication', weight: 0.001 },
  { ticker: 'T', name: 'AT&T Inc.', sector: 'Communication', weight: 0.001 },
];

// ── Efficient Frontier ──

function generateFrontierPoints(): ScatterPoint[] {
  const pts: ScatterPoint[] = [];
  // Parametric frontier: return = a*risk^0.5 + b, with some noise
  for (let i = 0; i < 30; i++) {
    const risk = 0.06 + (i / 29) * 0.18; // 6% to 24% annualised vol
    const ret = 0.04 + 0.55 * Math.sqrt(risk) + (Math.random() - 0.5) * 0.004;
    pts.push({ x: +risk.toFixed(4), y: +ret.toFixed(4) });
  }
  // Sort by risk
  pts.sort((a, b) => a.x - b.x);
  return pts;
}

const _frontierPts = generateFrontierPoints();
export const MOCK_EFFICIENT_FRONTIER: ScatterPoint[] = _frontierPts;
export const MOCK_OPTIMAL_POINT: ScatterPoint = {
  // Pick the point with highest Sharpe (return/risk) from the frontier
  ..._frontierPts.reduce((best, p) => (p.y / p.x > best.y / best.x ? p : best), _frontierPts[0]),
  label: 'Max Sharpe',
};

// ── Correlation Matrix ──

export const MOCK_CORRELATION_MATRIX: { assets: string[]; matrix: number[][] } = (() => {
  const assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'JNJ', 'XOM', 'WMT', 'NEE'];
  const n = assets.length;

  // Base correlations by sector proximity
  const sectorGroup = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]; // same index = same sector
  const matrix: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      const base = sectorGroup[i] === sectorGroup[j] ? 0.72 : 0.28;
      const noise = (Math.random() - 0.5) * 0.12;
      return Math.min(0.98, Math.max(-0.3, +(base + noise).toFixed(2)));
    }),
  );
  // Enforce symmetry
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      matrix[j][i] = matrix[i][j];
    }
  }
  return { assets, matrix };
})();

