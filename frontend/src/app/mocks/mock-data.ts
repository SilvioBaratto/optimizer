import { ScatterPoint } from '../shared/echarts-scatter/echarts-scatter';
import { HealthCheck, TableInfo } from '../models/database.model';
import { UniverseStats, Exchange, Instrument } from '../models/universe.model';
import { TickerProfile, PriceHistory, AnalystRecommendation, InsiderTransaction, TickerNews } from '../models/yfinance.model';
import { EconomicIndicator, BondYield, CountryMacroSummary } from '../models/macro.model';
import { StrategyInfo, PortfolioResult, BacktestPoint } from '../models/portfolio.model';

// ── Database ──

export const MOCK_HEALTH: HealthCheck = {
  status: 'healthy',
  latency_ms: 12,
  database: 'optimizer_db',
  version: 'PostgreSQL 16.2',
};

export const MOCK_TABLES: TableInfo[] = [
  { name: 'price_history', schema: 'public', row_count: 1_847_293, size_bytes: 524_288_000, size_pretty: '500 MB' },
  { name: 'financial_statements', schema: 'public', row_count: 2_614_780, size_bytes: 314_572_800, size_pretty: '300 MB' },
  { name: 'instruments', schema: 'public', row_count: 2_402, size_bytes: 1_048_576, size_pretty: '1 MB' },
  { name: 'ticker_profiles', schema: 'public', row_count: 2_402, size_bytes: 5_242_880, size_pretty: '5 MB' },
  { name: 'analyst_recommendations', schema: 'public', row_count: 48_040, size_bytes: 10_485_760, size_pretty: '10 MB' },
  { name: 'insider_transactions', schema: 'public', row_count: 124_100, size_bytes: 20_971_520, size_pretty: '20 MB' },
  { name: 'exchanges', schema: 'public', row_count: 12, size_bytes: 32_768, size_pretty: '32 KB' },
  { name: 'economic_indicators', schema: 'public', row_count: 3_240, size_bytes: 2_097_152, size_pretty: '2 MB' },
  { name: 'bond_yields', schema: 'public', row_count: 8_640, size_bytes: 1_048_576, size_pretty: '1 MB' },
  { name: 'trading_economics', schema: 'public', row_count: 15_600, size_bytes: 4_194_304, size_pretty: '4 MB' },
];

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

// ── Ticker Profiles ──

export const MOCK_PROFILES: Record<string, TickerProfile> = {
  '1': {
    ticker: 'AAPL', name: 'Apple Inc.', sector: 'Technology', industry: 'Consumer Electronics',
    market_cap: 3_420_000_000_000, pe_ratio: 33.2, dividend_yield: 0.0044, beta: 1.24,
    fifty_two_week_high: 248.25, fifty_two_week_low: 164.08, avg_volume: 54_200_000,
    description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
  },
  '2': {
    ticker: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology', industry: 'Software—Infrastructure',
    market_cap: 3_180_000_000_000, pe_ratio: 37.8, dividend_yield: 0.0072, beta: 0.89,
    fifty_two_week_high: 468.35, fifty_two_week_low: 362.90, avg_volume: 22_100_000,
    description: 'Microsoft Corporation develops and supports software, services, devices, and solutions worldwide.',
  },
  '7': {
    ticker: 'JPM', name: 'JPMorgan Chase & Co.', sector: 'Financial Services', industry: 'Banks—Diversified',
    market_cap: 680_000_000_000, pe_ratio: 12.4, dividend_yield: 0.0208, beta: 1.12,
    fifty_two_week_high: 242.80, fifty_two_week_low: 170.25, avg_volume: 9_800_000,
    description: 'JPMorgan Chase & Co. operates as a financial services company worldwide.',
  },
};

// ── Price History (AAPL, 252 trading days) ──

function generatePrices(): PriceHistory[] {
  const prices: PriceHistory[] = [];
  let close = 178.50;
  const startDate = new Date('2025-02-19');

  for (let i = 0; i < 252; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + Math.floor(i * 365 / 252));
    if (date.getDay() === 0) date.setDate(date.getDate() + 1);
    if (date.getDay() === 6) date.setDate(date.getDate() + 2);

    const dailyReturn = (Math.random() - 0.48) * 0.03;
    close = close * (1 + dailyReturn);
    const high = close * (1 + Math.random() * 0.015);
    const low = close * (1 - Math.random() * 0.015);
    const open = low + Math.random() * (high - low);
    const volume = Math.floor(40_000_000 + Math.random() * 30_000_000);

    prices.push({
      date: date.toISOString().split('T')[0],
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume,
    });
  }
  return prices;
}

export const MOCK_PRICES: PriceHistory[] = generatePrices();

// ── Analyst Recommendations ──

export const MOCK_RECOMMENDATIONS: AnalystRecommendation[] = [
  { period: '2026-02', strong_buy: 18, buy: 12, hold: 6, sell: 1, strong_sell: 0 },
  { period: '2026-01', strong_buy: 16, buy: 14, hold: 5, sell: 2, strong_sell: 0 },
  { period: '2025-12', strong_buy: 15, buy: 13, hold: 7, sell: 2, strong_sell: 1 },
  { period: '2025-11', strong_buy: 17, buy: 11, hold: 6, sell: 3, strong_sell: 0 },
];

// ── Insider Transactions ──

export const MOCK_INSIDER_TRANSACTIONS: InsiderTransaction[] = [
  { date: '2026-02-10', insider: 'Tim Cook', position: 'CEO', transaction_type: 'Sale', shares: 50_000, value: 12_125_000 },
  { date: '2026-01-28', insider: 'Luca Maestri', position: 'CFO', transaction_type: 'Sale', shares: 25_000, value: 5_987_500 },
  { date: '2026-01-15', insider: 'Jeff Williams', position: 'COO', transaction_type: 'Sale', shares: 15_000, value: 3_577_500 },
  { date: '2025-12-20', insider: 'Deirdre O\'Brien', position: 'SVP', transaction_type: 'Exercise', shares: 100_000, value: 23_450_000 },
];

// ── News ──

export const MOCK_NEWS: TickerNews[] = [
  { title: 'Apple Reports Record Q1 2026 Revenue Driven by AI Features', publisher: 'Reuters', link: '#', published: '2026-02-18T14:30:00Z', summary: 'Apple posted quarterly revenue of $124.3B, exceeding analyst expectations by 4%.' },
  { title: 'Apple Vision Pro 2 Expected to Launch in Summer 2026', publisher: 'Bloomberg', link: '#', published: '2026-02-15T09:00:00Z', summary: 'Sources indicate Apple is preparing a more affordable Vision Pro model.' },
  { title: 'iPhone 17 Supply Chain Ramp-Up Begins', publisher: 'Nikkei Asia', link: '#', published: '2026-02-12T06:00:00Z', summary: 'Component orders suggest stronger initial production volumes than iPhone 16.' },
];

// ── Macro ──

export const MOCK_INDICATORS: EconomicIndicator[] = [
  { id: '1', name: 'GDP Growth Rate', country: 'US', category: 'Growth', value: 2.8, previous: 3.1, unit: '%', frequency: 'Quarterly', last_updated: '2026-01-30' },
  { id: '2', name: 'Unemployment Rate', country: 'US', category: 'Labor', value: 3.9, previous: 4.0, unit: '%', frequency: 'Monthly', last_updated: '2026-02-07' },
  { id: '3', name: 'CPI YoY', country: 'US', category: 'Inflation', value: 2.6, previous: 2.9, unit: '%', frequency: 'Monthly', last_updated: '2026-02-12' },
  { id: '4', name: 'Federal Funds Rate', country: 'US', category: 'Monetary', value: 4.50, previous: 4.75, unit: '%', frequency: 'Meeting', last_updated: '2026-01-29' },
  { id: '5', name: 'Industrial Production', country: 'US', category: 'Growth', value: 0.3, previous: -0.1, unit: '% MoM', frequency: 'Monthly', last_updated: '2026-02-14' },
  { id: '6', name: 'Retail Sales MoM', country: 'US', category: 'Consumption', value: 0.6, previous: 0.4, unit: '%', frequency: 'Monthly', last_updated: '2026-02-14' },
  { id: '7', name: 'Consumer Confidence', country: 'US', category: 'Sentiment', value: 104.2, previous: 101.8, unit: 'Index', frequency: 'Monthly', last_updated: '2026-02-10' },
  { id: '8', name: 'PMI Manufacturing', country: 'US', category: 'Growth', value: 51.2, previous: 49.8, unit: 'Index', frequency: 'Monthly', last_updated: '2026-02-03' },
  { id: '9', name: 'GDP Growth Rate', country: 'GB', category: 'Growth', value: 1.2, previous: 0.9, unit: '%', frequency: 'Quarterly', last_updated: '2026-01-28' },
  { id: '10', name: 'CPI YoY', country: 'GB', category: 'Inflation', value: 3.1, previous: 3.4, unit: '%', frequency: 'Monthly', last_updated: '2026-02-10' },
  { id: '11', name: 'Bank Rate', country: 'GB', category: 'Monetary', value: 4.25, previous: 4.50, unit: '%', frequency: 'Meeting', last_updated: '2026-02-06' },
  { id: '12', name: 'GDP Growth Rate', country: 'EU', category: 'Growth', value: 0.8, previous: 0.5, unit: '%', frequency: 'Quarterly', last_updated: '2026-01-30' },
  { id: '13', name: 'HICP YoY', country: 'EU', category: 'Inflation', value: 2.2, previous: 2.4, unit: '%', frequency: 'Monthly', last_updated: '2026-02-10' },
  { id: '14', name: 'ECB Main Rate', country: 'EU', category: 'Monetary', value: 3.15, previous: 3.40, unit: '%', frequency: 'Meeting', last_updated: '2026-01-25' },
  { id: '15', name: 'GDP Growth Rate', country: 'JP', category: 'Growth', value: 1.0, previous: 0.7, unit: '%', frequency: 'Quarterly', last_updated: '2026-01-28' },
];

export const MOCK_BOND_YIELDS: BondYield[] = [
  { maturity: '1M', yield_pct: 4.32, change_bps: -2, country: 'US', date: '2026-02-19' },
  { maturity: '3M', yield_pct: 4.28, change_bps: -3, country: 'US', date: '2026-02-19' },
  { maturity: '6M', yield_pct: 4.18, change_bps: -5, country: 'US', date: '2026-02-19' },
  { maturity: '1Y', yield_pct: 4.05, change_bps: -4, country: 'US', date: '2026-02-19' },
  { maturity: '2Y', yield_pct: 3.92, change_bps: -6, country: 'US', date: '2026-02-19' },
  { maturity: '3Y', yield_pct: 3.88, change_bps: -5, country: 'US', date: '2026-02-19' },
  { maturity: '5Y', yield_pct: 3.95, change_bps: -3, country: 'US', date: '2026-02-19' },
  { maturity: '7Y', yield_pct: 4.08, change_bps: -2, country: 'US', date: '2026-02-19' },
  { maturity: '10Y', yield_pct: 4.22, change_bps: -1, country: 'US', date: '2026-02-19' },
  { maturity: '20Y', yield_pct: 4.52, change_bps: 0, country: 'US', date: '2026-02-19' },
  { maturity: '30Y', yield_pct: 4.45, change_bps: 1, country: 'US', date: '2026-02-19' },
];

export const MOCK_COUNTRY_SUMMARIES: CountryMacroSummary[] = [
  { country: 'US', gdp_growth: 2.8, inflation: 2.6, unemployment: 3.9, interest_rate: 4.50, debt_to_gdp: 123.4 },
  { country: 'GB', gdp_growth: 1.2, inflation: 3.1, unemployment: 4.2, interest_rate: 4.25, debt_to_gdp: 98.6 },
  { country: 'EU', gdp_growth: 0.8, inflation: 2.2, unemployment: 6.4, interest_rate: 3.15, debt_to_gdp: 88.2 },
  { country: 'JP', gdp_growth: 1.0, inflation: 2.8, unemployment: 2.5, interest_rate: 0.25, debt_to_gdp: 255.2 },
];

// ── Portfolio Strategies ──

export const MOCK_STRATEGIES: StrategyInfo[] = [
  { type: 'min_variance', name: 'Minimum Variance', description: 'Minimizes portfolio volatility', category: 'convex' },
  { type: 'max_sharpe', name: 'Maximum Sharpe', description: 'Maximizes risk-adjusted return', category: 'convex' },
  { type: 'max_utility', name: 'Maximum Utility', description: 'Maximizes mean-variance utility', category: 'convex' },
  { type: 'min_cvar', name: 'Minimum CVaR', description: 'Minimizes conditional value at risk', category: 'convex' },
  { type: 'efficient_frontier', name: 'Efficient Frontier', description: 'Target return on the efficient frontier', category: 'convex' },
  { type: 'risk_parity', name: 'Risk Parity', description: 'Equal risk contribution from all assets', category: 'convex' },
  { type: 'cvar_parity', name: 'CVaR Parity', description: 'Equal CVaR contribution from all assets', category: 'convex' },
  { type: 'max_diversification', name: 'Max Diversification', description: 'Maximizes the diversification ratio', category: 'convex' },
  { type: 'hrp', name: 'HRP', description: 'Hierarchical Risk Parity — tree-based allocation', category: 'hierarchical' },
  { type: 'herc', name: 'HERC', description: 'Hierarchical Equal Risk Contribution', category: 'hierarchical' },
  { type: 'equal_weighted', name: 'Equal Weighted', description: 'Simple 1/N allocation across assets', category: 'naive' },
];

// ── Portfolio Result ──

const WEIGHT_DATA: Array<{ ticker: string; name: string; sector: string; weight: number }> = [
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

export const MOCK_PORTFOLIO_RESULT: PortfolioResult = {
  weights: WEIGHT_DATA,
  metrics: {
    annualized_return: 0.12,
    annualized_volatility: 0.154,
    sharpe_ratio: 0.062,
    sortino_ratio: 0.089,
    max_drawdown: -0.128,
    cvar_95: -0.024,
    calmar_ratio: 0.937,
    tracking_error: null,
  },
  monthly_returns: [
    { month: '2023-03', return_pct: 0.032 }, { month: '2023-04', return_pct: 0.018 },
    { month: '2023-05', return_pct: -0.012 }, { month: '2023-06', return_pct: 0.041 },
    { month: '2023-07', return_pct: 0.028 }, { month: '2023-08', return_pct: -0.035 },
    { month: '2023-09', return_pct: -0.048 }, { month: '2023-10', return_pct: -0.022 },
    { month: '2023-11', return_pct: 0.065 }, { month: '2023-12', return_pct: 0.042 },
    { month: '2024-01', return_pct: 0.015 }, { month: '2024-02', return_pct: 0.038 },
    { month: '2024-03', return_pct: 0.029 }, { month: '2024-04', return_pct: -0.018 },
    { month: '2024-05', return_pct: 0.022 }, { month: '2024-06', return_pct: 0.035 },
    { month: '2024-07', return_pct: -0.008 }, { month: '2024-08', return_pct: 0.019 },
    { month: '2024-09', return_pct: -0.014 }, { month: '2024-10', return_pct: -0.031 },
    { month: '2024-11', return_pct: 0.052 }, { month: '2024-12', return_pct: 0.038 },
    { month: '2025-01', return_pct: 0.012 }, { month: '2025-02', return_pct: 0.025 },
    { month: '2025-03', return_pct: -0.009 }, { month: '2025-04', return_pct: 0.033 },
    { month: '2025-05', return_pct: 0.016 }, { month: '2025-06', return_pct: -0.021 },
    { month: '2025-07', return_pct: 0.045 }, { month: '2025-08', return_pct: -0.028 },
    { month: '2025-09', return_pct: 0.008 }, { month: '2025-10', return_pct: 0.037 },
    { month: '2025-11', return_pct: 0.024 }, { month: '2025-12', return_pct: 0.031 },
    { month: '2026-01', return_pct: -0.005 }, { month: '2026-02', return_pct: 0.018 },
  ],
  sector_allocations: [
    { sector: 'Technology', weight: 0.205 },
    { sector: 'Financial Services', weight: 0.125 },
    { sector: 'Healthcare', weight: 0.140 },
    { sector: 'Consumer Defensive', weight: 0.095 },
    { sector: 'Consumer Cyclical', weight: 0.085 },
    { sector: 'Industrials', weight: 0.105 },
    { sector: 'Energy', weight: 0.065 },
    { sector: 'Utilities', weight: 0.060 },
    { sector: 'Communication', weight: 0.055 },
    { sector: 'Real Estate', weight: 0.045 },
  ].sort((a, b) => b.weight - a.weight),
  backtest_cumulative: generateBacktestCumulative(),
  backtest_metrics: {
    annualized_return: 0.098,
    annualized_volatility: 0.162,
    sharpe_ratio: 0.048,
    sortino_ratio: 0.071,
    max_drawdown: -0.148,
    cvar_95: -0.028,
    calmar_ratio: 0.662,
    tracking_error: 0.032,
  },
};

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

function generateBacktestCumulative(): BacktestPoint[] {
  const points: BacktestPoint[] = [];
  let cum = 0;
  const start = new Date('2023-03-01');
  for (let i = 0; i < 36; i++) {
    const d = new Date(start);
    d.setMonth(d.getMonth() + i);
    const monthly = (Math.random() - 0.42) * 0.06;
    cum += monthly;
    points.push({ date: d.toISOString().split('T')[0], cumulative_return: Math.round(cum * 10000) / 10000 });
  }
  return points;
}
