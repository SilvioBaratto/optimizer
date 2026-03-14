import type {
  MacroCalibrationResponse,
  FredObservationPoint,
  MacroNewsItem,
  MacroNewsTheme,
  SectorPhaseRow,
  CountryMacroData,
  BondYieldSnapshot,
  BusinessCyclePhase,
  CompositeScorePoint,
} from '../models/macro-intelligence.model';

// ── Macro Calibration (from /api/v1/views/macro-calibration) ──

export const MOCK_MACRO_CALIBRATION: MacroCalibrationResponse = {
  phase: 'MID_EXPANSION',
  delta: 2.5,
  tau: 0.05,
  confidence: 0.78,
  rationale:
    'The US economy is in a mid-expansion phase characterised by above-trend GDP growth (2.8% annualised), a steepening yield curve (2s10s at +42bp), and contained credit spreads (HY OAS 385bp). ISM Manufacturing PMI at 52.8 signals continued industrial expansion, though the pace of acceleration is moderating from Q4 peaks. Labour market remains tight with unemployment at 3.9%, supporting consumer spending. The Federal Reserve has paused rate hikes, with markets pricing one cut by Q3 2026. European growth is lagging with Germany near stagnation, while UK shows tentative recovery. Overall regime supports moderate risk-taking with a tilt toward cyclical sectors and value factors.',
  macro_summary:
    'GDP: 2.8% (US), 0.8% (EU), 1.1% (UK) | Inflation: 2.4% (US), 2.1% (EU), 2.6% (UK) | Rates: 5.25% (Fed), 3.75% (ECB), 4.50% (BoE) | PMI: 52.8 (US), 47.2 (DE), 51.1 (UK) | VIX: 15.8 | HY OAS: 385bp | 2s10s: +42bp',
  timestamp: '2026-03-04T08:00:00Z',
};

// ── FRED Time-Series Data ──

function generateFredSeries(
  base: number,
  volatility: number,
  trend: number,
  count: number,
  startDate: string,
): FredObservationPoint[] {
  const points: FredObservationPoint[] = [];
  let value = base;
  const start = new Date(startDate);
  for (let i = 0; i < count; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    value += trend + (Math.random() - 0.5) * volatility;
    value = Math.max(0, value);
    points.push({
      date: d.toISOString().split('T')[0],
      value: Math.round(value * 100) / 100,
    });
  }
  return points;
}

export const MOCK_FRED_PMI: FredObservationPoint[] = generateFredSeries(51, 1.2, 0.02, 744, '2024-03-01');
export const MOCK_FRED_YIELD_SPREAD: FredObservationPoint[] = generateFredSeries(0.2, 0.15, 0.003, 105, '2025-12-01');
export const MOCK_FRED_HY_OAS: FredObservationPoint[] = generateFredSeries(400, 20, -0.5, 105, '2025-12-01');
export const MOCK_FRED_VIX: FredObservationPoint[] = generateFredSeries(16, 3, -0.02, 105, '2025-12-01');

// ── Country Macro Data ──

export const MOCK_COUNTRY_DATA: CountryMacroData[] = [
  {
    country: 'USA',
    country_code: 'US',
    gdp_growth: 2.8,
    inflation: 2.4,
    unemployment: 3.9,
    interest_rate: 5.25,
    yield_10y: 4.32,
    yield_2y: 3.90,
    yield_spread_bps: 42,
    yield_sparkline: [4.1, 4.15, 4.2, 4.18, 4.25, 4.3, 4.28, 4.35, 4.32, 4.3, 4.28, 4.32],
  },
  {
    country: 'Germany',
    country_code: 'DE',
    gdp_growth: 0.3,
    inflation: 2.1,
    unemployment: 5.8,
    interest_rate: 3.75,
    yield_10y: 2.45,
    yield_2y: 2.62,
    yield_spread_bps: -17,
    yield_sparkline: [2.3, 2.35, 2.4, 2.38, 2.42, 2.48, 2.45, 2.5, 2.47, 2.44, 2.42, 2.45],
  },
  {
    country: 'France',
    country_code: 'FR',
    gdp_growth: 0.8,
    inflation: 2.3,
    unemployment: 7.4,
    interest_rate: 3.75,
    yield_10y: 3.15,
    yield_2y: 2.85,
    yield_spread_bps: 30,
    yield_sparkline: [3.0, 3.05, 3.1, 3.08, 3.12, 3.18, 3.15, 3.2, 3.17, 3.14, 3.12, 3.15],
  },
  {
    country: 'UK',
    country_code: 'GB',
    gdp_growth: 1.1,
    inflation: 2.6,
    unemployment: 4.2,
    interest_rate: 4.50,
    yield_10y: 4.18,
    yield_2y: 3.95,
    yield_spread_bps: 23,
    yield_sparkline: [4.0, 4.05, 4.1, 4.08, 4.12, 4.2, 4.18, 4.22, 4.19, 4.16, 4.15, 4.18],
  },
];

// ── Bond Yield Snapshots (for yield curve chart) ──

const US_MATURITIES = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'];

export const MOCK_BOND_YIELDS_TODAY: BondYieldSnapshot[] = [
  {
    country: 'US',
    date: '2026-03-04',
    curve: [
      { maturity: '1M', yield_pct: 5.18 }, { maturity: '3M', yield_pct: 5.12 },
      { maturity: '6M', yield_pct: 4.95 }, { maturity: '1Y', yield_pct: 4.62 },
      { maturity: '2Y', yield_pct: 3.90 }, { maturity: '3Y', yield_pct: 3.78 },
      { maturity: '5Y', yield_pct: 3.85 }, { maturity: '7Y', yield_pct: 4.05 },
      { maturity: '10Y', yield_pct: 4.32 }, { maturity: '20Y', yield_pct: 4.58 },
      { maturity: '30Y', yield_pct: 4.52 },
    ],
  },
  {
    country: 'DE',
    date: '2026-03-04',
    curve: [
      { maturity: '1M', yield_pct: 3.55 }, { maturity: '3M', yield_pct: 3.48 },
      { maturity: '6M', yield_pct: 3.30 }, { maturity: '1Y', yield_pct: 3.05 },
      { maturity: '2Y', yield_pct: 2.62 }, { maturity: '3Y', yield_pct: 2.50 },
      { maturity: '5Y', yield_pct: 2.38 }, { maturity: '7Y', yield_pct: 2.42 },
      { maturity: '10Y', yield_pct: 2.45 }, { maturity: '20Y', yield_pct: 2.55 },
      { maturity: '30Y', yield_pct: 2.48 },
    ],
  },
  {
    country: 'FR',
    date: '2026-03-04',
    curve: [
      { maturity: '1M', yield_pct: 3.58 }, { maturity: '3M', yield_pct: 3.52 },
      { maturity: '6M', yield_pct: 3.35 }, { maturity: '1Y', yield_pct: 3.12 },
      { maturity: '2Y', yield_pct: 2.85 }, { maturity: '3Y', yield_pct: 2.78 },
      { maturity: '5Y', yield_pct: 2.85 }, { maturity: '7Y', yield_pct: 2.98 },
      { maturity: '10Y', yield_pct: 3.15 }, { maturity: '20Y', yield_pct: 3.45 },
      { maturity: '30Y', yield_pct: 3.38 },
    ],
  },
  {
    country: 'GB',
    date: '2026-03-04',
    curve: [
      { maturity: '1M', yield_pct: 4.35 }, { maturity: '3M', yield_pct: 4.28 },
      { maturity: '6M', yield_pct: 4.15 }, { maturity: '1Y', yield_pct: 4.02 },
      { maturity: '2Y', yield_pct: 3.95 }, { maturity: '3Y', yield_pct: 3.88 },
      { maturity: '5Y', yield_pct: 3.92 }, { maturity: '7Y', yield_pct: 4.05 },
      { maturity: '10Y', yield_pct: 4.18 }, { maturity: '20Y', yield_pct: 4.42 },
      { maturity: '30Y', yield_pct: 4.35 },
    ],
  },
];

export const MOCK_BOND_YIELDS_1Y_AGO: BondYieldSnapshot[] = [
  {
    country: 'US',
    date: '2025-03-04',
    curve: [
      { maturity: '1M', yield_pct: 5.30 }, { maturity: '3M', yield_pct: 5.28 },
      { maturity: '6M', yield_pct: 5.20 }, { maturity: '1Y', yield_pct: 4.95 },
      { maturity: '2Y', yield_pct: 4.52 }, { maturity: '3Y', yield_pct: 4.35 },
      { maturity: '5Y', yield_pct: 4.20 }, { maturity: '7Y', yield_pct: 4.25 },
      { maturity: '10Y', yield_pct: 4.30 }, { maturity: '20Y', yield_pct: 4.55 },
      { maturity: '30Y', yield_pct: 4.48 },
    ],
  },
  {
    country: 'DE',
    date: '2025-03-04',
    curve: [
      { maturity: '1M', yield_pct: 3.80 }, { maturity: '3M', yield_pct: 3.72 },
      { maturity: '6M', yield_pct: 3.55 }, { maturity: '1Y', yield_pct: 3.30 },
      { maturity: '2Y', yield_pct: 2.85 }, { maturity: '3Y', yield_pct: 2.70 },
      { maturity: '5Y', yield_pct: 2.55 }, { maturity: '7Y', yield_pct: 2.50 },
      { maturity: '10Y', yield_pct: 2.48 }, { maturity: '20Y', yield_pct: 2.52 },
      { maturity: '30Y', yield_pct: 2.45 },
    ],
  },
  {
    country: 'FR',
    date: '2025-03-04',
    curve: [
      { maturity: '1M', yield_pct: 3.82 }, { maturity: '3M', yield_pct: 3.75 },
      { maturity: '6M', yield_pct: 3.60 }, { maturity: '1Y', yield_pct: 3.38 },
      { maturity: '2Y', yield_pct: 3.05 }, { maturity: '3Y', yield_pct: 2.95 },
      { maturity: '5Y', yield_pct: 2.92 }, { maturity: '7Y', yield_pct: 3.00 },
      { maturity: '10Y', yield_pct: 3.12 }, { maturity: '20Y', yield_pct: 3.40 },
      { maturity: '30Y', yield_pct: 3.32 },
    ],
  },
  {
    country: 'GB',
    date: '2025-03-04',
    curve: [
      { maturity: '1M', yield_pct: 4.55 }, { maturity: '3M', yield_pct: 4.48 },
      { maturity: '6M', yield_pct: 4.38 }, { maturity: '1Y', yield_pct: 4.25 },
      { maturity: '2Y', yield_pct: 4.15 }, { maturity: '3Y', yield_pct: 4.08 },
      { maturity: '5Y', yield_pct: 4.05 }, { maturity: '7Y', yield_pct: 4.12 },
      { maturity: '10Y', yield_pct: 4.22 }, { maturity: '20Y', yield_pct: 4.48 },
      { maturity: '30Y', yield_pct: 4.40 },
    ],
  },
];

// ── FRED Credit Spread Series (for Band C Tab 2) ──

export const MOCK_FRED_IG_OAS: FredObservationPoint[] = generateFredSeries(95, 8, -0.1, 105, '2025-12-01');

// ── Composite Score History (monthly S_t over 24 months) ──

export const MOCK_COMPOSITE_HISTORY: CompositeScorePoint[] = [
  { month: '2024-04', score: 1 }, { month: '2024-05', score: 2 }, { month: '2024-06', score: 2 },
  { month: '2024-07', score: 1 }, { month: '2024-08', score: 0 }, { month: '2024-09', score: -1 },
  { month: '2024-10', score: -2 }, { month: '2024-11', score: -1 }, { month: '2024-12', score: 0 },
  { month: '2025-01', score: 1 }, { month: '2025-02', score: 1 }, { month: '2025-03', score: 2 },
  { month: '2025-04', score: 3 }, { month: '2025-05', score: 2 }, { month: '2025-06', score: 2 },
  { month: '2025-07', score: 1 }, { month: '2025-08', score: 1 }, { month: '2025-09', score: 0 },
  { month: '2025-10', score: 1 }, { month: '2025-11', score: 2 }, { month: '2025-12', score: 2 },
  { month: '2026-01', score: 2 }, { month: '2026-02', score: 2 }, { month: '2026-03', score: 2 },
];

// ── News Themes ──

export const MOCK_NEWS_THEMES: MacroNewsTheme[] = [
  { id: 'monetary_policy', label: 'Monetary Policy', count: 12 },
  { id: 'inflation', label: 'Inflation', count: 8 },
  { id: 'growth', label: 'Growth & GDP', count: 10 },
  { id: 'labour', label: 'Labour Market', count: 6 },
  { id: 'credit', label: 'Credit & Spreads', count: 5 },
  { id: 'geopolitics', label: 'Geopolitics', count: 7 },
  { id: 'trade', label: 'Trade & Tariffs', count: 4 },
  { id: 'housing', label: 'Housing', count: 3 },
  { id: 'energy', label: 'Energy & Commodities', count: 6 },
];

// ── News Articles ──

export const MOCK_NEWS_ITEMS: MacroNewsItem[] = [
  { id: 'n1', headline: 'Fed holds rates steady, signals potential cut in Q3', snippet: 'The Federal Reserve maintained the federal funds rate at 5.25%, with Chair Powell noting progress on inflation but emphasising data-dependency for future decisions.', url: '#', publisher: 'Reuters', published_at: '2026-03-04T06:30:00Z', theme: 'monetary_policy' },
  { id: 'n2', headline: 'US ISM Manufacturing PMI rises to 52.8 in February', snippet: 'The Institute for Supply Management reported manufacturing PMI at 52.8, the fourth consecutive month of expansion, driven by new orders and production gains.', url: '#', publisher: 'Bloomberg', published_at: '2026-03-03T14:00:00Z', theme: 'growth' },
  { id: 'n3', headline: 'Euro area CPI holds at 2.1% in February', snippet: 'Eurostat confirmed annual inflation in the euro area at 2.1%, near the ECB target. Core inflation eased to 2.6% from 2.8%.', url: '#', publisher: 'Financial Times', published_at: '2026-03-03T10:00:00Z', theme: 'inflation' },
  { id: 'n4', headline: 'US nonfarm payrolls add 215K in February', snippet: 'The Bureau of Labor Statistics reported 215,000 new jobs, slightly above consensus. Unemployment held steady at 3.9%.', url: '#', publisher: 'Reuters', published_at: '2026-03-02T13:30:00Z', theme: 'labour' },
  { id: 'n5', headline: 'High yield spreads narrow to 385bp amid risk-on sentiment', snippet: 'The ICE BofA High Yield Option-Adjusted Spread tightened to 385bp, its lowest level since mid-2024, as investors rotate into credit.', url: '#', publisher: 'Bloomberg', published_at: '2026-03-02T09:00:00Z', theme: 'credit' },
  { id: 'n6', headline: 'ECB signals further easing as eurozone growth stalls', snippet: 'ECB President Lagarde highlighted downside risks to the eurozone outlook, with Germany contracting 0.1% in Q4 and France growing just 0.2%.', url: '#', publisher: 'Financial Times', published_at: '2026-03-01T16:00:00Z', theme: 'monetary_policy' },
  { id: 'n7', headline: 'China trade surplus widens on weak imports', snippet: 'China reported a trade surplus of $95.2 billion in February, as imports fell 8.2% year-on-year reflecting tepid domestic demand.', url: '#', publisher: 'Reuters', published_at: '2026-03-01T08:00:00Z', theme: 'trade' },
  { id: 'n8', headline: 'Brent crude rises above $82 on OPEC+ supply discipline', snippet: 'Oil prices climbed as OPEC+ signalled continuation of production cuts through Q2, while US stockpiles fell more than expected.', url: '#', publisher: 'Bloomberg', published_at: '2026-02-28T15:00:00Z', theme: 'energy' },
  { id: 'n9', headline: 'UK GDP grows 0.3% in Q4, beating expectations', snippet: 'The UK economy expanded 0.3% quarter-on-quarter in Q4 2025, avoiding a technical recession and buoyed by services sector growth.', url: '#', publisher: 'Financial Times', published_at: '2026-02-28T09:30:00Z', theme: 'growth' },
  { id: 'n10', headline: 'US housing starts fall 4.2% in January', snippet: 'Residential construction activity declined amid elevated mortgage rates, though permits rose 2.1% suggesting future recovery.', url: '#', publisher: 'Reuters', published_at: '2026-02-27T13:00:00Z', theme: 'housing' },
  { id: 'n11', headline: 'VIX drops to 15.8 as equity volatility subsides', snippet: 'The CBOE Volatility Index fell to 15.8, well below its long-term average of 20, reflecting complacency in equity markets.', url: '#', publisher: 'Bloomberg', published_at: '2026-02-27T10:00:00Z', theme: 'credit' },
  { id: 'n12', headline: 'Germany manufacturing PMI drops to 47.2', snippet: 'German factory activity contracted for the 20th consecutive month, weighed down by weak export demand and high energy costs.', url: '#', publisher: 'Financial Times', published_at: '2026-02-26T09:00:00Z', theme: 'growth' },
  { id: 'n13', headline: 'US core PCE inflation eases to 2.4% in January', snippet: 'The Fed preferred inflation gauge moderated to 2.4% annually, down from 2.6%, supporting the case for rate cuts later in 2026.', url: '#', publisher: 'Reuters', published_at: '2026-02-26T08:30:00Z', theme: 'inflation' },
  { id: 'n14', headline: 'BoE holds at 4.50%, dovish tilt in minutes', snippet: 'The Bank of England kept rates unchanged but the MPC voted 7-2, with two members preferring a cut, signalling easing bias.', url: '#', publisher: 'Bloomberg', published_at: '2026-02-25T12:00:00Z', theme: 'monetary_policy' },
  { id: 'n15', headline: 'Middle East tensions push safe-haven flows into bunds', snippet: 'German 10-year yields fell 8bp as escalating regional tensions drove investors toward European government bonds.', url: '#', publisher: 'Financial Times', published_at: '2026-02-25T08:00:00Z', theme: 'geopolitics' },
  { id: 'n16', headline: 'US consumer confidence rebounds in February', snippet: 'The Conference Board Consumer Confidence Index rose to 108.2 from 104.5, driven by improved labor market perceptions.', url: '#', publisher: 'Reuters', published_at: '2026-02-24T15:00:00Z', theme: 'growth' },
  { id: 'n17', headline: 'EU proposes new tariffs on Chinese EVs', snippet: 'The European Commission announced provisional tariffs of up to 38% on Chinese electric vehicle imports, escalating trade tensions.', url: '#', publisher: 'Financial Times', published_at: '2026-02-24T10:00:00Z', theme: 'trade' },
  { id: 'n18', headline: 'Natural gas prices surge 12% on cold weather forecast', snippet: 'European TTF natural gas futures jumped as forecasters predicted below-normal temperatures through mid-March.', url: '#', publisher: 'Bloomberg', published_at: '2026-02-23T14:00:00Z', theme: 'energy' },
  { id: 'n19', headline: 'French unemployment rises to 7.4% in Q4', snippet: 'France reported a modest uptick in unemployment, reflecting slowing economic momentum despite ECB monetary support.', url: '#', publisher: 'Reuters', published_at: '2026-02-23T09:00:00Z', theme: 'labour' },
  { id: 'n20', headline: 'IG credit spreads stable at 95bp despite equity rally', snippet: 'Investment-grade corporate bond spreads held firm, with new issuance well-absorbed by yield-seeking investors.', url: '#', publisher: 'Bloomberg', published_at: '2026-02-22T11:00:00Z', theme: 'credit' },
];

// ── Sector Rotation Table (static from theory chapter 07) ──

export const SECTOR_ROTATION_TABLE: SectorPhaseRow[] = [
  { sector: 'Energy', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Materials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', CONTRACTION: 'UW' } },
  { sector: 'Industrials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Consumer Disc.', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', CONTRACTION: 'UW' } },
  { sector: 'Consumer Staples', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'UW', LATE_EXPANSION: 'N', CONTRACTION: 'OW' } },
  { sector: 'Health Care', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'N', LATE_EXPANSION: 'OW', CONTRACTION: 'OW' } },
  { sector: 'Financials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Info Technology', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'OW', CONTRACTION: 'N' } },
  { sector: 'Communication', phases: { EARLY_EXPANSION: 'N', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Utilities', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'UW', LATE_EXPANSION: 'N', CONTRACTION: 'OW' } },
  { sector: 'Real Estate', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', CONTRACTION: 'N' } },
];
