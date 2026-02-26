import {
  PipelineNode,
  MomentEstimationOutput,
  ViewFormation,
  EfficientFrontierPoint,
  OptimizationConfig,
  StrategyComparison,
} from '../models/optimization.model';
import { seededRandom } from './generators';

// ── Pipeline Nodes ──

export const MOCK_PIPELINE_NODES: PipelineNode[] = [
  { id: 'p1', label: 'Data Validation', status: 'completed', duration_ms: 120, detail: '85 assets validated' },
  { id: 'p2', label: 'Pre-Selection', status: 'completed', duration_ms: 340, detail: 'Reduced to 65 assets' },
  { id: 'p3', label: 'Moment Estimation', status: 'completed', duration_ms: 890, detail: 'Ledoit-Wolf covariance' },
  { id: 'p4', label: 'View Formation', status: 'completed', duration_ms: 210, detail: '5 Black-Litterman views' },
  { id: 'p5', label: 'Optimization', status: 'running', duration_ms: 1540, detail: 'Max Sharpe solver' },
  { id: 'p6', label: 'Validation', status: 'pending', detail: 'Walk-forward CV' },
  { id: 'p7', label: 'Rebalancing', status: 'pending', detail: 'Threshold check' },
];

// ── Moment Estimation Output ──

export const MOCK_MOMENT_OUTPUT: MomentEstimationOutput = {
  muEstimator: 'shrunk',
  covEstimator: 'ledoit_wolf',
  expectedReturns: [
    { ticker: 'AAPL', value: 0.112 },
    { ticker: 'MSFT', value: 0.098 },
    { ticker: 'GOOGL', value: 0.105 },
    { ticker: 'AMZN', value: 0.118 },
    { ticker: 'NVDA', value: 0.142 },
    { ticker: 'META', value: 0.125 },
    { ticker: 'JPM', value: 0.082 },
    { ticker: 'V', value: 0.091 },
    { ticker: 'JNJ', value: 0.058 },
    { ticker: 'WMT', value: 0.064 },
    { ticker: 'XOM', value: 0.076 },
    { ticker: 'TSLA', value: 0.168 },
    { ticker: 'KO', value: 0.052 },
    { ticker: 'NEE', value: 0.071 },
    { ticker: 'UNH', value: 0.088 },
  ],
  topEigenvalues: [0.342, 0.128, 0.085, 0.062, 0.041, 0.033, 0.028, 0.022, 0.018, 0.015],
};

// ── View Formations (Black-Litterman) ──

export const MOCK_VIEW_FORMATIONS: ViewFormation[] = [
  { id: 'v1', type: 'absolute', assets: ['NVDA'], value: 0.15, confidence: 0.8, source: 'Analyst consensus' },
  { id: 'v2', type: 'relative', assets: ['AAPL', 'MSFT'], value: 0.02, confidence: 0.6, source: 'Factor model' },
  { id: 'v3', type: 'absolute', assets: ['XOM'], value: 0.08, confidence: 0.5, source: 'Macro outlook' },
  { id: 'v4', type: 'relative', assets: ['JPM', 'BAC'], value: 0.015, confidence: 0.7, source: 'Sector analysis' },
  { id: 'v5', type: 'absolute', assets: ['JNJ'], value: 0.055, confidence: 0.9, source: 'Defensive positioning' },
];

// ── Efficient Frontier (50 points) ──

export const MOCK_EFFICIENT_FRONTIER_POINTS: EfficientFrontierPoint[] = (() => {
  const rng = seededRandom(55);
  const points: EfficientFrontierPoint[] = [];
  // Parametric hyperbola: realistic concave "bullet" shape
  // Min-variance at sigma~8%, return~5.5%; curve steepens left, flattens right
  const sigmaMin = 0.08;
  const retAtMin = 0.055;
  for (let i = 0; i < 50; i++) {
    const t = i / 49; // 0..1
    const sigma = sigmaMin + t * 0.17; // 8% to 25%
    // Hyperbolic upper branch: steep initial rise, diminishing returns
    const ret = retAtMin + 0.38 * Math.sqrt(sigma - sigmaMin + 0.001)
      - 0.6 * (sigma - sigmaMin) + (rng() - 0.5) * 0.001;
    const sharpe = ret / sigma;
    points.push({
      risk: Math.round(sigma * 10000) / 10000,
      return: Math.round(ret * 10000) / 10000,
      sharpe: Math.round(sharpe * 1000) / 1000,
    });
  }
  // Mark key portfolios
  const optimal = points.reduce((best, p) => (p.sharpe > best.sharpe ? p : best), points[0]);
  optimal.label = 'Max Sharpe';
  points[0].label = 'Min Variance';
  return points;
})();

// ── Optimization Config ──

export const MOCK_OPTIMIZATION_CONFIG: OptimizationConfig = {
  strategy: 'max_sharpe',
  riskMeasure: 'variance',
  muEstimator: 'shrunk',
  covEstimator: 'ledoit_wolf',
  riskAversion: 1.0,
  cvarBeta: 0.95,
  robustKappa: 0,
  preSelection: true,
};

// ── Strategy Comparisons ──

export const MOCK_STRATEGY_COMPARISONS: StrategyComparison[] = [
  { strategy: 'Max Sharpe', annualizedReturn: 0.098, annualizedVol: 0.154, sharpe: 0.636, maxDrawdown: -0.128, cvar95: -0.024 },
  { strategy: 'Min Variance', annualizedReturn: 0.072, annualizedVol: 0.108, sharpe: 0.667, maxDrawdown: -0.082, cvar95: -0.016 },
  { strategy: 'Risk Parity', annualizedReturn: 0.085, annualizedVol: 0.132, sharpe: 0.644, maxDrawdown: -0.105, cvar95: -0.020 },
];
