import { computeDrawdownSeries } from './drawdown';

describe('computeDrawdownSeries', () => {
  it('returns an empty array for empty input', () => {
    expect(computeDrawdownSeries([])).toEqual([]);
  });

  it('returns zero drawdown for a monotonically increasing curve', () => {
    const result = computeDrawdownSeries([
      { date: '2024-01-01', value: 100 },
      { date: '2024-01-02', value: 105 },
      { date: '2024-01-03', value: 110 },
    ]);
    expect(result.every((p) => p.drawdown === 0)).toBe(true);
  });

  it('computes the fractional drawdown from the running peak', () => {
    const result = computeDrawdownSeries([
      { date: '2024-01-01', value: 100 },
      { date: '2024-01-02', value: 110 },
      { date: '2024-01-03', value: 88 },
      { date: '2024-01-04', value: 99 },
    ]);
    expect(result[0].drawdown).toBe(0);
    expect(result[1].drawdown).toBe(0);
    expect(result[2].drawdown).toBeCloseTo(-0.2, 10);
    expect(result[3].drawdown).toBeCloseTo(-0.1, 10);
  });

  it('preserves the date of each input point', () => {
    const result = computeDrawdownSeries([
      { date: '2024-01-01', value: 100 },
      { date: '2024-01-02', value: 99 },
    ]);
    expect(result.map((p) => p.date)).toEqual(['2024-01-01', '2024-01-02']);
  });
});
