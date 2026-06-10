import { computeQuintiles, deriveSelectRows } from './factor.model';
import type { FactorSelectApiResponse } from './factor.model';

describe('computeQuintiles', () => {
  it('when scores is empty, an empty array is returned', () => {
    expect(computeQuintiles({})).toEqual([]);
  });

  it('when five distinct scores are ranked, every quintile 1–5 is assigned', () => {
    const result = computeQuintiles({ A: 5, B: 4, C: 3, D: 2, E: 1 });
    expect(result.map((r) => r.quintile)).toEqual([1, 2, 3, 4, 5]);
  });

  it('when scores tie, the descending sort preserves input order', () => {
    const result = computeQuintiles({ AAA: 3, BBB: 3 });
    expect(result.map((r) => r.ticker)).toEqual(['AAA', 'BBB']);
  });
});

describe('deriveSelectRows', () => {
  it('when response is null, an empty array is returned', () => {
    expect(deriveSelectRows(null)).toEqual([]);
  });

  it('when two rows share a status, ticker localeCompare breaks the tie alphabetically', () => {
    const response: FactorSelectApiResponse = {
      selected_tickers: ['MSFT', 'AAPL'],
      count: 2,
      turnover: null,
      buffer_zone: { entered: [], exited: [] },
    };
    const result = deriveSelectRows(response, []);
    expect(result[0].ticker).toBe('AAPL');
    expect(result[1].ticker).toBe('MSFT');
    expect(result[0].status).toBe('Selected');
    expect(result[1].status).toBe('Selected');
  });
});
