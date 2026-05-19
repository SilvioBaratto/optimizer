import { tickerColorMap } from './ticker-color-map';

describe('tickerColorMap', () => {
  const PALETTE = [
    '#111111',
    '#222222',
    '#333333',
    '#444444',
    '#555555',
    '#666666',
    '#777777',
    '#888888',
  ];

  beforeAll(() => {
    PALETTE.forEach((value, index) => {
      document.documentElement.style.setProperty(`--color-chart-${index + 1}`, value);
    });
  });

  afterAll(() => {
    PALETTE.forEach((_, index) => {
      document.documentElement.style.removeProperty(`--color-chart-${index + 1}`);
    });
  });

  it('when input is empty, an empty mapping is returned', () => {
    expect(tickerColorMap([])).toEqual({});
  });

  it('when input has a single ticker, it maps to the first chart token', () => {
    expect(tickerColorMap(['AAPL'])).toEqual({ AAPL: PALETTE[0] });
  });

  it('when input has fewer than eight tickers, each maps to a distinct token by order', () => {
    const tickers = ['A', 'B', 'C', 'D'];
    expect(tickerColorMap(tickers)).toEqual({
      A: PALETTE[0],
      B: PALETTE[1],
      C: PALETTE[2],
      D: PALETTE[3],
    });
  });

  it('when input has nine tickers, the ninth wraps to the first chart token (mod 8)', () => {
    const tickers = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9'];
    const map = tickerColorMap(tickers);
    expect(map['T9']).toBe(PALETTE[0]);
    expect(map['T1']).toBe(PALETTE[0]);
    expect(map['T8']).toBe(PALETTE[7]);
  });

  it('when called twice with the same input order, the same mapping is returned', () => {
    const tickers = ['MSFT', 'GOOG', 'NVDA'];
    expect(tickerColorMap(tickers)).toEqual(tickerColorMap(tickers));
  });

  it('when input has seventeen tickers, indices wrap twice through the palette', () => {
    const tickers = Array.from({ length: 17 }, (_, i) => `S${i}`);
    const map = tickerColorMap(tickers);
    expect(map['S0']).toBe(PALETTE[0]);
    expect(map['S8']).toBe(PALETTE[0]);
    expect(map['S16']).toBe(PALETTE[0]);
    expect(map['S15']).toBe(PALETTE[7]);
  });
});
