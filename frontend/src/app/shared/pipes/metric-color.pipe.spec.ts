import { TestBed } from '@angular/core/testing';
import { MetricColorPipe, safeNum } from './metric-color.pipe';

describe('MetricColorPipe', () => {
  let pipe: MetricColorPipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new MetricColorPipe());
  });

  describe('return kind', () => {
    it('when value is positive, signed percent text with gain class is returned', () => {
      expect(pipe.transform(0.0123, 'return')).toEqual({ text: '+1.23%', colorClass: 'text-gain' });
    });

    it('when value is negative, signed percent text with loss class is returned', () => {
      expect(pipe.transform(-0.045, 'return')).toEqual({ text: '-4.50%', colorClass: 'text-loss' });
    });

    it('when value is zero, percent text with flat class is returned', () => {
      expect(pipe.transform(0, 'return')).toEqual({ text: '0.00%', colorClass: 'text-flat' });
    });
  });

  describe('percent kind', () => {
    it('when value is positive, unsigned percent text with gain class is returned', () => {
      expect(pipe.transform(0.15, 'percent')).toEqual({ text: '15.00%', colorClass: 'text-gain' });
    });

    it('when value is negative, unsigned percent text with loss class is returned', () => {
      expect(pipe.transform(-0.07, 'percent')).toEqual({ text: '-7.00%', colorClass: 'text-loss' });
    });
  });

  describe('ratio kind', () => {
    it('when value is positive, decimal text with gain class is returned', () => {
      expect(pipe.transform(1.234, 'ratio')).toEqual({ text: '1.23', colorClass: 'text-gain' });
    });

    it('when value is negative, decimal text with loss class is returned', () => {
      expect(pipe.transform(-0.5, 'ratio')).toEqual({ text: '-0.50', colorClass: 'text-loss' });
    });
  });

  describe('bps kind', () => {
    it('when value is positive, bps text with gain class is returned', () => {
      expect(pipe.transform(0.0025, 'bps')).toEqual({ text: '25 bps', colorClass: 'text-gain' });
    });

    it('when value is negative, bps text with loss class is returned', () => {
      expect(pipe.transform(-0.001, 'bps')).toEqual({ text: '-10 bps', colorClass: 'text-loss' });
    });
  });

  describe('null fallback', () => {
    it('when value is null, dash text with empty colorClass is returned', () => {
      expect(pipe.transform(null, 'return')).toEqual({ text: '--', colorClass: '' });
    });

    it('when value is undefined, dash text with empty colorClass is returned', () => {
      expect(pipe.transform(undefined, 'percent')).toEqual({ text: '--', colorClass: '' });
    });

    it('when value is NaN, dash text with empty colorClass is returned', () => {
      expect(pipe.transform(Number.NaN, 'ratio')).toEqual({ text: '--', colorClass: '' });
    });

    it('when value is Infinity, dash text with empty colorClass is returned', () => {
      expect(pipe.transform(Number.POSITIVE_INFINITY, 'bps')).toEqual({ text: '--', colorClass: '' });
    });

    it('when value is a non-number, dash text with empty colorClass is returned', () => {
      expect(pipe.transform('0.5' as unknown, 'return')).toEqual({ text: '--', colorClass: '' });
    });
  });

  describe('default kind', () => {
    it('when kind argument is omitted, return formatting is applied', () => {
      expect(pipe.transform(0.01)).toEqual({ text: '+1.00%', colorClass: 'text-gain' });
    });
  });
});

describe('safeNum', () => {
  it('when value is a finite number, the number is returned', () => {
    expect(safeNum(1.5)).toBe(1.5);
    expect(safeNum(0)).toBe(0);
    expect(safeNum(-0)).toBe(-0);
  });

  it('when value is null/undefined/NaN/Infinity/non-number, null is returned', () => {
    expect(safeNum(null)).toBeNull();
    expect(safeNum(undefined)).toBeNull();
    expect(safeNum(Number.NaN)).toBeNull();
    expect(safeNum(Number.POSITIVE_INFINITY)).toBeNull();
    expect(safeNum(Number.NEGATIVE_INFINITY)).toBeNull();
    expect(safeNum('1')).toBeNull();
    expect(safeNum({})).toBeNull();
  });
});
