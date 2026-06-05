import { TestBed } from '@angular/core/testing';
import { FormatRatioPipe } from './format-ratio.pipe';

describe('FormatRatioPipe', () => {
  let pipe: FormatRatioPipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new FormatRatioPipe());
  });

  it('when value is null, the dash placeholder is returned', () => {
    expect(pipe.transform(null)).toBe('--');
  });

  it('when value is NaN, the dash placeholder is returned', () => {
    expect(pipe.transform(Number.NaN)).toBe('--');
  });

  it('when value is 2.5, two decimals are returned', () => {
    expect(pipe.transform(2.5)).toBe('2.50');
  });

  it('when value is zero, zero at two decimals is returned', () => {
    expect(pipe.transform(0)).toBe('0.00');
  });

  it('when value is negative, the sign is preserved', () => {
    expect(pipe.transform(-1.5)).toBe('-1.50');
  });

  it('when a decimals argument is supplied, the precision changes', () => {
    expect(pipe.transform(2.5, 1)).toBe('2.5');
  });

  it('when value is Infinity, it passes through as Infinity', () => {
    expect(pipe.transform(Number.POSITIVE_INFINITY)).toBe('Infinity');
  });
});
