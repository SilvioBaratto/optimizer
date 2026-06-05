import { TestBed } from '@angular/core/testing';
import { FormatPercentPipe } from './format-percent.pipe';

describe('FormatPercentPipe', () => {
  let pipe: FormatPercentPipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new FormatPercentPipe());
  });

  it('when value is null, the dash placeholder is returned', () => {
    expect(pipe.transform(null)).toBe('--');
  });

  it('when value is NaN, the dash placeholder is returned', () => {
    expect(pipe.transform(Number.NaN)).toBe('--');
  });

  it('when value is 0.1, ten percent at two decimals is returned', () => {
    expect(pipe.transform(0.1)).toBe('10.00%');
  });

  it('when value is zero, zero percent is returned', () => {
    expect(pipe.transform(0)).toBe('0.00%');
  });

  it('when value is negative, the sign is preserved', () => {
    expect(pipe.transform(-0.05)).toBe('-5.00%');
  });

  it('when a decimals argument is supplied, the precision changes', () => {
    expect(pipe.transform(0.1, 0)).toBe('10%');
  });

  it('when value is Infinity, it passes through as Infinity percent', () => {
    expect(pipe.transform(Number.POSITIVE_INFINITY)).toBe('Infinity%');
  });
});
