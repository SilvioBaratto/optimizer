import { TestBed } from '@angular/core/testing';
import { FormatCurrencyPipe } from './format-currency.pipe';

describe('FormatCurrencyPipe', () => {
  let pipe: FormatCurrencyPipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new FormatCurrencyPipe());
  });

  it('when value is null, the dash placeholder is returned', () => {
    expect(pipe.transform(null)).toBe('--');
  });

  it('when value is NaN, the dash placeholder is returned', () => {
    expect(pipe.transform(Number.NaN)).toBe('--');
  });

  it('when value is a number, a USD currency string is returned', () => {
    expect(pipe.transform(1234.5)).toBe('$1,234.50');
  });

  it('when value is zero, the zero currency string is returned', () => {
    expect(pipe.transform(0)).toBe('$0.00');
  });

  it('when value is negative, the sign is preserved', () => {
    expect(pipe.transform(-10)).toBe('-$10.00');
  });

  it('when a currency code is supplied, it overrides the default', () => {
    expect(pipe.transform(10, 'EUR')).toBe('€10.00');
  });
});
