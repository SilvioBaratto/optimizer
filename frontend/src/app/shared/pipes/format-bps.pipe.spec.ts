import { TestBed } from '@angular/core/testing';
import { FormatBpsPipe } from './format-bps.pipe';

describe('FormatBpsPipe', () => {
  let pipe: FormatBpsPipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new FormatBpsPipe());
  });

  it('when value is null, the dash placeholder is returned', () => {
    expect(pipe.transform(null)).toBe('--');
  });

  it('when value is undefined, the dash placeholder is returned', () => {
    expect(pipe.transform(undefined)).toBe('--');
  });

  it('when value is NaN, the dash placeholder is returned', () => {
    expect(pipe.transform(Number.NaN)).toBe('--');
  });

  it('when value is 0.01, 100 bps is returned', () => {
    expect(pipe.transform(0.01)).toBe('100 bps');
  });

  it('when value is zero, 0 bps is returned', () => {
    expect(pipe.transform(0)).toBe('0 bps');
  });

  it('when value is negative, signed bps is returned', () => {
    expect(pipe.transform(-0.001)).toBe('-10 bps');
  });

  it('when value is Infinity, it passes through as Infinity bps', () => {
    expect(pipe.transform(Number.POSITIVE_INFINITY)).toBe('Infinity bps');
  });
});
