import { TestBed } from '@angular/core/testing';
import { FormatReturnPipe } from './format-return.pipe';

describe('FormatReturnPipe', () => {
  let pipe: FormatReturnPipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new FormatReturnPipe());
  });

  it('when value is null, the flat placeholder is returned', () => {
    expect(pipe.transform(null)).toEqual({ text: '--', colorClass: 'text-flat' });
  });

  it('when value is NaN, the flat placeholder is returned', () => {
    expect(pipe.transform(Number.NaN)).toEqual({ text: '--', colorClass: 'text-flat' });
  });

  it('when value is positive, a signed gain is returned', () => {
    expect(pipe.transform(0.05)).toEqual({ text: '+5.00%', colorClass: 'text-gain' });
  });

  it('when value is negative, a signed loss is returned', () => {
    expect(pipe.transform(-0.05)).toEqual({ text: '-5.00%', colorClass: 'text-loss' });
  });

  it('when value is zero, the flat class is returned', () => {
    expect(pipe.transform(0)).toEqual({ text: '0.00%', colorClass: 'text-flat' });
  });

  it('when value is Infinity, it passes through as a gain', () => {
    expect(pipe.transform(Number.POSITIVE_INFINITY)).toEqual({ text: '+Infinity%', colorClass: 'text-gain' });
  });
});
