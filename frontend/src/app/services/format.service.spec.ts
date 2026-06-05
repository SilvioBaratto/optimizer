import { TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../testing';
import { FormatService } from './format.service';

describe('FormatService', () => {
  let svc: FormatService;

  beforeEach(async () => {
    await configureTestBed({ withHttp: false });
    svc = TestBed.inject(FormatService);
  });

  it('when injected twice, the providedIn root singleton resolves once', () => {
    expect(TestBed.inject(FormatService)).toBe(svc);
  });

  describe('formatReturn', () => {
    it('when value is null, the flat placeholder is returned', () => {
      expect(svc.formatReturn(null)).toEqual({ text: '--', colorClass: 'text-flat' });
    });

    it('when value is NaN, the flat placeholder is returned', () => {
      expect(svc.formatReturn(NaN)).toEqual({ text: '--', colorClass: 'text-flat' });
    });

    it('when value is positive, a gain is returned', () => {
      expect(svc.formatReturn(0.05)).toEqual({ text: '+5.00%', colorClass: 'text-gain' });
    });

    it('when value is negative, a loss is returned', () => {
      expect(svc.formatReturn(-0.05)).toEqual({ text: '-5.00%', colorClass: 'text-loss' });
    });

    it('when value is zero, the flat class is returned', () => {
      expect(svc.formatReturn(0)).toEqual({ text: '0.00%', colorClass: 'text-flat' });
    });
  });

  describe('formatCurrency', () => {
    it('when value is null, the placeholder is returned', () => {
      expect(svc.formatCurrency(null)).toBe('--');
    });

    it('when value is a number, a USD string is returned', () => {
      expect(svc.formatCurrency(1234.5)).toBe('$1,234.50');
    });

    it('when value is negative, the sign is preserved', () => {
      expect(svc.formatCurrency(-10)).toBe('-$10.00');
    });

    it('when compact, a compact currency string is returned', () => {
      // Compact suffix is ICU-data dependent across runners; assert the parts.
      const out = svc.formatCurrencyCompact(1_500_000);
      expect(out).toContain('$');
      expect(out).toContain('1.5');
      expect(out).toContain('M');
    });

    it('when compact value is null, the placeholder is returned', () => {
      expect(svc.formatCurrencyCompact(null)).toBe('--');
    });
  });

  describe('formatBps', () => {
    it('when value is null, the placeholder is returned', () => {
      expect(svc.formatBps(null)).toBe('--');
    });

    it('when value is NaN, the placeholder is returned', () => {
      expect(svc.formatBps(NaN)).toBe('--');
    });

    it('when value is 0.01, 100 bps is returned', () => {
      expect(svc.formatBps(0.01)).toBe('100 bps');
    });

    it('when value is zero, 0 bps is returned', () => {
      expect(svc.formatBps(0)).toBe('0 bps');
    });
  });

  describe('formatRatio', () => {
    it('when value is null, the placeholder is returned', () => {
      expect(svc.formatRatio(null)).toBe('--');
    });

    it('when value is 2.5, two decimals are returned', () => {
      expect(svc.formatRatio(2.5)).toBe('2.50');
    });

    it('when decimals is overridden, the precision changes', () => {
      expect(svc.formatRatio(2.5, 1)).toBe('2.5');
    });
  });

  describe('formatPercent', () => {
    it('when value is null, the placeholder is returned', () => {
      expect(svc.formatPercent(null)).toBe('--');
    });

    it('when value is 0.1, ten percent is returned', () => {
      expect(svc.formatPercent(0.1)).toBe('10.00%');
    });

    it('when value is negative, the sign is preserved', () => {
      expect(svc.formatPercent(-0.1)).toBe('-10.00%');
    });

    it('when decimals is overridden, the precision changes', () => {
      expect(svc.formatPercent(0.1, 0)).toBe('10%');
    });
  });

  describe('formatInteger', () => {
    it('when value is null, the placeholder is returned', () => {
      expect(svc.formatInteger(null)).toBe('--');
    });

    it('when value is 1234, a grouped integer is returned', () => {
      expect(svc.formatInteger(1234)).toBe('1,234');
    });
  });

  describe('formatDate', () => {
    it('when date is null, the placeholder is returned', () => {
      expect(svc.formatDate(null)).toBe('--');
    });

    it('when date is an invalid string, the placeholder is returned', () => {
      expect(svc.formatDate('not-a-date')).toBe('--');
    });

    it('when format is iso, an ISO day string is returned', () => {
      expect(svc.formatDate('2026-01-15T12:00:00.000Z', 'iso')).toBe('2026-01-15');
    });

    it('when format is medium, a long date is returned for a Date input', () => {
      const out = svc.formatDate(new Date('2026-01-15T12:00:00.000Z'), 'medium');
      expect(out).toContain('Jan');
      expect(out).toContain('2026');
    });

    it('when format is short, a short date is returned', () => {
      expect(svc.formatDate('2026-01-15T12:00:00.000Z', 'short')).toContain('Jan');
    });
  });

  describe('formatDuration', () => {
    it('when ms is null, the placeholder is returned', () => {
      expect(svc.formatDuration(null)).toBe('--');
    });

    it('when ms is NaN, the placeholder is returned', () => {
      expect(svc.formatDuration(NaN)).toBe('--');
    });

    it('when under a minute, seconds are returned', () => {
      expect(svc.formatDuration(1000)).toBe('1s');
    });

    it('when over a minute, minutes and seconds are returned', () => {
      expect(svc.formatDuration(65_000)).toBe('1m 5s');
    });

    it('when over an hour, hours and minutes are returned', () => {
      expect(svc.formatDuration(3_660_000)).toBe('1h 1m');
    });
  });
});
