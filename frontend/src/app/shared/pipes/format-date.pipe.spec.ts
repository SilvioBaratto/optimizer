import { TestBed } from '@angular/core/testing';
import { FormatDatePipe } from './format-date.pipe';

const ISO = '2026-01-15T12:00:00.000Z'; // noon UTC avoids tz day-rollover

describe('FormatDatePipe', () => {
  let pipe: FormatDatePipe;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    pipe = TestBed.runInInjectionContext(() => new FormatDatePipe());
  });

  it('when value is null, the dash placeholder is returned', () => {
    expect(pipe.transform(null)).toBe('--');
  });

  it('when value is undefined, the dash placeholder is returned', () => {
    expect(pipe.transform(undefined)).toBe('--');
  });

  it('when value is an invalid date string, the dash placeholder is returned', () => {
    expect(pipe.transform('not-a-date')).toBe('--');
  });

  it('when format is iso, an ISO day string is returned', () => {
    expect(pipe.transform(ISO, 'iso')).toBe('2026-01-15');
  });

  it('when format is short, a short month-day string is returned', () => {
    expect(pipe.transform(ISO, 'short')).toContain('Jan');
  });

  it('when format is medium, a long date is returned', () => {
    const out = pipe.transform(ISO, 'medium');
    expect(out).toContain('Jan');
    expect(out).toContain('2026');
  });

  it('when no format is given, it defaults to medium', () => {
    const out = pipe.transform(ISO);
    expect(out).toContain('Jan');
    expect(out).toContain('2026');
  });

  it('when value is a Date object, it formats the same as the ISO string', () => {
    expect(pipe.transform(new Date(ISO), 'iso')).toBe('2026-01-15');
  });
});
