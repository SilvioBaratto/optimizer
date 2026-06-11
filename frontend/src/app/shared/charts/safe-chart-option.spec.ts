// Unit tests for safeChartOption and EMPTY_STATE_OPTION.
// TDD Red phase: these tests should fail before safe-chart-option.ts exists.

import { safeChartOption, EMPTY_STATE_OPTION } from './safe-chart-option';

describe('safeChartOption', () => {
  // ── Populated array → builder is called and result returned ───────────────

  it('calls builder and returns its result when data is a populated array', () => {
    const data = [1, 2, 3];
    const expected = { series: [{ type: 'bar' }] };
    const builder = jasmine.createSpy('builder').and.returnValue(expected);

    const result = safeChartOption(data, builder);

    expect(builder).toHaveBeenCalledWith(data);
    expect(result).toBe(expected);
  });

  // ── null → EMPTY_STATE_OPTION (reference equality) ────────────────────────

  it('returns EMPTY_STATE_OPTION (by reference) when data is null', () => {
    const builder = jasmine.createSpy('builder');

    const result = safeChartOption(null, builder);

    expect(result).toBe(EMPTY_STATE_OPTION);
    expect(builder).not.toHaveBeenCalled();
  });

  // ── undefined → EMPTY_STATE_OPTION (reference equality) ──────────────────

  it('returns EMPTY_STATE_OPTION (by reference) when data is undefined', () => {
    const builder = jasmine.createSpy('builder');

    const result = safeChartOption(undefined, builder);

    expect(result).toBe(EMPTY_STATE_OPTION);
    expect(builder).not.toHaveBeenCalled();
  });

  // ── empty array → EMPTY_STATE_OPTION (reference equality) ────────────────

  it('returns EMPTY_STATE_OPTION (by reference) when data is an empty array', () => {
    const builder = jasmine.createSpy('builder');

    const result = safeChartOption([], builder);

    expect(result).toBe(EMPTY_STATE_OPTION);
    expect(builder).not.toHaveBeenCalled();
  });

  // ── non-array object → builder is called (not empty-state) ───────────────

  it('calls builder when data is a non-array record/object', () => {
    const data = { value: 42 };
    const expected = { series: [] };
    const builder = jasmine.createSpy('builder').and.returnValue(expected);

    const result = safeChartOption(data, builder);

    expect(builder).toHaveBeenCalledWith(data);
    expect(result).toBe(expected);
  });
});
