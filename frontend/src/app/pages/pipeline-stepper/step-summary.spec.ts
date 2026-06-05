import { stepSummary } from './step-summary';
import type { PipelineStepId } from '../../models/pipeline-builder.model';

describe('stepSummary', () => {
  it('when the result is null, an empty summary is returned', () => {
    expect(stepSummary('load', null)).toBe('');
  });

  it('when a load result is given, the ticker/day/currency line is built', () => {
    expect(stepSummary('load', { n_tickers: 8, n_trading_days: 252, base_currency: 'EUR' }))
      .toBe('8 tickers · 252 days · EUR');
  });

  it('when an optimize result has a net sharpe, it is formatted to two decimals', () => {
    expect(stepSummary('optimize', { n_selected: 20, net_sharpe: 1.234, hockey_stick_warning: false }))
      .toBe('20 holdings · net Sharpe 1.23');
  });

  it('when an optimize result has no net sharpe, a dash is shown', () => {
    expect(stepSummary('optimize', { n_selected: 20, net_sharpe: null }))
      .toBe('20 holdings · net Sharpe —');
  });

  it('when a persist result is not persisted, the reason is appended', () => {
    expect(stepSummary('persist', { persisted: false, reason: 'dry run' }))
      .toBe('not persisted — dry run');
  });

  it('when the step id has no formatter, the call throws and degrades to empty', () => {
    expect(stepSummary('bogus_step' as PipelineStepId, { any: 1 })).toBe('');
  });
});
