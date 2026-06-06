import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ConfigPanelComponent } from './config-panel';
import type { PortfolioOptimizeRequest } from '../models/portfolio.model';

describe('ConfigPanelComponent', () => {
  let fixture: ComponentFixture<ConfigPanelComponent>;
  let comp: ConfigPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [ConfigPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(ConfigPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when strategy is max_utility, the risk-aversion control is shown', () => {
    fixture.componentRef.setInput('strategy', 'max_utility');
    expect(comp.showRiskAversion()).toBe(true);
    expect(comp.showCvarBeta()).toBe(false);
  });

  it('when strategy is min_cvar, the cvar-beta control is shown', () => {
    fixture.componentRef.setInput('strategy', 'min_cvar');
    expect(comp.showCvarBeta()).toBe(true);
    expect(comp.showRiskAversion()).toBe(false);
  });

  it('when strategy is null, onSubmit emits nothing (guard)', () => {
    let emitted = false;
    comp.runOptimization.subscribe(() => (emitted = true));
    comp.onSubmit();
    expect(emitted).toBe(false);
  });

  it('when a strategy is set, onSubmit emits the built request', () => {
    fixture.componentRef.setInput('strategy', 'max_utility');
    let req: PortfolioOptimizeRequest | undefined;
    comp.runOptimization.subscribe((r) => (req = r));
    comp.onSubmit();
    expect(req?.strategy).toBe('max_utility');
    expect(req?.risk_aversion).toBe(1.0); // included because showRiskAversion()
  });
});
