import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { AnalyticsPaneComponent } from './analytics-pane';
import type {
  OptimizeStepResult,
  ReportStepResult,
} from '../../../models/pipeline-builder.model';

function setup(): {
  fixture: ComponentFixture<AnalyticsPaneComponent>;
  host: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  const fixture = TestBed.createComponent(AnalyticsPaneComponent);
  fixture.detectChanges();
  return { fixture, host: fixture.nativeElement as HTMLElement };
}

describe('AnalyticsPaneComponent', () => {
  it('when mounted with default inputs, placeholder text "Risk & Exposure — Phase 2" is rendered', () => {
    const { host } = setup();
    expect(host.textContent).toContain('Risk & Exposure — Phase 2');
  });

  it('when mounted, no chart element (<canvas> or <echarts>) is in the DOM', () => {
    const { host } = setup();
    expect(host.querySelector('canvas')).toBeNull();
    expect(host.querySelector('echarts')).toBeNull();
    expect(host.querySelector('[data-region="echarts"]')).toBeNull();
  });

  it('when riskMetrics input is set, the component accepts it without throwing and still renders the placeholder', () => {
    const { fixture, host } = setup();
    const metrics: ReportStepResult['metrics'] = {
      risk: { volatility: 0.18, max_drawdown: -0.32 },
      returns: { net_sharpe: 0.85 },
    };
    expect(() =>
      fixture.componentRef.setInput('riskMetrics', metrics),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.riskMetrics()).toEqual(metrics);
    expect(host.textContent).toContain('Risk & Exposure — Phase 2');
    expect(host.querySelector('canvas')).toBeNull();
  });

  it('when exposures input is set, the component accepts it without throwing', () => {
    const { fixture } = setup();
    const sectorBreakdown: OptimizeStepResult['sector_breakdown'] = {
      Technology: 0.4,
      Healthcare: 0.3,
      Financials: 0.3,
    };
    expect(() =>
      fixture.componentRef.setInput('exposures', sectorBreakdown),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.exposures()).toEqual(sectorBreakdown);
  });

  it('when freshly constructed, both inputs default to undefined', () => {
    const { fixture } = setup();
    expect(fixture.componentInstance.riskMetrics()).toBeUndefined();
    expect(fixture.componentInstance.exposures()).toBeUndefined();
  });

  it('when mounted, the placeholder element carries an identifying data-region for shell layout assertions', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="analytics-pane"]')).not.toBeNull();
  });
});
