import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
  makeBrinsonResponse,
} from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { SaaTaaPanelComponent } from './saa-taa-panel';

// ─── Suite 1: signal-level computed values (original tests) ──────────────────

describe('SaaTaaPanelComponent', () => {
  let fixture: ComponentFixture<SaaTaaPanelComponent>;
  let comp: SaaTaaPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [SaaTaaPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(SaaTaaPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when brinson is null, multiLevel is empty and hasData is false', () => {
    expect(comp.hasData()).toBe(false);
    expect(comp.multiLevel()).toEqual([]);
  });

  it('when populated, multiLevel derives one sector-level row from the brinson sectors', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    expect(comp.hasData()).toBe(true);
    expect(comp.multiLevel().length).toBe(1);
    expect(comp.multiLevel()[0].level).toBe('Sector');
    expect(comp.multiLevel()[0].name).toBe('Technology');
    expect(comp.waterfallValues()).toEqual([30]); // totalEffect 0.003 × 10000
  });
});

// ─── Suite 2: DOM rendering — criterion (T3 / issue #1020) ───────────────────
// Criteria:
//   - Empty-state text renders when brinson is null (non-blank, no table).
//   - Populated state renders waterfall + data table.

describe('SaaTaaPanelComponent — DOM rendering (issue #1020)', () => {
  let fixture: ComponentFixture<SaaTaaPanelComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [SaaTaaPanelComponent],
      withHttp: false,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(SaaTaaPanelComponent);
  });

  it('when brinson is null, the panel renders non-blank empty-state text', () => {
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent?.trim() ?? '';

    expect(text.length).toBeGreaterThan(
      0,
      'Expected non-blank empty-state text when brinson is null',
    );
  });

  it('when brinson is null, no data table is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    expect(el.querySelector('table')).toBeNull(
      'Expected no data table in the empty state',
    );
  });

  it('when brinson has data, the waterfall component element is rendered', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    // EchartsWaterfallComponent initialises its canvas asynchronously (afterNextRender
    // + dynamic import), so we verify the custom element itself is in the DOM rather
    // than waiting for the canvas.
    expect(el.querySelector('app-echarts-waterfall')).toBeTruthy(
      'Expected app-echarts-waterfall element when brinson is populated',
    );
  });

  it('when brinson has data, a data table is rendered', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    expect(el.querySelector('table')).toBeTruthy(
      'Expected a data table to render when brinson is populated',
    );
  });

  it('when brinson has data, every sector name appears in the rendered table', () => {
    const brinson = makeBrinsonResponse();
    fixture.componentRef.setInput('brinson', brinson);
    fixture.detectChanges();

    const tableText = (fixture.nativeElement as HTMLElement).querySelector('table')?.textContent ?? '';

    for (const row of brinson.sectors) {
      expect(tableText).toContain(
        row.sector,
        `Expected sector "${row.sector}" to appear in the SAA/TAA data table`,
      );
    }
  });
});
