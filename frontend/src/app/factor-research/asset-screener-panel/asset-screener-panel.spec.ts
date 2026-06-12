import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub, makeFactorICReport } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { AssetScreenerPanelComponent } from './asset-screener-panel';

describe('AssetScreenerPanelComponent', () => {
  let fixture: ComponentFixture<AssetScreenerPanelComponent>;
  let comp: AssetScreenerPanelComponent;

  const REPORTS = [
    makeFactorICReport({ factor: 'book_to_price', group: 'value' }),
    makeFactorICReport({ factor: 'momentum_12_1', group: 'momentum' }),
  ];

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [AssetScreenerPanelComponent],
      withHttp: false,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(AssetScreenerPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when no filter is active, all reports pass through', () => {
    fixture.componentRef.setInput('icReports', REPORTS);
    expect(comp.filteredReports().length).toBe(2);
  });

  it('when a group is toggled on, only that group remains; toggling off restores all', () => {
    fixture.componentRef.setInput('icReports', REPORTS);
    comp.toggleGroup('value');
    expect(comp.activeFilters()).toEqual(['value']);
    expect(comp.filteredReports().length).toBe(1);
    expect(comp.filteredReports()[0].group).toBe('value');
    comp.toggleGroup('value');
    expect(comp.activeFilters()).toEqual([]);
    expect(comp.filteredReports().length).toBe(2);
  });

  it('when filtered, scatter points derive from the filtered reports (IC vs ICIR)', () => {
    fixture.componentRef.setInput('icReports', REPORTS);
    comp.toggleGroup('value');
    const pts = comp.scatterPoints();
    expect(pts.length).toBe(1);
    expect(pts[0].x).toBe(0.05); // ic
    expect(pts[0].y).toBe(0.8); // icir
  });

  it('when errorMessage is set, a role="alert" element renders with the message', () => {
    fixture.componentRef.setInput('errorMessage', 'TE fetch failed');
    fixture.detectChanges();
    const alert = (fixture.nativeElement as HTMLElement).querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert?.textContent).toContain('TE fetch failed');
  });

  it('when errorMessage is null, no role="alert" element renders', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('[role="alert"]')).toBeNull();
  });

  it('when TE observations are present, the TE table renders and null values show as "—"', () => {
    fixture.componentRef.setInput('teObservations', [
      { id: '1', country: 'USA', indicator_key: 'pmi', date: '2024-01-01', value: 52.3, created_at: '', updated_at: '' },
      { id: '2', country: 'USA', indicator_key: 'cpi', date: '2024-02-01', value: null, created_at: '', updated_at: '' },
    ]);
    fixture.detectChanges();
    const rows = comp.teRows();
    expect(rows[0]['value']).toBe('52.300');
    expect(rows[1]['value']).toBe('—');
  });

  it('when there are no TE observations, the empty hint is shown', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain('No observations loaded yet');
  });

  it('when the Fetch TE button is clicked, fetchTe is emitted', () => {
    fixture.detectChanges();
    let fired = false;
    comp.fetchTe.subscribe(() => (fired = true));
    const btn = Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll('button'),
    ).find((b) => b.textContent?.includes('Fetch TE'));
    btn?.click();
    expect(fired).toBe(true);
  });

  it('when a group is active, isGroupActive reflects it and a Clear all button appears', () => {
    fixture.componentRef.setInput('icReports', REPORTS);
    comp.toggleGroup('value');
    fixture.detectChanges();
    expect(comp.isGroupActive('value')).toBe(true);
    expect(comp.isGroupActive('momentum')).toBe(false);
    expect((fixture.nativeElement as HTMLElement).textContent).toContain('Clear all');
  });
});
