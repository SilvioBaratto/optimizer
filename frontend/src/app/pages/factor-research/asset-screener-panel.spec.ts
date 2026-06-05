import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeFactorICReport } from '../../../testing';
import { AssetScreenerPanelComponent } from './asset-screener-panel';

describe('AssetScreenerPanelComponent', () => {
  let fixture: ComponentFixture<AssetScreenerPanelComponent>;
  let comp: AssetScreenerPanelComponent;

  const REPORTS = [
    makeFactorICReport({ factor: 'book_to_price', group: 'value' }),
    makeFactorICReport({ factor: 'momentum_12_1', group: 'momentum' }),
  ];

  beforeEach(async () => {
    await configureTestBed({ imports: [AssetScreenerPanelComponent], withHttp: false });
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
});
