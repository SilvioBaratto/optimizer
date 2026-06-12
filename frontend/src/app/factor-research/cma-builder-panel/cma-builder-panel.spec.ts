import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub, makeCMASet } from '../../../testing';
import { CmaBuilderPanelComponent } from './cma-builder-panel';

describe('CmaBuilderPanelComponent', () => {
  let fixture: ComponentFixture<CmaBuilderPanelComponent>;
  let comp: CmaBuilderPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [CmaBuilderPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(CmaBuilderPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when there are no CMA sets, the active set is undefined and rows are empty', () => {
    expect(comp.activeSet()).toBeUndefined();
    expect(comp.returnsRows()).toEqual([]);
    expect(comp.scatterPoints()).toEqual([]);
  });

  it('when a set is present, returns rows and scatter points derive from its assets', () => {
    fixture.componentRef.setInput('cmaSets', [makeCMASet()]);
    expect(comp.returnsRows().length).toBe(1);
    expect(comp.scatterPoints()[0]).toEqual({ x: 0.2, y: 0.07, label: 'AAPL' });
  });

  it('when the selected set changes, the active set follows', () => {
    fixture.componentRef.setInput('cmaSets', [
      makeCMASet({ label: 'A' }),
      makeCMASet({ label: 'B' }),
    ]);
    comp.selectedSet.set(1);
    expect(comp.activeSet().label).toBe('B');
  });

  it('when a CMA set is present, the returns table and CMA scatter chart render', () => {
    installResizeObserverStub();
    fixture.componentRef.setInput('cmaSets', [makeCMASet()]);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('app-data-table')).not.toBeNull();
    expect(el.querySelector('app-echarts-scatter')).not.toBeNull();
  });

  it('when errorMessage is set, a role="alert" element renders with the message', () => {
    fixture.componentRef.setInput('errorMessage', 'CMA fetch failed');
    fixture.detectChanges();
    const alert = (fixture.nativeElement as HTMLElement).querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert?.textContent).toContain('CMA fetch failed');
  });
});
