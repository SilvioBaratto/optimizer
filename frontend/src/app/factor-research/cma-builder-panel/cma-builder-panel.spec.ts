import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeCMASet } from '../../../testing';
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
});
