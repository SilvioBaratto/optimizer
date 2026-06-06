import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub, makeTAASignal } from '../../../testing';
import { TaaPanelComponent } from './taa-panel';

describe('TaaPanelComponent', () => {
  let fixture: ComponentFixture<TaaPanelComponent>;
  let comp: TaaPanelComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [TaaPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(TaaPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when there are no signals, the signal rows are empty', () => {
    expect(comp.signalRows()).toEqual([]);
  });

  it('when populated, signal rows derive delta and pass the regime through', () => {
    fixture.componentRef.setInput('signals', [makeTAASignal()]);
    const rows = comp.signalRows();
    expect(rows.length).toBe(1);
    expect(rows[0]['deltaBps']).toBeCloseTo(0.05); // 0.25 - 0.20
    expect(rows[0]['regime']).toBe('expansion');
    expect(rows[0]['factor']).toBe('Momentum'); // title-cased
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => comp.ngOnDestroy()).not.toThrow();
  });
});
