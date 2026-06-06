import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { VarPanelComponent } from './var-panel';
import type { VaRResult } from '../risk.model';

function varResult(confidence: number): VaRResult {
  return {
    method: 'historical',
    confidence,
    horizon: 1,
    var: -0.03,
    cvar: -0.05,
    portfolioValue: 1_000_000,
    varDollar: -30_000,
    cvarDollar: -50_000,
  };
}

function clickButton(fixture: ComponentFixture<unknown>, text: string): void {
  const btn = Array.from(fixture.nativeElement.querySelectorAll('button')).find(
    (b) => (b as HTMLElement).textContent?.trim().includes(text),
  ) as HTMLButtonElement | undefined;
  btn?.click();
}

describe('VarPanelComponent', () => {
  let fixture: ComponentFixture<VarPanelComponent>;
  let comp: VarPanelComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [VarPanelComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(VarPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when results are empty, the VaR figures show the dash placeholder', () => {
    expect(comp.activeResult()).toBeUndefined();
    expect(comp.varDollarStr()).toBe('--');
    expect(comp.cvarDollarStr()).toBe('--');
  });

  it('when populated, the active result and comparison rows derive from the input', () => {
    fixture.componentRef.setInput('varResults', [varResult(0.95)]);
    expect(comp.activeResult()?.confidence).toBe(0.95);
    expect(comp.comparisonRows().length).toBe(1);
    expect(comp.varDollarStr()).not.toBe('--');
  });

  it('when a method button is clicked, methodChange emits that method', () => {
    fixture.detectChanges();
    let method: string | undefined;
    comp.methodChange.subscribe((m) => (method = m));
    clickButton(fixture, 'Parametric');
    expect(method).toBe('parametric');
  });

  it('when a confidence button is clicked, confidenceChange emits that level', () => {
    fixture.detectChanges();
    let conf: number | undefined;
    comp.confidenceChange.subscribe((c) => (conf = c));
    clickButton(fixture, '99');
    expect(conf).toBe(0.99);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => comp.ngOnDestroy()).not.toThrow();
  });
});
