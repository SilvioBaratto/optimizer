import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub, makeCorrelationData } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { CorrelationPanelComponent } from './correlation-panel';

function clickButton(fixture: ComponentFixture<unknown>, text: string): void {
  const btn = Array.from(fixture.nativeElement.querySelectorAll('button')).find(
    (b) => (b as HTMLElement).textContent?.trim().includes(text),
  ) as HTMLButtonElement | undefined;
  btn?.click();
}

describe('CorrelationPanelComponent', () => {
  let fixture: ComponentFixture<CorrelationPanelComponent>;
  let comp: CorrelationPanelComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [CorrelationPanelComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(CorrelationPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when empty data is given, it mounts without throwing', () => {
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('when populated, it mounts the heatmap without throwing', () => {
    fixture.componentRef.setInput('correlationData', makeCorrelationData());
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('when a method button is clicked, methodChange emits that method', () => {
    fixture.componentRef.setInput('correlationData', makeCorrelationData());
    fixture.detectChanges();
    let method: string | undefined;
    comp.methodChange.subscribe((m) => (method = m));
    clickButton(fixture, 'Spearman');
    expect(method).toBe('spearman');
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => comp.ngOnDestroy()).not.toThrow();
  });
});
