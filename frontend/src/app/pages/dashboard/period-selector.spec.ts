import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { PeriodSelectorComponent, type DashboardPeriod } from './period-selector';

describe('PeriodSelectorComponent', () => {
  let fixture: ComponentFixture<PeriodSelectorComponent>;
  let comp: PeriodSelectorComponent;

  function button(period: DashboardPeriod): HTMLButtonElement {
    return fixture.nativeElement.querySelector(`[data-testid="period-${period}"]`) as HTMLButtonElement;
  }

  beforeEach(async () => {
    await configureTestBed({ imports: [PeriodSelectorComponent], withHttp: false });
    fixture = TestBed.createComponent(PeriodSelectorComponent);
    comp = fixture.componentInstance;
  });

  it('when rendered, all four period options are shown', () => {
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('button').length).toBe(4);
  });

  it('when a value is set, its button is aria-selected', () => {
    fixture.componentRef.setInput('value', '5Y');
    fixture.detectChanges();
    expect(button('5Y').getAttribute('aria-selected')).toBe('true');
    expect(button('1Y').getAttribute('aria-selected')).toBe('false');
  });

  it('when the current value is clicked, no change is emitted', () => {
    fixture.componentRef.setInput('value', '3Y');
    fixture.detectChanges();
    let emitted = false;
    comp.valueChange.subscribe(() => (emitted = true));
    button('3Y').click();
    expect(emitted).toBe(false);
  });

  it('when a different value is clicked, valueChange emits it', () => {
    fixture.componentRef.setInput('value', '3Y');
    fixture.detectChanges();
    let value: DashboardPeriod | undefined;
    comp.valueChange.subscribe((v) => (value = v));
    button('5Y').click();
    expect(value).toBe('5Y');
  });
});
