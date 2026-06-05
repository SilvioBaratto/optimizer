import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { ProgressBarComponent } from './progress-bar';

describe('ProgressBarComponent', () => {
  let fixture: ComponentFixture<ProgressBarComponent>;
  let comp: ProgressBarComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [ProgressBarComponent], withHttp: false });
    fixture = TestBed.createComponent(ProgressBarComponent);
    comp = fixture.componentInstance;
  });

  it('when total is zero, pct guards against divide-by-zero and returns 0', () => {
    fixture.componentRef.setInput('current', 5);
    fixture.componentRef.setInput('total', 0);
    fixture.detectChanges();
    expect(comp.pct()).toBe(0);
  });

  it('when current is half of total, pct is 50', () => {
    fixture.componentRef.setInput('current', 50);
    fixture.componentRef.setInput('total', 100);
    fixture.detectChanges();
    expect(comp.pct()).toBe(50);
    const bar = fixture.nativeElement.querySelector('.bg-accent') as HTMLElement;
    expect(bar.style.width).toBe('50%');
  });

  it('when a label is set, it renders', () => {
    setInput(fixture, 'label', 'Loading prices');
    expect(fixture.nativeElement.textContent).toContain('Loading prices');
  });
});
