import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { HelpTooltipComponent } from './help-tooltip';

// getBoundingClientRect returns zeros in the test DOM; stub it so position maths
// is exercised deterministically.
function stubRect(fixture: ComponentFixture<unknown>): void {
  spyOn(fixture.nativeElement, 'getBoundingClientRect').and.returnValue({
    top: 80, left: 50, bottom: 100, right: 70, width: 20, height: 20, x: 50, y: 80,
    toJSON: () => ({}),
  } as DOMRect);
}

describe('HelpTooltipComponent', () => {
  let fixture: ComponentFixture<HelpTooltipComponent>;
  let comp: HelpTooltipComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [HelpTooltipComponent], withHttp: false });
    fixture = TestBed.createComponent(HelpTooltipComponent);
    comp = fixture.componentInstance;
  });

  it('when first rendered, the tooltip is hidden', () => {
    setInput(fixture, 'text', 'Explains the metric');
    expect(comp.isVisible()).toBe(false);
    expect(fixture.nativeElement.querySelector('[role="tooltip"]')).toBeNull();
  });

  it('when shown, the tooltip becomes visible with its text and computed position', () => {
    stubRect(fixture);
    setInput(fixture, 'text', 'Explains the metric');
    comp.show();
    fixture.detectChanges();
    expect(comp.isVisible()).toBe(true);
    expect(comp.tooltipPosition().top).toBe(108); // bottom 100 + gap 8
    expect(fixture.nativeElement.querySelector('[role="tooltip"]').textContent).toContain('Explains the metric');
  });

  it('when hidden after showing, the tooltip is removed', () => {
    stubRect(fixture);
    setInput(fixture, 'text', 'x');
    comp.show();
    fixture.detectChanges();
    comp.hide();
    fixture.detectChanges();
    expect(comp.isVisible()).toBe(false);
    expect(fixture.nativeElement.querySelector('[role="tooltip"]')).toBeNull();
  });

  it('when a link is provided, a learn-more anchor is rendered', () => {
    stubRect(fixture);
    setInput(fixture, 'text', 'x');
    setInput(fixture, 'link', 'https://docs.example.com');
    comp.show();
    fixture.detectChanges();
    const a = fixture.nativeElement.querySelector('a') as HTMLAnchorElement;
    expect(a.getAttribute('href')).toBe('https://docs.example.com');
  });
});
