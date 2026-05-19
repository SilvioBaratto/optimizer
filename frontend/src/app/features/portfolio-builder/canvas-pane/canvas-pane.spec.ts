import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { CanvasPaneComponent } from './canvas-pane';
import type { WeightItem } from '../../../models/pipeline-builder.model';

function setup(): {
  fixture: ComponentFixture<CanvasPaneComponent>;
  host: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  const fixture = TestBed.createComponent(CanvasPaneComponent);
  fixture.detectChanges();
  return { fixture, host: fixture.nativeElement as HTMLElement };
}

describe('CanvasPaneComponent', () => {
  it('when mounted with default inputs, placeholder text "Allocation — Phase 2" is rendered', () => {
    const { host } = setup();
    expect(host.textContent).toContain('Allocation — Phase 2');
  });

  it('when mounted, no chart element (<canvas> or <echarts>) is in the DOM', () => {
    const { host } = setup();
    expect(host.querySelector('canvas')).toBeNull();
    expect(host.querySelector('echarts')).toBeNull();
    expect(host.querySelector('[data-region="echarts"]')).toBeNull();
  });

  it('when weights input is set, the component accepts it without throwing and still renders the placeholder', () => {
    const { fixture, host } = setup();
    const weights: WeightItem[] = [
      { ticker: 'AAPL', weight: 0.4 },
      { ticker: 'MSFT', weight: 0.6 },
    ];
    expect(() =>
      fixture.componentRef.setInput('weights', weights),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.weights()).toEqual(weights);
    expect(host.textContent).toContain('Allocation — Phase 2');
    expect(host.querySelector('canvas')).toBeNull();
  });

  it('when frontier input is set with a list of points, the component accepts it without throwing', () => {
    const { fixture } = setup();
    const frontier: Record<string, number>[] = [
      { risk: 0.1, return: 0.05 },
      { risk: 0.2, return: 0.09 },
    ];
    expect(() =>
      fixture.componentRef.setInput('frontier', frontier),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.frontier()).toEqual(frontier);
  });

  it('when sessionId input is set to a string, the component accepts it without throwing', () => {
    const { fixture } = setup();
    expect(() =>
      fixture.componentRef.setInput('sessionId', 'sid-42'),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.sessionId()).toBe('sid-42');
  });

  it('when sessionId input is set to null, the component accepts it without throwing', () => {
    const { fixture } = setup();
    expect(() =>
      fixture.componentRef.setInput('sessionId', null),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.sessionId()).toBeNull();
  });

  it('when freshly constructed, all three inputs default to undefined', () => {
    const { fixture } = setup();
    expect(fixture.componentInstance.weights()).toBeUndefined();
    expect(fixture.componentInstance.frontier()).toBeUndefined();
    expect(fixture.componentInstance.sessionId()).toBeUndefined();
  });

  it('when mounted, the placeholder element carries an identifying data-region for shell layout assertions', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="canvas-pane"]')).not.toBeNull();
  });
});
