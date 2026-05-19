import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PortfolioBuilderComponent } from './portfolio-builder';

function setup(): HTMLElement {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('PortfolioBuilderComponent', () => {
  it('when instantiated, component creates without error', () => {
    const fixture = TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    }).createComponent(PortfolioBuilderComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when rendered, each of the 5 [data-region] selectors resolves to a non-null element', () => {
    const host = setup();
    for (const region of [
      'stage-strip',
      'left',
      'center',
      'right',
      'action-bar',
    ]) {
      expect(host.querySelector(`[data-region="${region}"]`)).not.toBeNull();
    }
  });

  it('when rendered, multi-column grid container exposes responsive grid utility classes', () => {
    const host = setup();
    const leftPane = host.querySelector('[data-region="left"]');
    const gridEl = leftPane?.parentElement;
    expect(gridEl).not.toBeNull();
    const cls = gridEl?.getAttribute('class') ?? '';
    expect(cls).toContain('lg:grid-cols-[320px_1fr_300px]');
    expect(cls).toContain('max-lg:grid-cols-1');
  });
});
