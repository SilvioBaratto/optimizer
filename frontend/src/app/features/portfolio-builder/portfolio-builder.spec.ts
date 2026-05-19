import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { BUILDER_STAGES, type BuilderStage } from './builder-stage';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';

function configure(): void {
  const tick = new Subject<number>();
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      { provide: JOB_POLL_TICK, useValue: tick.asObservable() },
    ],
  });
}

function setup(): HTMLElement {
  configure();
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('PortfolioBuilderComponent', () => {
  it('when instantiated, component creates without error', () => {
    configure();
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
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

  it('when rendered, app-stage-strip is mounted inside the stage-strip region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="stage-strip"]');
    expect(region?.querySelector('app-stage-strip')).not.toBeNull();
  });

  it('when rendered, app-inputs-pane is mounted inside the left region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="left"]');
    expect(region?.querySelector('app-inputs-pane')).not.toBeNull();
  });

  it('when rendered, app-canvas-pane is mounted inside the center region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="center"]');
    expect(region?.querySelector('app-canvas-pane')).not.toBeNull();
  });

  it('when rendered, app-analytics-pane is mounted inside the right region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="right"]');
    expect(region?.querySelector('app-analytics-pane')).not.toBeNull();
  });

  it('when rendered, app-action-bar is mounted inside the action-bar region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="action-bar"]');
    expect(region?.querySelector('app-action-bar')).not.toBeNull();
  });

  it('when stage-strip emits stageSelect, store.setStage is invoked and currentStage updates', () => {
    configure();
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    fixture.detectChanges();

    const host = fixture.nativeElement as HTMLElement;
    const buttons = Array.from(
      host.querySelectorAll<HTMLButtonElement>('button[role="tab"]'),
    );
    expect(buttons.length).toBe(6);

    const target: BuilderStage = 'review';
    buttons[BUILDER_STAGES.indexOf(target)].click();
    fixture.detectChanges();

    expect(fixture.componentInstance.store.currentStage()).toBe(target);
  });
});
