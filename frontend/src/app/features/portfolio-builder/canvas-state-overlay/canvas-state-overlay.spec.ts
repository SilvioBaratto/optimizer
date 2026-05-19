import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { CanvasStateOverlayComponent } from './canvas-state-overlay';
import type { BuilderResultStatus } from '../builder-result.model';

@Component({
  selector: 'test-host',
  imports: [CanvasStateOverlayComponent],
  template: `
    <app-canvas-state-overlay
      [status]="status()"
      [retryFn]="retryFn()"
    >
      <div data-test="projected">CHILD</div>
    </app-canvas-state-overlay>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class HostComponent {
  readonly status = signal<BuilderResultStatus>('ok');
  readonly retryFn = signal<(() => void) | undefined>(undefined);
}

function setupHost(): {
  fixture: ComponentFixture<HostComponent>;
  host: HostComponent;
  el: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  const fixture = TestBed.createComponent(HostComponent);
  fixture.detectChanges();
  return {
    fixture,
    host: fixture.componentInstance,
    el: fixture.nativeElement as HTMLElement,
  };
}

function setStatus(
  fixture: ComponentFixture<HostComponent>,
  status: BuilderResultStatus,
): void {
  fixture.componentInstance.status.set(status);
  fixture.detectChanges();
}

function setRetryFn(
  fixture: ComponentFixture<HostComponent>,
  fn: (() => void) | undefined,
): void {
  fixture.componentInstance.retryFn.set(fn);
  fixture.detectChanges();
}

describe('CanvasStateOverlayComponent', () => {
  it('when status is "running", [data-region="overlay-loading"] is rendered with a spinner and skeleton rows, and projected content is hidden', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'running');

    const overlay = el.querySelector('[data-region="overlay-loading"]');
    expect(overlay).not.toBeNull();
    expect(overlay!.querySelector('[data-element="spinner"]')).not.toBeNull();
    expect(overlay!.querySelectorAll('app-loading-skeleton').length).toBeGreaterThan(0);
    expect(el.querySelector('[data-test="projected"]')).toBeNull();
  });

  it('when status is "error", [data-region="overlay-error"] and [data-action="retry"] are rendered, and projected content is hidden', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'error');

    expect(el.querySelector('[data-region="overlay-error"]')).not.toBeNull();
    expect(el.querySelector('[data-action="retry"]')).not.toBeNull();
    expect(el.querySelector('[data-test="projected"]')).toBeNull();
  });

  it('when status is "error" and the retry button is clicked, the retryFn input is invoked exactly once', () => {
    const { fixture, el } = setupHost();
    const spy = jasmine.createSpy('retryFn');
    setRetryFn(fixture, spy);
    setStatus(fixture, 'error');

    const btn = el.querySelector<HTMLButtonElement>('[data-action="retry"]');
    expect(btn).not.toBeNull();
    btn!.click();

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('when status is "error" and the retry button is clicked without a retryFn input, no error is thrown and nothing is invoked', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'error');

    const btn = el.querySelector<HTMLButtonElement>('[data-action="retry"]');
    expect(btn).not.toBeNull();
    expect(() => btn!.click()).not.toThrow();
  });

  it('when status is "idle", [data-region="overlay-idle"] is rendered with a CTA message, and projected content is hidden', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'idle');

    const overlay = el.querySelector('[data-region="overlay-idle"]');
    expect(overlay).not.toBeNull();
    expect(overlay!.textContent?.trim().length).toBeGreaterThan(0);
    expect(el.querySelector('[data-test="projected"]')).toBeNull();
  });

  it('when status is "stale", projected content is rendered with opacity-60 + pointer-events-none and [data-region="overlay-stale-badge"] is present', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'stale');

    const projected = el.querySelector<HTMLElement>('[data-test="projected"]');
    expect(projected).not.toBeNull();

    const dimWrapper = projected!.closest<HTMLElement>('[data-element="stale-content"]');
    expect(dimWrapper).not.toBeNull();
    expect(dimWrapper!.classList.contains('opacity-60')).toBe(true);
    expect(dimWrapper!.classList.contains('pointer-events-none')).toBe(true);

    expect(el.querySelector('[data-region="overlay-stale-badge"]')).not.toBeNull();
  });

  it('when status is "ok", only the projected content is rendered and no overlay region exists', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'ok');

    expect(el.querySelector('[data-test="projected"]')).not.toBeNull();
    expect(el.querySelector('[data-region="overlay-loading"]')).toBeNull();
    expect(el.querySelector('[data-region="overlay-error"]')).toBeNull();
    expect(el.querySelector('[data-region="overlay-idle"]')).toBeNull();
    expect(el.querySelector('[data-region="overlay-stale-badge"]')).toBeNull();
  });

  it('when status transitions from "running" to "ok", overlay is removed and projected content appears', () => {
    const { fixture, el } = setupHost();
    setStatus(fixture, 'running');
    expect(el.querySelector('[data-region="overlay-loading"]')).not.toBeNull();
    expect(el.querySelector('[data-test="projected"]')).toBeNull();

    setStatus(fixture, 'ok');
    expect(el.querySelector('[data-region="overlay-loading"]')).toBeNull();
    expect(el.querySelector('[data-test="projected"]')).not.toBeNull();
  });
});
