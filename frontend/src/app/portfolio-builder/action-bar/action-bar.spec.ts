import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { ActionBarComponent } from './action-bar';
import { BuilderResultService } from '../builder-result.service';
import { BuilderStore } from '../state/builder.store';
import { StepStatus } from '../../core/models/pipeline-builder.model';

function setup(): {
  fixture: ComponentFixture<ActionBarComponent>;
  host: HTMLElement;
  store: BuilderStore;
} {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      BuilderStore,
      BuilderResultService,
    ],
  });
  const store = TestBed.inject(BuilderStore);
  const fixture = TestBed.createComponent(ActionBarComponent);
  fixture.detectChanges();
  return { fixture, host: fixture.nativeElement as HTMLElement, store };
}

function btn(host: HTMLElement, action: string): HTMLButtonElement {
  const el = host.querySelector<HTMLButtonElement>(
    `button[data-action="${action}"]`,
  );
  if (!el) throw new Error(`button[data-action=${action}] missing`);
  return el;
}

describe('ActionBarComponent', () => {
  it('when mounted, three buttons exist with data-action="optimize", "rebalance", "export"', () => {
    const { host } = setup();
    expect(host.querySelectorAll('button[data-action]').length).toBe(3);
    expect(host.querySelector('button[data-action="optimize"]')).not.toBeNull();
    expect(host.querySelector('button[data-action="rebalance"]')).not.toBeNull();
    expect(host.querySelector('button[data-action="export"]')).not.toBeNull();
  });

  it('when sessionId is null, all three buttons are disabled and aria-disabled', () => {
    const { fixture, host } = setup();
    fixture.detectChanges();

    for (const action of ['optimize', 'rebalance', 'export']) {
      const b = btn(host, action);
      expect(b.disabled).toBe(true);
      expect(b.getAttribute('aria-disabled')).toBe('true');
    }
  });

  it('when sessionId is set and status is Pending, Optimize/Rebalance/Export are enabled', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    fixture.detectChanges();

    for (const action of ['optimize', 'rebalance', 'export']) {
      const b = btn(host, action);
      expect(b.disabled).toBe(false);
      expect(b.getAttribute('aria-disabled')).toBe('false');
    }
  });

  it('when status is Running, Optimize and Rebalance are disabled while Export remains enabled', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    store.setStatus(StepStatus.Running);
    fixture.detectChanges();

    expect(btn(host, 'optimize').disabled).toBe(true);
    expect(btn(host, 'rebalance').disabled).toBe(true);
    expect(btn(host, 'export').disabled).toBe(false);
  });

  it('when status leaves Running back to Completed, Optimize/Rebalance re-enable', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    store.setStatus(StepStatus.Running);
    fixture.detectChanges();
    store.setStatus(StepStatus.Completed);
    fixture.detectChanges();

    expect(btn(host, 'optimize').disabled).toBe(false);
    expect(btn(host, 'rebalance').disabled).toBe(false);
  });

  it('when Optimize is clicked with sessionId set, store.optimize() is invoked exactly once', () => {
    const { fixture, host, store } = setup();
    const spy = spyOn(store, 'optimize');
    store.setSessionId('s1');
    fixture.detectChanges();

    btn(host, 'optimize').click();

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('when Rebalance is clicked with sessionId set, store.rebalance() is invoked exactly once', () => {
    const { fixture, host, store } = setup();
    const spy = spyOn(store, 'rebalance');
    store.setSessionId('s1');
    fixture.detectChanges();

    btn(host, 'rebalance').click();

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('when Export is clicked with sessionId set, store.export() is invoked exactly once', () => {
    const { fixture, host, store } = setup();
    const spy = spyOn(store, 'export');
    store.setSessionId('s1');
    fixture.detectChanges();

    btn(host, 'export').click();

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('when Optimize is clicked while disabled (sessionId null), store.optimize() is not invoked', () => {
    const { fixture, host, store } = setup();
    const spy = spyOn(store, 'optimize');
    fixture.detectChanges();

    btn(host, 'optimize').click();

    expect(spy).not.toHaveBeenCalled();
  });

  it('when mounted, an aria-live="polite" status region exists for surfacing in-flight state', () => {
    const { host } = setup();
    const region = host.querySelector('[aria-live="polite"]');
    expect(region).not.toBeNull();
  });

  it('when status is Running, the aria-live status region announces "Running"', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    store.setStatus(StepStatus.Running);
    fixture.detectChanges();
    const region = host.querySelector('[aria-live="polite"]');
    expect(region?.textContent ?? '').toContain('Running');
  });

  it('when status is Error, the aria-live status region announces "Error"', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    store.setStatus(StepStatus.Error);
    fixture.detectChanges();
    const region = host.querySelector('[aria-live="polite"]');
    expect(region?.textContent ?? '').toContain('Error');
  });

  it('when Optimize is clicked with sessionId set, both store.optimize() and store.triggerResultRun() fire exactly once', () => {
    const { fixture, host, store } = setup();
    const optSpy = spyOn(store, 'optimize');
    const triggerSpy = spyOn(store, 'triggerResultRun');
    store.setSessionId('s1');
    fixture.detectChanges();

    btn(host, 'optimize').click();

    expect(optSpy).toHaveBeenCalledTimes(1);
    expect(triggerSpy).toHaveBeenCalledTimes(1);
  });

  it('when resultStatus is "running", Optimize and Rebalance are disabled while Export stays enabled', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    store.setResultStatus('running');
    fixture.detectChanges();

    expect(btn(host, 'optimize').disabled).toBe(true);
    expect(btn(host, 'rebalance').disabled).toBe(true);
    expect(btn(host, 'export').disabled).toBe(false);
  });

  it('when resultStatus is "stale", the aria-live status region announces "Stale"', () => {
    const { fixture, host, store } = setup();
    store.setSessionId('s1');
    store.setResultStatus('stale');
    fixture.detectChanges();
    const region = host.querySelector('[aria-live="polite"]');
    expect(region?.textContent ?? '').toContain('Stale');
  });
});
