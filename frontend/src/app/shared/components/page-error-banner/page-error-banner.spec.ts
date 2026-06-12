import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { By } from '@angular/platform-browser';

import { PageErrorBannerComponent } from './page-error-banner';

// Source-blind contract tests for PageErrorBannerComponent — issue #957.
// Derived solely from acceptance criteria: visible non-blank error state on
// page-level fatal errors; role="alert" for screen readers; retry affordance.

describe('PageErrorBannerComponent', () => {
  let fixture: ComponentFixture<PageErrorBannerComponent>;
  let component: PageErrorBannerComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PageErrorBannerComponent],
      providers: [provideZonelessChangeDetection()],
    }).compileComponents();

    fixture = TestBed.createComponent(PageErrorBannerComponent);
    component = fixture.componentInstance;
  });

  // ── Criterion: "every page/panel shows a visible, non-blank error state
  //   when its primary API call returns an error" ──────────────────────────

  it('when message is provided, message text appears in the DOM', () => {
    fixture.componentRef.setInput('message', 'Failed to load portfolio data');
    fixture.detectChanges();

    expect((fixture.nativeElement as HTMLElement).textContent)
      .toContain('Failed to load portfolio data');
  });

  it('when message is provided, rendered content is non-blank', () => {
    fixture.componentRef.setInput('message', 'Primary API call returned 500');
    fixture.detectChanges();

    const text = (fixture.nativeElement as HTMLElement).textContent?.trim();
    expect(text).toBeTruthy();
  });

  // ── Criterion: "outer element carries role=alert so screen readers
  //   announce the error" ────────────────────────────────────────────────

  it('when rendered, an element with role="alert" is present', () => {
    fixture.componentRef.setInput('message', 'Something went wrong');
    fixture.detectChanges();

    const alertEl = fixture.debugElement.query(By.css('[role="alert"]'));
    expect(alertEl).not.toBeNull();
  });

  // ── Criterion: "retry button is keyboard-focusable with an accessible
  //   label" ─────────────────────────────────────────────────────────────

  it('when rendered, a native <button> element is present for keyboard focus', () => {
    fixture.componentRef.setInput('message', 'Connection lost');
    fixture.detectChanges();

    const button = fixture.debugElement.query(By.css('button'));
    expect(button).not.toBeNull();
  });

  // ── Criterion: "emits a retry output (no payload) when the retry button
  //   is clicked" — spec requirement (a) loads on init / (b) handles error ─

  it('when retry button is clicked, retry output emits once', () => {
    fixture.componentRef.setInput('message', 'Something went wrong');
    fixture.detectChanges();

    let emitCount = 0;
    component.retry.subscribe(() => { emitCount++; });

    const button = fixture.debugElement.query(By.css('button'));
    button.triggerEventHandler('click', null);

    expect(emitCount).toBe(1);
  });

  it('when retry button is clicked multiple times, retry output emits for each click', () => {
    fixture.componentRef.setInput('message', 'Error loading data');
    fixture.detectChanges();

    let emitCount = 0;
    component.retry.subscribe(() => { emitCount++; });

    const button = fixture.debugElement.query(By.css('button'));
    button.triggerEventHandler('click', null);
    button.triggerEventHandler('click', null);

    expect(emitCount).toBe(2);
  });

  // ── Criterion: different message inputs each render their own text ───────

  it('when message changes, updated message text appears in the DOM', () => {
    fixture.componentRef.setInput('message', 'Initial error');
    fixture.detectChanges();

    fixture.componentRef.setInput('message', 'Updated error message');
    fixture.detectChanges();

    expect((fixture.nativeElement as HTMLElement).textContent)
      .toContain('Updated error message');
  });
});
