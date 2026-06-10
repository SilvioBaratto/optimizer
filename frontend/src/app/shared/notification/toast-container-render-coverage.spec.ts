/**
 * Render-coverage spec for ToastContainerComponent — issue #907.
 *
 * Coverage gap addressed:
 *   The sibling toast-container.spec.ts calls `comp.borderClass(level)` directly
 *   (pure-method assertion) and only renders a `success` toast in the DOM.
 *   The `error` / `warning` / `info` [class] bindings on the rendered <div> and
 *   the @for element-count branch therefore stay uncovered at the DOM level.
 *   This spec raises one toast per level via the injected NotificationService and
 *   asserts the RENDERED toast <div> carries the expected border token.
 *
 * Already covered elsewhere — NOT duplicated here:
 *   - alert-banner.ts  (@if isDismissed, @if dismissible, 4-level [class])
 *     → fully covered by alert-banner.spec.ts
 *   - notification.service.ts  (autoDismissMs > 0 setTimeout branch)
 *     → fully covered by notification.service.spec.ts
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ToastContainerComponent } from './toast-container';
import { NotificationService, type NotificationLevel } from './notification.service';

describe('ToastContainerComponent (render coverage)', () => {
  let fixture: ComponentFixture<ToastContainerComponent>;
  let notifications: NotificationService;

  /** All direct children of the [role="status"] host — one per rendered toast. */
  const toastEls = (): NodeListOf<HTMLElement> =>
    fixture.nativeElement.querySelectorAll('[role="status"] > div');

  beforeEach(async () => {
    await configureTestBed({ imports: [ToastContainerComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ToastContainerComponent);
    notifications = TestBed.inject(NotificationService);
    fixture.detectChanges();
  });

  // ── @for empty branch ────────────────────────────────────────────────────────

  it('when no toasts are raised, no toast elements render', () => {
    expect(toastEls().length).toBe(0);
  });

  // ── [class] binding — rendered DOM assertions (data-driven) ─────────────────

  const CASES: ReadonlyArray<[NotificationLevel, string]> = [
    ['success', 'border-l-gain'],
    ['error',   'border-l-loss'],
    ['warning', 'border-l-warning'],
    ['info',    'border-l-accent'],
  ];

  for (const [level, token] of CASES) {
    it(`when a ${level} toast is raised, the rendered <div> carries '${token}' and 'border-l-4'`, () => {
      // Arrange — autoDismissMs: 0 prevents the setTimeout branch from clearing the toast
      notifications[level]('test message', { autoDismissMs: 0 });

      // Act
      fixture.detectChanges();

      // Assert — DOM [class] binding, not the pure method
      const el = toastEls()[0];
      expect(el).withContext(`expected a toast <div> to exist for level "${level}"`).toBeTruthy();
      expect(el.className).withContext('border-l-4 utility must be present').toContain('border-l-4');
      expect(el.className).withContext(`color token for level "${level}" must be present`).toContain(token);
    });
  }

  // ── @for multiple-elements branch ────────────────────────────────────────────

  it('when two toasts are raised, two toast elements render', () => {
    notifications.error('first', { autoDismissMs: 0 });
    notifications.info('second', { autoDismissMs: 0 });

    fixture.detectChanges();

    expect(toastEls().length).toBe(2);
  });
});
