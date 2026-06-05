import { TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { NotificationService } from './notification.service';

describe('NotificationService', () => {
  let svc: NotificationService;

  beforeEach(async () => {
    // Install the fake clock AFTER the async TestBed compile so the real timer
    // backing compileComponents' promise resolution is not intercepted.
    await configureTestBed({ withHttp: false });
    jasmine.clock().install();
    svc = TestBed.inject(NotificationService);
  });

  afterEach(() => jasmine.clock().uninstall());

  it('when injected twice, the providedIn root singleton resolves once', () => {
    expect(TestBed.inject(NotificationService)).toBe(svc);
  });

  it('when starting, the toast list is empty', () => {
    expect(svc.toasts()).toEqual([]);
  });

  const levels = ['success', 'error', 'warning', 'info'] as const;
  for (const level of levels) {
    it(`when ${level} is raised, a toast of that level is added`, () => {
      svc[level]('hello');
      expect(svc.toasts().length).toBe(1);
      expect(svc.toasts()[0].level).toBe(level);
      expect(svc.toasts()[0].message).toBe('hello');
    });
  }

  it('when dismiss is called with an id, that toast is removed', () => {
    svc.success('a', { autoDismissMs: 0 });
    const id = svc.toasts()[0].id;
    svc.dismiss(id);
    expect(svc.toasts()).toEqual([]);
  });

  it('when the auto-dismiss delay elapses, the toast clears itself', () => {
    svc.info('auto');
    expect(svc.toasts().length).toBe(1);
    jasmine.clock().tick(5000);
    expect(svc.toasts().length).toBe(0);
  });

  it('when autoDismissMs is zero, the toast persists past any delay', () => {
    svc.error('sticky', { autoDismissMs: 0 });
    jasmine.clock().tick(60_000);
    expect(svc.toasts().length).toBe(1);
  });
});
