import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ToastContainerComponent } from './toast-container';
import { NotificationService, type NotificationLevel } from './notification.service';

describe('ToastContainerComponent', () => {
  let fixture: ComponentFixture<ToastContainerComponent>;
  let comp: ToastContainerComponent;
  let notifications: NotificationService;

  const toastEls = () => fixture.nativeElement.querySelectorAll('[role="status"] > div');

  beforeEach(async () => {
    await configureTestBed({ imports: [ToastContainerComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ToastContainerComponent);
    comp = fixture.componentInstance;
    notifications = TestBed.inject(NotificationService);
    fixture.detectChanges();
  });

  it('when there are no toasts, none are rendered', () => {
    expect(toastEls().length).toBe(0);
  });

  it('when a success toast is raised, it is rendered with its message', () => {
    notifications.success('Saved', { autoDismissMs: 0 });
    fixture.detectChanges();
    expect(toastEls().length).toBe(1);
    expect(fixture.nativeElement.textContent).toContain('Saved');
  });

  it('when a toast is dismissed, it is removed', () => {
    notifications.success('Saved', { autoDismissMs: 0 });
    fixture.detectChanges();
    const id = notifications.toasts()[0].id;
    notifications.dismiss(id);
    fixture.detectChanges();
    expect(toastEls().length).toBe(0);
  });

  const cases: ReadonlyArray<[NotificationLevel, string]> = [
    ['success', 'border-l-4 border-l-gain'],
    ['error', 'border-l-4 border-l-loss'],
    ['warning', 'border-l-4 border-l-warning'],
    ['info', 'border-l-4 border-l-accent'],
  ];

  for (const [level, classes] of cases) {
    it(`when level is ${level}, the border class matches`, () => {
      expect(comp.borderClass(level)).toBe(classes);
    });
  }
});
