import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { AlertBannerComponent } from './alert-banner';

type Level = 'success' | 'error' | 'warning' | 'info';

describe('AlertBannerComponent', () => {
  let fixture: ComponentFixture<AlertBannerComponent>;
  let comp: AlertBannerComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [AlertBannerComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(AlertBannerComponent);
    comp = fixture.componentInstance;
  });

  const cases: ReadonlyArray<[Level, string]> = [
    ['success', 'bg-gain/10 border-gain/20'],
    ['error', 'bg-loss/10 border-loss/20'],
    ['warning', 'bg-warning/10 border-warning/20'],
    ['info', 'bg-accent/10 border-accent/20'],
  ];

  for (const [level, classes] of cases) {
    it(`when level is ${level}, the matching classes are returned`, () => {
      setInput(fixture, 'message', 'msg');
      setInput(fixture, 'level', level);
      expect(comp.levelClasses()).toBe(classes);
    });
  }

  it('when not dismissed, the banner and its message are rendered', () => {
    setInput(fixture, 'message', 'Heads up');
    expect(fixture.nativeElement.textContent).toContain('Heads up');
  });

  it('when the dismiss button is clicked, it dismisses and emits', () => {
    setInput(fixture, 'message', 'Heads up');
    let dismissed = false;
    comp.dismissed.subscribe(() => (dismissed = true));
    fixture.nativeElement.querySelector('button[aria-label="Dismiss"]').click();
    fixture.detectChanges();
    expect(comp.isDismissed()).toBe(true);
    expect(dismissed).toBe(true);
    expect(fixture.nativeElement.querySelector('div')).toBeNull();
  });

  it('when not dismissible, no dismiss button is rendered', () => {
    setInput(fixture, 'message', 'Heads up');
    setInput(fixture, 'dismissible', false);
    expect(fixture.nativeElement.querySelector('button[aria-label="Dismiss"]')).toBeNull();
  });
});
