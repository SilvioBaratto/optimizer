import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ModalContainerComponent } from './modal-container';
import { ModalService } from './modal.service';

// Render-coverage spec: focus-trap branches, no-title arm, X-button close and
// focus restore. The fixture is attached to document.body so element.focus()
// updates document.activeElement. Basic open/close lives in modal-container.spec.ts.

@Component({
  selector: 'stub-with-button',
  template: '<button type="button" id="stub-btn">Body</button>',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class StubWithButton {}

@Component({
  selector: 'stub-no-focusable',
  template: '<span>plain text only</span>',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class StubNoFocusable {}

@Component({
  selector: 'stub-rich',
  template: '<button id="b1">1</button><input id="i1" /><button id="b2">2</button>',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class StubRich {
  readonly label = input('');
}

describe('ModalContainerComponent render-coverage', () => {
  let fixture: ComponentFixture<ModalContainerComponent>;
  let comp: ModalContainerComponent;
  let modal: ModalService;

  beforeEach(async () => {
    await configureTestBed({ imports: [ModalContainerComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ModalContainerComponent);
    comp = fixture.componentInstance;
    modal = TestBed.inject(ModalService);
    document.body.appendChild(fixture.nativeElement);
    fixture.detectChanges();
  });

  afterEach(() => {
    modal.close();
    fixture.nativeElement.remove();
  });

  it('when opened without a title, the header is omitted but the dialog renders', () => {
    modal.open({ component: StubWithButton, size: 'md' });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('#modal-title')).toBeNull();
    const dialog = fixture.nativeElement.querySelector('[role="dialog"]') as HTMLElement;
    expect(dialog.getAttribute('aria-labelledby')).toBeNull();
  });

  it('when the close (X) button is clicked, the modal closes', () => {
    modal.open({ component: StubWithButton, title: 'T', size: 'sm' });
    fixture.detectChanges();
    spyOn(modal, 'close');
    (fixture.nativeElement.querySelector('[aria-label="Close"]') as HTMLElement).click();
    expect(modal.close).toHaveBeenCalled();
  });

  it('when given an unknown size, sizeClass returns undefined', () => {
    expect(comp.sizeClass('xl' as 'sm')).toBeUndefined();
  });

  it('when a non-Tab key is pressed, onKeydown returns without trapping focus', () => {
    modal.open({ component: StubWithButton, title: 'T', size: 'sm' });
    fixture.detectChanges();
    expect(() => comp.onKeydown(new KeyboardEvent('keydown', { key: 'a' }))).not.toThrow();
  });

  it('when no modal is open, a Tab keydown is ignored', () => {
    expect(() => comp.onKeydown(new KeyboardEvent('keydown', { key: 'Tab' }))).not.toThrow();
  });

  it('when Tab is pressed on the last focusable element, focus wraps to the first', () => {
    modal.open({ component: StubWithButton, title: 'T', size: 'sm' });
    fixture.detectChanges();
    const focusable = fixture.nativeElement.querySelectorAll('a[href], button, input, [tabindex]');
    const first = focusable[0] as HTMLElement;
    const last = focusable[focusable.length - 1] as HTMLElement;
    last.focus();
    expect(document.activeElement).toBe(last);

    comp.onKeydown(new KeyboardEvent('keydown', { key: 'Tab' }));
    expect(document.activeElement).toBe(first);
  });

  it('when Shift+Tab is pressed on the first focusable element, focus wraps to the last', () => {
    modal.open({ component: StubWithButton, title: 'T', size: 'sm' });
    fixture.detectChanges();
    const focusable = fixture.nativeElement.querySelectorAll('a[href], button, input, [tabindex]');
    const first = focusable[0] as HTMLElement;
    const last = focusable[focusable.length - 1] as HTMLElement;
    first.focus();
    expect(document.activeElement).toBe(first);

    comp.onKeydown(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true }));
    expect(document.activeElement).toBe(last);
  });

  it('when the dialog has no focusable elements, a Tab keydown is a no-op', () => {
    modal.open({ component: StubNoFocusable, size: 'sm' });
    fixture.detectChanges();
    expect(() => comp.onKeydown(new KeyboardEvent('keydown', { key: 'Tab' }))).not.toThrow();
  });

  it('when opened with inputs, the projected component is created and rendered', () => {
    modal.open({ component: StubRich, title: 'T', size: 'lg', inputs: { label: 'hi' } });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('#i1')).not.toBeNull();
  });

  it('when Tab is pressed on a middle element, focus is not wrapped', () => {
    modal.open({ component: StubRich, title: 'T', size: 'lg' });
    fixture.detectChanges();
    const mid = fixture.nativeElement.querySelector('#i1') as HTMLElement;
    mid.focus();
    comp.onKeydown(new KeyboardEvent('keydown', { key: 'Tab' }));
    expect(document.activeElement).toBe(mid);
  });

  it('when the dialog view has not rendered yet, a Tab keydown is ignored', () => {
    modal.open({ component: StubWithButton, title: 'T', size: 'sm' });
    // No detectChanges → the dialog viewChild is unresolved (dialog is null).
    expect(() => comp.onKeydown(new KeyboardEvent('keydown', { key: 'Tab' }))).not.toThrow();
    fixture.detectChanges();
  });

  it('when the modal closes, focus is restored to the previously focused element', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    expect(document.activeElement).toBe(trigger);

    modal.open({ component: StubWithButton, title: 'T', size: 'sm' });
    fixture.detectChanges();

    modal.close();
    fixture.detectChanges();
    expect(document.activeElement).toBe(trigger);

    trigger.remove();
  });
});
