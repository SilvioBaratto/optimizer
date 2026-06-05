import { ChangeDetectionStrategy, Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ModalContainerComponent } from './modal-container';
import { ModalService } from './modal.service';

@Component({
  selector: 'modal-stub',
  template: '<button type="button">Stub body</button>',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class StubModalContent {}

describe('ModalContainerComponent', () => {
  let fixture: ComponentFixture<ModalContainerComponent>;
  let comp: ModalContainerComponent;
  let modal: ModalService;

  function openModal(): void {
    modal.open({ component: StubModalContent, title: 'Hello', size: 'sm' });
    fixture.detectChanges();
  }

  beforeEach(async () => {
    await configureTestBed({ imports: [ModalContainerComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ModalContainerComponent);
    comp = fixture.componentInstance;
    modal = TestBed.inject(ModalService);
    fixture.detectChanges(); // resolve viewChild(outlet) so the effect can mount
  });

  it('when config is null, nothing is rendered', () => {
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).toBeNull();
  });

  it('when opened, the dialog, title and projected component are rendered', () => {
    openModal();
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('#modal-title').textContent).toContain('Hello');
    expect(fixture.nativeElement.textContent).toContain('Stub body');
  });

  it('when the backdrop is clicked, the modal closes', () => {
    openModal();
    spyOn(modal, 'close');
    (fixture.nativeElement.querySelector('.bg-black\\/40') as HTMLElement).click();
    expect(modal.close).toHaveBeenCalled();
  });

  it('when Escape is pressed, the modal closes', () => {
    openModal();
    spyOn(modal, 'close');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(modal.close).toHaveBeenCalled();
  });

  it('when given a size, sizeClass maps to the matching max-width', () => {
    expect(comp.sizeClass('sm')).toBe('max-w-sm');
    expect(comp.sizeClass('md')).toBe('max-w-lg');
    expect(comp.sizeClass('lg')).toBe('max-w-2xl');
  });
});
