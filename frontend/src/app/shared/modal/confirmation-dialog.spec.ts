import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { ConfirmationDialogComponent } from './confirmation-dialog';

describe('ConfirmationDialogComponent', () => {
  let fixture: ComponentFixture<ConfirmationDialogComponent>;
  let comp: ConfirmationDialogComponent;

  const confirmBtn = () => fixture.nativeElement.querySelector('[data-testid="confirm-dialog-confirm"]') as HTMLButtonElement;
  const cancelBtn = () => fixture.nativeElement.querySelector('[data-testid="confirm-dialog-cancel"]') as HTMLButtonElement;

  beforeEach(async () => {
    await configureTestBed({ imports: [ConfirmationDialogComponent], withHttp: false });
    fixture = TestBed.createComponent(ConfirmationDialogComponent);
    comp = fixture.componentInstance;
  });

  it('when closed, no dialog is rendered', () => {
    setInput(fixture, 'open', false);
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).toBeNull();
  });

  it('when open, the dialog is rendered', () => {
    setInput(fixture, 'open', true);
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).not.toBeNull();
  });

  it('when the confirm button is clicked, confirmed is emitted', () => {
    setInput(fixture, 'open', true);
    let confirmed = false;
    comp.confirmed.subscribe(() => (confirmed = true));
    confirmBtn().click();
    expect(confirmed).toBe(true);
  });

  it('when the cancel button is clicked, cancelled is emitted', () => {
    setInput(fixture, 'open', true);
    let cancelled = false;
    comp.cancelled.subscribe(() => (cancelled = true));
    cancelBtn().click();
    expect(cancelled).toBe(true);
  });

  it('when inflight, both buttons are disabled', () => {
    setInput(fixture, 'open', true);
    setInput(fixture, 'inflight', true);
    expect(confirmBtn().disabled).toBe(true);
    expect(cancelBtn().disabled).toBe(true);
  });

  it('when destructive, the confirm class uses the loss colour', () => {
    setInput(fixture, 'destructive', true);
    expect(comp.confirmClass()).toContain('bg-loss');
  });

  it('when not destructive, the confirm class uses the accent colour', () => {
    expect(comp.confirmClass()).toContain('bg-accent');
  });
});
