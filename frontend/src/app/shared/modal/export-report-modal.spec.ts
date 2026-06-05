import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ExportReportModalComponent } from './export-report-modal';
import { ModalService } from './modal.service';
import { NotificationService } from '../notification/notification.service';

describe('ExportReportModalComponent', () => {
  let fixture: ComponentFixture<ExportReportModalComponent>;
  let comp: ExportReportModalComponent;
  let modal: ModalService;
  let notifications: NotificationService;

  beforeEach(async () => {
    await configureTestBed({ imports: [ExportReportModalComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ExportReportModalComponent);
    comp = fixture.componentInstance;
    modal = TestBed.inject(ModalService);
    notifications = TestBed.inject(NotificationService);
    spyOn(modal, 'close');
    spyOn(notifications, 'info');
  });

  it('when first created, the default sections are selected', () => {
    expect(comp.selectedSections()).toEqual(['summary', 'performance', 'allocation', 'risk']);
  });

  it('when a selected section is toggled, it is removed', () => {
    comp.toggleSection('summary');
    expect(comp.isSectionSelected('summary')).toBe(false);
  });

  it('when an unselected section is toggled, it is added', () => {
    comp.toggleSection('appendix');
    expect(comp.isSectionSelected('appendix')).toBe(true);
  });

  it('when generate is invoked, an info toast is raised and the modal closes', () => {
    comp.onGenerate();
    expect(notifications.info).toHaveBeenCalledWith('Report generation coming soon');
    expect(modal.close).toHaveBeenCalled();
  });

  it('when cancel is invoked, the modal closes', () => {
    comp.onCancel();
    expect(modal.close).toHaveBeenCalled();
  });
});
