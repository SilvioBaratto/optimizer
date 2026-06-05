import { ChangeDetectionStrategy, Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ModalService } from './modal.service';

@Component({ template: '', changeDetection: ChangeDetectionStrategy.OnPush })
class DummyModal {}

describe('ModalService', () => {
  let svc: ModalService;

  beforeEach(async () => {
    await configureTestBed({ withHttp: false });
    svc = TestBed.inject(ModalService);
  });

  it('when injected twice, the providedIn root singleton resolves once', () => {
    expect(TestBed.inject(ModalService)).toBe(svc);
  });

  it('when starting, no modal config is present', () => {
    expect(svc.config()).toBeNull();
  });

  it('when opened without a size, the config defaults to md', () => {
    svc.open({ component: DummyModal });
    expect(svc.config()?.size).toBe('md');
    expect(svc.config()?.component).toBe(DummyModal);
  });

  it('when opened with an explicit size and title, both are stored', () => {
    svc.open({ component: DummyModal, size: 'lg', title: 'Export', inputs: { a: 1 } });
    expect(svc.config()?.size).toBe('lg');
    expect(svc.config()?.title).toBe('Export');
    expect(svc.config()?.inputs).toEqual({ a: 1 });
  });

  it('when closed, the config returns to null', () => {
    svc.open({ component: DummyModal });
    svc.close();
    expect(svc.config()).toBeNull();
  });
});
