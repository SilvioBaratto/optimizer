import { TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { GlobalSearchService } from './global-search.service';

describe('GlobalSearchService', () => {
  let svc: GlobalSearchService;

  beforeEach(async () => {
    await configureTestBed({ withHttp: false });
    svc = TestBed.inject(GlobalSearchService);
  });

  it('when injected twice, the providedIn root singleton resolves once', () => {
    expect(TestBed.inject(GlobalSearchService)).toBe(svc);
  });

  it('when starting, the panel is closed', () => {
    expect(svc.isOpen()).toBe(false);
  });

  it('when opened, isOpen is true', () => {
    svc.open();
    expect(svc.isOpen()).toBe(true);
  });

  it('when closed after opening, isOpen is false', () => {
    svc.open();
    svc.close();
    expect(svc.isOpen()).toBe(false);
  });

  it('when toggled from closed, isOpen flips to true then false', () => {
    svc.toggle();
    expect(svc.isOpen()).toBe(true);
    svc.toggle();
    expect(svc.isOpen()).toBe(false);
  });
});
