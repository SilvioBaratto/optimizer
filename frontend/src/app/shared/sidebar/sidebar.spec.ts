import { signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { configureTestBed } from '../../../testing';
import { SidebarComponent } from './sidebar';
import { BreakpointService } from '../../services/breakpoint.service';
import { PortfolioApiService } from '../../services/portfolio-api.service';
import { ICON_PROVIDER } from '../../icons';

describe('SidebarComponent', () => {
  let fixture: ComponentFixture<SidebarComponent>;
  let comp: SidebarComponent;
  // Fake breakpoint — never the real one (it runs ResizeObserver/innerWidth).
  const isMobile = signal(false);
  const isTablet = signal(false);

  beforeEach(async () => {
    isMobile.set(false);
    isTablet.set(false);
    const fakeBreakpoint = { isMobile, isTablet, chartHeight: signal(380) };
    // PortfolioPicker (rendered in the sidebar) only needs list(); stub it so no HTTP.
    const fakePortfolioApi = { list: () => of({ items: [], total: 0 }) };

    await configureTestBed({
      imports: [SidebarComponent],
      withRouter: true,
      withHttp: false,
      providers: [
        { provide: BreakpointService, useValue: fakeBreakpoint as unknown as BreakpointService },
        { provide: PortfolioApiService, useValue: fakePortfolioApi as unknown as PortfolioApiService },
        ICON_PROVIDER,
      ],
    });
    fixture = TestBed.createComponent(SidebarComponent);
    comp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('when on desktop, the sidebar is always shown', () => {
    expect(comp.showSidebar()).toBe(true);
  });

  it('when on mobile and closed, the sidebar is hidden', () => {
    isMobile.set(true);
    expect(comp.showSidebar()).toBe(false);
  });

  it('when on mobile and opened, the sidebar is shown', () => {
    isMobile.set(true);
    fixture.componentRef.setInput('isOpen', true);
    fixture.detectChanges();
    expect(comp.showSidebar()).toBe(true);
  });

  it('when on tablet and closed, the sidebar is hidden', () => {
    isTablet.set(true);
    expect(comp.showSidebar()).toBe(false);
  });

  it('when a nav item is clicked, closeSidebar is emitted', () => {
    let closed = false;
    comp.closeSidebar.subscribe(() => (closed = true));
    comp.onNavClick();
    expect(closed).toBe(true);
  });
});
