/**
 * Contract tests — issue #1049, criterion 2
 *
 * "The removed widgets (efficient-frontier, backtest-sparkline, drift-overlay,
 *  trade-list, diagnostics-strip) and the builder canvas/inputs/analytics panes
 *  are gone."
 *
 * Tests are source-blind: derived from the acceptance criterion only.
 * Each test checks that a named artefact is absent from the rendered HTML.
 * Tests are RED today (the widgets are still present) and will turn GREEN once
 * the template is cleaned up.
 *
 * String matching is intentionally broad — it catches component selectors
 * (e.g. <app-efficient-frontier>), CSS class names, data-* attributes, and IDs
 * that embed the widget name.  NO_ERRORS_SCHEMA is used so that the component
 * renders even before its child components are declared.
 */

import { NO_ERRORS_SCHEMA } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { PortfolioDto } from '../app/core/models/portfolio-api.model';
import { PortfolioBuilderComponent } from '../app/portfolio-builder/portfolio-builder';
import { PortfolioApiService } from '../app/core/services/portfolio-api.service';

describe('PortfolioBuilder — Removed Widgets', () => {
  let html: string;

  beforeEach(() => {
    const apiSpy = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', ['create']);
    apiSpy.create.and.returnValue(of({} as PortfolioDto));

    TestBed.configureTestingModule({
      imports: [PortfolioBuilderComponent],
      providers: [{ provide: PortfolioApiService, useValue: apiSpy }],
      schemas: [NO_ERRORS_SCHEMA],
    });

    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    fixture.detectChanges();
    html = (fixture.nativeElement as HTMLElement).innerHTML;
  });

  // ── Removed widgets ───────────────────────────────────────────────────────

  it('when portfolio-builder renders, efficient-frontier widget is absent', () => {
    expect(html).not.toContain('efficient-frontier');
  });

  it('when portfolio-builder renders, backtest-sparkline widget is absent', () => {
    expect(html).not.toContain('backtest-sparkline');
  });

  it('when portfolio-builder renders, drift-overlay widget is absent', () => {
    expect(html).not.toContain('drift-overlay');
  });

  it('when portfolio-builder renders, trade-list widget is absent', () => {
    expect(html).not.toContain('trade-list');
  });

  it('when portfolio-builder renders, diagnostics-strip widget is absent', () => {
    expect(html).not.toContain('diagnostics-strip');
  });

  // ── Removed panes ────────────────────────────────────────────────────────

  it('when portfolio-builder renders, builder canvas pane is absent', () => {
    expect(html).not.toContain('builder-canvas');
  });

  it('when portfolio-builder renders, builder inputs pane is absent', () => {
    expect(html).not.toContain('builder-inputs');
  });

  it('when portfolio-builder renders, builder analytics pane is absent', () => {
    expect(html).not.toContain('builder-analytics');
  });
});
