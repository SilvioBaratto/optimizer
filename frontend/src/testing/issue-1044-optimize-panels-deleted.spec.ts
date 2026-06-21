/**
 * Issue #1044 — Criterion 4 [T3]
 *
 * "preprocessing-panel, moment-panel, view-panel, pipeline-builder component dirs
 *  and their specs are deleted; optimization-studio.ts/.html no longer import or
 *  render them."
 *
 * Renders OptimizationStudioComponent and asserts that the selectors for the
 * four deleted panels are absent from the compiled DOM. CUSTOM_ELEMENTS_SCHEMA
 * ensures any selector that slips through appears as a DOM element (failing
 * the toBeNull() assertion).
 *
 * Also asserts that the new simplified layout renders the optimizer-panel and
 * results-panel as the sole interactive panels.
 */

import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideNoopAnimations } from '@angular/platform-browser/animations';
import { provideRouter } from '@angular/router';
import { installResizeObserverStub } from './test-globals';
import { ICON_PROVIDER } from '../app/icons';
import { OptimizationStudioComponent } from '../app/optimization-studio/optimization-studio';

describe('OptimizationStudioComponent — deleted panels absent from DOM (issue #1044 criterion 4)', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [OptimizationStudioComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideNoopAnimations(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    }).compileComponents();

    fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();
  });

  afterEach(() => localStorage.clear());

  it('when optimize page renders, app-preprocessing-panel is not in the DOM', () => {
    expect(fixture.debugElement.query(By.css('app-preprocessing-panel')))
      .withContext('preprocessing-panel must be deleted — its selector must not appear in the template')
      .toBeNull();
  });

  it('when optimize page renders, app-moment-panel is not in the DOM', () => {
    expect(fixture.debugElement.query(By.css('app-moment-panel')))
      .withContext('moment-panel must be deleted — its selector must not appear in the template')
      .toBeNull();
  });

  it('when optimize page renders, app-view-panel is not in the DOM', () => {
    expect(fixture.debugElement.query(By.css('app-view-panel')))
      .withContext('view-panel must be deleted — its selector must not appear in the template')
      .toBeNull();
  });

  it('when optimize page renders, app-pipeline-builder is not in the DOM', () => {
    expect(fixture.debugElement.query(By.css('app-pipeline-builder')))
      .withContext('pipeline-builder must be deleted — its selector must not appear in the template')
      .toBeNull();
  });

  it('when optimize page renders, app-optimizer-panel is present in the DOM', () => {
    expect(fixture.debugElement.query(By.css('app-optimizer-panel')))
      .withContext('optimizer-panel must be present in the simplified layout')
      .not.toBeNull();
  });

  it('when optimize page renders, app-results-panel is present in the DOM', () => {
    expect(fixture.debugElement.query(By.css('app-results-panel')))
      .withContext('results-panel must be present in the simplified layout')
      .not.toBeNull();
  });
});
