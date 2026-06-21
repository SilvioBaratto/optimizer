/**
 * Criterion 3 acceptance tests — source-blind, authored from AC only.
 *
 * AC: "Every page/panel shows a visible, non-blank error state (message or alert
 * component) when its primary API call returns an error."
 *
 * One test per wired page. Each test:
 *   1. Mounts the page component.
 *   2. Flushes its primary HTTP call with HTTP 500.
 *   3. Asserts a [role="alert"] element with non-blank text is rendered.
 *
 * Tests fail until the component both (a) catches the error with catchError and
 * (b) renders a visible error element with [role="alert"] and non-blank text.
 *
 * Selector contract: components must mark their error states with [role="alert"].
 * Tests will fail if components use a different attribute — that is intentional:
 * the criterion requires a "visible, non-blank error state", and [role="alert"] is
 * the semantic HTML marker for that concept in an Angular component.
 */

import { ApplicationRef } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { HttpTestingController } from '@angular/common/http/testing';

import { OptimizationStudioComponent } from '../../app/optimization-studio/optimization-studio';

import { configureTestBed } from '../';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Flush every pending HTTP request with a 500. */
function flushAllWith500(http: HttpTestingController): void {
  const pending = http.match(() => true);
  pending.forEach((req) =>
    req.flush({ detail: 'Internal Server Error' }, { status: 500, statusText: 'Internal Server Error' }),
  );
}

/** Assert that at least one [role="alert"] element is visible with non-blank text. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function expectNonBlankAlert(fixture: ComponentFixture<any>): void {
  const el = fixture.debugElement.query(By.css('[role="alert"]'));
  expect(el)
    .withContext('Expected a [role="alert"] element to be present in the DOM after a 500 response')
    .not.toBeNull();
  expect(el?.nativeElement.textContent?.trim())
    .withContext('Expected the [role="alert"] element to have non-blank text content')
    .not.toBe('');
}

// ---------------------------------------------------------------------------
// optimization-studio — primary call: POST /optimize (or initial setup call)
// ---------------------------------------------------------------------------

describe('Criterion 3 – OptimizationStudioComponent shows non-blank error state on primary API failure', () => {
  let http: HttpTestingController;
  let appRef: ApplicationRef;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    http = TestBed.inject(HttpTestingController);
    appRef = TestBed.inject(ApplicationRef);
  });

  afterEach(() => http.verify());

  it('when the primary API call returns HTTP 500, a [role="alert"] element with non-blank text is rendered', async () => {
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    flushAllWith500(http);
    appRef.tick();
    fixture.detectChanges();

    expectNonBlankAlert(fixture);
  });
});

