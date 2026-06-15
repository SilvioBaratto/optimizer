/**
 * Source-blind spec for the `mapHttpError` helper (issue #1008).
 *
 * Derived solely from acceptance criteria — no implementation source was read.
 * Every test must fail until `frontend/src/app/core/http-error.ts` is created
 * and exports a conforming `mapHttpError`.
 *
 * Criteria under test:
 *   AC-1  A single exported `mapHttpError` lives in `core/http-error.ts`.
 *   AC-3  A unit spec proves the helper re-throws the exact input error
 *         unchanged (identity).
 *   AC-3b The returned stream errors (does not complete with a value).
 */

import { throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

// This import will fail (module does not exist yet) — that is the intended
// Red state. When the file is created and the export is present all tests
// in this suite become runnable.
import { mapHttpError } from './http-error';

describe('mapHttpError (core/http-error)', () => {
  // -----------------------------------------------------------------------
  // AC-1: the export exists and is a factory function
  // -----------------------------------------------------------------------

  it('when imported, mapHttpError is a function', () => {
    expect(typeof mapHttpError).toBe('function');
  });

  it('when called, mapHttpError returns an operator function', () => {
    const operator = mapHttpError();
    expect(typeof operator).toBe('function');
  });

  // -----------------------------------------------------------------------
  // AC-3: identity re-throw — exact same reference must reach the subscriber
  // -----------------------------------------------------------------------

  it('when an Error is thrown, the subscriber receives the exact same Error reference', (done) => {
    const inputError = new Error('original error');

    throwError(() => inputError)
      .pipe(catchError(mapHttpError()))
      .subscribe({
        next: () => fail('must not emit a next value'),
        error: (err: unknown) => {
          expect(err).toBe(inputError);
          done();
        },
        complete: () => fail('must not complete'),
      });
  });

  it('when a plain object is thrown, the subscriber receives the exact same object reference', (done) => {
    const inputObj = { status: 422, error: { detail: 'validation error' } };

    throwError(() => inputObj)
      .pipe(catchError(mapHttpError()))
      .subscribe({
        next: () => fail('must not emit a next value'),
        error: (err: unknown) => {
          expect(err).toBe(inputObj);
          done();
        },
        complete: () => fail('must not complete'),
      });
  });

  it('when a string error is thrown, the subscriber receives the exact same string', (done) => {
    const inputString = 'unexpected string error';

    throwError(() => inputString)
      .pipe(catchError(mapHttpError()))
      .subscribe({
        next: () => fail('must not emit a next value'),
        error: (err: unknown) => {
          expect(err).toBe(inputString);
          done();
        },
        complete: () => fail('must not complete'),
      });
  });

  it('when null is thrown, the subscriber receives null', (done) => {
    throwError(() => null)
      .pipe(catchError(mapHttpError()))
      .subscribe({
        next: () => fail('must not emit a next value'),
        error: (err: unknown) => {
          expect(err).toBeNull();
          done();
        },
        complete: () => fail('must not complete'),
      });
  });

  // -----------------------------------------------------------------------
  // AC-3b: the stream errors — subscriber's error callback fires, not next
  // -----------------------------------------------------------------------

  it('when an error is thrown, the next callback is never called', (done) => {
    const nextSpy = jasmine.createSpy('next');

    throwError(() => new Error('any'))
      .pipe(catchError(mapHttpError()))
      .subscribe({
        next: nextSpy,
        error: () => {
          expect(nextSpy).not.toHaveBeenCalled();
          done();
        },
        complete: () => fail('must not complete'),
      });
  });

  it('when an error is thrown, the complete callback is never called', (done) => {
    const completeSpy = jasmine.createSpy('complete');

    throwError(() => new Error('any'))
      .pipe(catchError(mapHttpError()))
      .subscribe({
        next: () => fail('must not emit a next value'),
        error: () => {
          expect(completeSpy).not.toHaveBeenCalled();
          done();
        },
        complete: completeSpy,
      });
  });

  it('when an error is thrown, the error callback fires exactly once', (done) => {
    const errorSpy = jasmine.createSpy('error');

    throwError(() => new Error('once'))
      .pipe(catchError(mapHttpError()))
      .subscribe({
        error: (err: unknown) => {
          errorSpy(err);
          expect(errorSpy).toHaveBeenCalledTimes(1);
          done();
        },
      });
  });

  // -----------------------------------------------------------------------
  // Independence: calling mapHttpError() twice creates independent operators
  // Each call must be self-contained — shared mutable state between calls
  // would violate the single-responsibility and identity guarantees.
  // -----------------------------------------------------------------------

  it('when mapHttpError is called multiple times, each invocation re-throws independently', (done) => {
    const errorA = new Error('error A');
    const errorB = { code: 'B' };
    let received: unknown[] = [];

    throwError(() => errorA)
      .pipe(catchError(mapHttpError()))
      .subscribe({
        error: (err: unknown) => {
          received.push(err);
          // Now test the second independent operator instance
          throwError(() => errorB)
            .pipe(catchError(mapHttpError()))
            .subscribe({
              error: (err2: unknown) => {
                received.push(err2);
                expect(received[0]).toBe(errorA);
                expect(received[1]).toBe(errorB);
                done();
              },
            });
        },
      });
  });
});
