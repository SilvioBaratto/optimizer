import { throwError } from 'rxjs';

/**
 * Returns a `catchError` selector that re-throws the received error unchanged.
 *
 * Division of labor:
 *   - The `apiHttpInterceptor` owns cross-cutting concerns: it normalizes every
 *     `HttpErrorResponse` into an `ApiError { status, message, details }`,
 *     dispatches a global toast for 4xx/5xx (except silent statuses), and
 *     retries eligible GET requests with exponential back-off.
 *   - `mapHttpError` is intentionally an identity re-throw: it only passes the
 *     raw error downstream so a page component can observe it and render a
 *     contextual empty state without duplicating the interception logic.
 *
 * Usage:
 *   ```ts
 *   this.http.post<Foo>(url, body).pipe(catchError(mapHttpError()))
 *   ```
 */
export function mapHttpError(): (err: unknown) => ReturnType<typeof throwError> {
  return (err: unknown) => throwError(() => err);
}
