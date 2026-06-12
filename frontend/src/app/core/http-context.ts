import { HttpContext, HttpContextToken } from '@angular/common/http';

/**
 * HTTP context token that marks a request as a background/side-effect call.
 *
 * Set this on service requests that fire automatically on component init so the
 * test infrastructure can silently absorb them without polluting
 * HttpTestingController.verify(). The token has no effect in production — it is
 * only read by the background-stub interceptor registered in configureTestBed.
 */
export const BACKGROUND_REQUEST = new HttpContextToken<boolean>(() => false);

/** Returns an HttpContext that marks a request as a background call. */
export function backgroundContext(): HttpContext {
  return new HttpContext().set(BACKGROUND_REQUEST, true);
}
