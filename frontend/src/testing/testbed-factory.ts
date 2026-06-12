import {
  EnvironmentProviders,
  Provider,
  Type,
  provideZonelessChangeDetection,
} from '@angular/core';
import {
  HttpErrorResponse,
  withInterceptors,
  provideHttpClient,
  type HttpInterceptorFn,
} from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed } from '@angular/core/testing';
import { throwError } from 'rxjs';

import { BACKGROUND_REQUEST } from '../app/core/http-context';

type TestProvider = Provider | EnvironmentProviders;

export interface ConfigureTestBedOptions {
  imports?: Type<unknown>[];
  providers?: TestProvider[];
  /** Add provideHttpClient + provideHttpClientTesting (default true). */
  withHttp?: boolean;
  /** Add provideRouter([]) (default false). */
  withRouter?: boolean;
}

/**
 * Functional interceptor that silently stubs requests tagged with the
 * BACKGROUND_REQUEST context token. Tagged requests return a 204 error so the
 * component's catchError path is exercised without polluting HttpTestingController.
 * Untagged requests (including service-level unit tests) pass through unchanged.
 */
const backgroundStubInterceptor: HttpInterceptorFn = (req, next) => {
  if (req.context.get(BACKGROUND_REQUEST)) {
    return throwError(
      () => new HttpErrorResponse({ status: 204, statusText: 'No Content (test stub)' }),
    );
  }
  return next(req);
};

/**
 * Configure a zoneless TestBed and compile components.
 *
 * Always leads with provideZonelessChangeDetection() — the app is zoneless and
 * omitting it hangs Karma. Never adds BrowserModule/CommonModule/any NgModule.
 */
export async function configureTestBed(
  options: ConfigureTestBedOptions = {},
): Promise<void> {
  const { imports = [], providers = [], withHttp = true, withRouter = false } = options;
  // Clear persisted portfolio state so a leaked `optimizer.currentPortfolioId`
  // from a prior spec can't make the root PortfolioContextService fire an
  // unmocked GET /portfolio/ (it now fetches the list lazily on a non-null id).
  localStorage.clear();
  TestBed.configureTestingModule({
    imports,
    providers: [...baseProviders(withHttp, withRouter), ...providers],
  });
  await TestBed.compileComponents();
}

function baseProviders(withHttp: boolean, withRouter: boolean): TestProvider[] {
  const providers: TestProvider[] = [provideZonelessChangeDetection()];
  if (withHttp) {
    providers.push(
      provideHttpClient(withInterceptors([backgroundStubInterceptor])),
      provideHttpClientTesting(),
    );
  }
  if (withRouter) {
    providers.push(provideRouter([]));
  }
  return providers;
}

/** Resolve the HttpTestingController from the active TestBed. */
export function injectHttp(): HttpTestingController {
  return TestBed.inject(HttpTestingController);
}
