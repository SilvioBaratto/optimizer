import {
  EnvironmentProviders,
  Provider,
  Type,
  provideZonelessChangeDetection,
} from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed } from '@angular/core/testing';

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
 * Configure a zoneless TestBed and compile components.
 *
 * Always leads with provideZonelessChangeDetection() — the app is zoneless and
 * omitting it hangs Karma. Never adds BrowserModule/CommonModule/any NgModule.
 */
export async function configureTestBed(
  options: ConfigureTestBedOptions = {},
): Promise<void> {
  const { imports = [], providers = [], withHttp = true, withRouter = false } = options;
  TestBed.configureTestingModule({
    imports,
    providers: [...baseProviders(withHttp, withRouter), ...providers],
  });
  await TestBed.compileComponents();
}

function baseProviders(withHttp: boolean, withRouter: boolean): TestProvider[] {
  const providers: TestProvider[] = [provideZonelessChangeDetection()];
  if (withHttp) {
    providers.push(provideHttpClient(), provideHttpClientTesting());
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
