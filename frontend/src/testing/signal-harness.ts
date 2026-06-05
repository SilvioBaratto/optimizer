import { ComponentFixture } from '@angular/core/testing';

// The app is zoneless: signal input()s must be set then change-detected before
// assertions. componentRef.setInput is the supported path for required inputs.

/** Set a signal `input()` by name on a mounted fixture, then run change detection. */
export function setInput<T>(
  fixture: ComponentFixture<T>,
  name: string,
  value: unknown,
): void {
  fixture.componentRef.setInput(name, value);
  fixture.detectChanges();
}
