/**
 * Issue #1036 — LayoutComponent must not render <app-context-bar> in the DOM.
 * Source-blind: derived from acceptance criteria only.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { LayoutComponent } from './layout';
import { ICON_PROVIDER } from '../../icons';

describe('LayoutComponent — context-bar removed (#1036)', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [LayoutComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideRouter([]),
        provideNoopAnimations(),
        ICON_PROVIDER,
      ],
    });
  });

  it('when layout is rendered, no app-context-bar element appears in the DOM', () => {
    const fixture = TestBed.createComponent(LayoutComponent);
    fixture.detectChanges();
    const contextBar = (fixture.nativeElement as HTMLElement).querySelector('app-context-bar');
    expect(contextBar).toBeNull();
  });
});
