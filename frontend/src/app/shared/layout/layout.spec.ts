import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { LayoutComponent } from './layout';
import { ICON_PROVIDER } from '../../icons';

describe('LayoutComponent', () => {
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

  function render(): HTMLElement {
    const fixture = TestBed.createComponent(LayoutComponent);
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  it('renders exactly one <app-context-bar>', () => {
    const el = render();
    expect(el.querySelectorAll('app-context-bar').length).toBe(1);
  });

  it('renders the mobile header with a menu button', () => {
    const el = render();
    const header = el.querySelector('header');
    expect(header).not.toBeNull();
    expect(header!.querySelector('button[aria-label="Open menu"]')).not.toBeNull();
  });

  it('renders the sidebar', () => {
    const el = render();
    expect(el.querySelectorAll('app-sidebar').length).toBe(1);
  });

  it('places the context bar inside #main-content above the scroll wrapper', () => {
    const el = render();
    const main = el.querySelector('#main-content');
    expect(main).not.toBeNull();
    const contextBar = main!.querySelector('app-context-bar');
    expect(contextBar).not.toBeNull();
    const scroll = main!.querySelector('.overflow-y-auto');
    expect(scroll).not.toBeNull();
    // Context bar appears before the scroll wrapper in document order.
    expect(
      contextBar!.compareDocumentPosition(scroll!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('applies sticky positioning classes to the context bar element', () => {
    const el = render();
    const contextBar = el.querySelector('app-context-bar');
    expect(contextBar).not.toBeNull();
    const classes = contextBar!.className;
    expect(classes).toContain('sticky');
    expect(classes).toContain('top-0');
  });
});
