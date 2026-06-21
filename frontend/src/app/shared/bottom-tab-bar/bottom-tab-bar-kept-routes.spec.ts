/**
 * Issue #1036 — BottomTabBarComponent lists only kept routes; no deleted
 * routes (/risk-center, /ai-control-room) appear in the rendered DOM.
 * Source-blind: derived from acceptance criteria only.
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { BottomTabBarComponent } from './bottom-tab-bar';
import { ICON_PROVIDER } from '../../icons';

const KEPT_ROUTES = [
  '/',
  '/portfolio-builder',
  '/optimization-studio',
  '/backtesting',
  '/macro-intelligence',
  '/settings',
];

const DELETED_ROUTES = [
  '/risk-center',
  '/ai-control-room',
  '/rebalancing',
  '/attribution',
  '/factor-research',
];

describe('BottomTabBarComponent — kept routes only (#1036)', () => {
  let fixture: ComponentFixture<BottomTabBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BottomTabBarComponent],
      providers: [provideRouter([]), ICON_PROVIDER],
    }).compileComponents();

    fixture = TestBed.createComponent(BottomTabBarComponent);
    fixture.detectChanges();
  });

  DELETED_ROUTES.forEach(route => {
    it(`when rendered, no tab anchor links to deleted route "${route}"`, () => {
      const anchors = Array.from<HTMLAnchorElement>(
        fixture.nativeElement.querySelectorAll('a'),
      );
      const hrefs = anchors.map(a => a.getAttribute('href'));
      expect(hrefs).not.toContain(route);
    });
  });

  it('when all anchor hrefs are collected, each one belongs to the kept-route set', () => {
    const anchors = Array.from<HTMLAnchorElement>(
      fixture.nativeElement.querySelectorAll('a'),
    );
    anchors.forEach(a => {
      const href = a.getAttribute('href');
      if (href) {
        expect(KEPT_ROUTES).withContext(`unexpected href "${href}"`).toContain(href);
      }
    });
  });
});
