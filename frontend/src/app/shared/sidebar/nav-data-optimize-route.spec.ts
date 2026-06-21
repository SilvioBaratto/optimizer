/**
 * Source-blind contract tests for issue #1042 — nav-data slice.
 *
 * Derived solely from the acceptance criterion:
 *   "shared/sidebar/nav-data.ts and mobile bottom-tab-bar point the
 *    Optimize item at /optimize."
 *
 * These tests MUST FAIL on the current nav-data (route '/optimization-studio')
 * and PASS once the nav item's route is flipped to '/optimize'.
 */
import type { NavItem } from './nav-data';

import { NAV_GROUPS } from './nav-data';

describe('nav-data – Optimize item route (#1042)', () => {
  /** Flatten all items across all nav groups into a single array. */
  function allItems(): NavItem[] {
    return NAV_GROUPS.flatMap((g) => g.items);
  }

  function findOptimize(): NavItem | undefined {
    return allItems().find((i) => i.name === 'Optimize');
  }

  it('when nav groups are inspected, then an item named "Optimize" exists', () => {
    expect(findOptimize()).toBeDefined();
  });

  it('when the Optimize nav item is found, then its route is "/optimize"', () => {
    const item = findOptimize();
    expect(item!.route).toBe('/optimize');
  });

  it('when the Optimize nav item is found, then its route does NOT point at "/optimization-studio"', () => {
    const item = findOptimize();
    expect(item!.route).not.toBe('/optimization-studio');
  });

  it('when all nav items are inspected, then no item routes to "/optimization-studio"', () => {
    const staleItems = allItems().filter((i) => i.route === '/optimization-studio');
    expect(staleItems.length).toBe(0);
  });
});
