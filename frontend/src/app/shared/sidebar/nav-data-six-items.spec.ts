/**
 * Issue #1036 — NAV_GROUPS trimmed to exactly six items in flow order.
 * Source-blind: derived from acceptance criteria only.
 */
import { NAV_GROUPS } from './nav-data';

describe('NAV_GROUPS — trimmed to six items (#1036)', () => {
  const allItems = NAV_GROUPS.flatMap(g => g.items);

  const EXPECTED_FLOW: { name: string; route: string }[] = [
    { name: 'Dashboard',           route: '/' },
    { name: 'Portfolio Builder',   route: '/portfolio-builder' },
    { name: 'Optimize',            route: '/optimization-studio' },
    { name: 'Backtesting',         route: '/backtesting' },
    { name: 'Macro Intelligence',  route: '/macro-intelligence' },
    { name: 'Settings',            route: '/settings' },
  ];

  const DELETED_ROUTES = [
    '/risk-center',
    '/ai-control-room',
    '/rebalancing',
    '/attribution',
    '/factor-research',
  ];

  it('when NAV_GROUPS is flattened, exactly six items are returned', () => {
    expect(allItems.length).toBe(6);
  });

  EXPECTED_FLOW.forEach(({ name, route }, index) => {
    it(`when item at position ${index + 1} is read, its route is "${route}"`, () => {
      expect(allItems[index]?.route).toBe(route);
    });

    it(`when item at position ${index + 1} is read, its name is "${name}"`, () => {
      expect(allItems[index]?.name).toBe(name);
    });
  });

  it('when the Settings group is located by route, pinBottom is true', () => {
    const settingsGroup = NAV_GROUPS.find(g =>
      g.items.some(item => item.route === '/settings'),
    );
    expect(settingsGroup?.pinBottom).toBeTrue();
  });

  DELETED_ROUTES.forEach(route => {
    it(`when all routes are collected, deleted route "${route}" is absent`, () => {
      expect(allItems.map(i => i.route)).not.toContain(route);
    });
  });
});
