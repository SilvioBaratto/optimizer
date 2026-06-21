export type NavIcon =
  | 'layout-dashboard'
  | 'briefcase'
  | 'pie-chart'
  | 'clock'
  | 'globe'
  | 'settings';

export interface NavItem {
  name: string;
  route: string;
  icon: NavIcon;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
  pinBottom?: boolean;
}

export const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Core',
    items: [
      { name: 'Dashboard',          route: '/',                   icon: 'layout-dashboard' },
      { name: 'Portfolio Builder',  route: '/portfolio-builder',  icon: 'briefcase' },
      { name: 'Optimize',           route: '/optimize',            icon: 'pie-chart' },
      { name: 'Backtesting',        route: '/backtesting',        icon: 'clock' },
      { name: 'Macro Intelligence', route: '/macro-intelligence', icon: 'globe' },
    ],
  },
  {
    label: 'Settings',
    items: [
      { name: 'Settings', route: '/settings', icon: 'settings' },
    ],
    pinBottom: true,
  },
];
