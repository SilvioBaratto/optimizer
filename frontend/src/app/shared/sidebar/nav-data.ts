export type NavIcon =
  | 'layout-dashboard'
  | 'briefcase'
  | 'pie-chart'
  | 'clock'
  | 'shield-alert'
  | 'flask-conical'
  | 'arrow-left-right'
  | 'bar-chart-3'
  | 'cpu'
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
    label: 'Portfolio',
    items: [
      { name: 'Dashboard', route: '/', icon: 'layout-dashboard' },
      { name: 'Portfolio Builder', route: '/portfolio-builder', icon: 'briefcase' },
      { name: 'Rebalancing', route: '/rebalancing', icon: 'arrow-left-right' },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { name: 'Optimization Studio', route: '/optimization-studio', icon: 'pie-chart' },
      { name: 'Backtesting Lab', route: '/backtesting', icon: 'clock' },
      { name: 'Performance Attribution', route: '/attribution', icon: 'bar-chart-3' },
    ],
  },
  {
    label: 'Risk & Research',
    items: [
      { name: 'Risk Center', route: '/risk-center', icon: 'shield-alert' },
      { name: 'Factor & Research', route: '/factor-research', icon: 'flask-conical' },
      { name: 'Macro Intelligence', route: '/macro-intelligence', icon: 'globe' },
    ],
  },
  {
    label: 'AI',
    items: [
      { name: 'AI Control Room', route: '/ai-control-room', icon: 'cpu' },
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
