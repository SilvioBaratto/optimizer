export type NavIcon =
  | 'dashboard'
  | 'briefcase'
  | 'chart-pie'
  | 'clock'
  | 'shield-exclamation'
  | 'beaker'
  | 'arrows-right-left'
  | 'chart-bar-square'
  | 'cpu-chip'
  | 'globe-alt'
  | 'cog-6-tooth';

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
      { name: 'Dashboard', route: '/', icon: 'dashboard' },
      { name: 'Portfolio Builder', route: '/portfolio-builder', icon: 'briefcase' },
      { name: 'Rebalancing', route: '/rebalancing', icon: 'arrows-right-left' },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { name: 'Optimization Studio', route: '/optimization-studio', icon: 'chart-pie' },
      { name: 'Backtesting Lab', route: '/backtesting', icon: 'clock' },
      { name: 'Performance Attribution', route: '/attribution', icon: 'chart-bar-square' },
    ],
  },
  {
    label: 'Risk & Research',
    items: [
      { name: 'Risk Center', route: '/risk-center', icon: 'shield-exclamation' },
      { name: 'Factor & Research', route: '/factor-research', icon: 'beaker' },
      { name: 'Macro Intelligence', route: '/macro-intelligence', icon: 'globe-alt' },
    ],
  },
  {
    label: 'AI',
    items: [
      { name: 'AI Control Room', route: '/ai-control-room', icon: 'cpu-chip' },
    ],
  },
  {
    label: 'Settings',
    items: [
      { name: 'Settings', route: '/settings', icon: 'cog-6-tooth' },
    ],
    pinBottom: true,
  },
];
