import { Routes } from '@angular/router';

import { portfolioRequiredGuard } from './core/guards/portfolio-required.guard';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./shared/layout/layout').then((m) => m.LayoutComponent),
    children: [
      {
        path: '',
        loadComponent: () => import('./pages/dashboard/dashboard').then((m) => m.DashboardComponent),
        title: 'Dashboard',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'portfolio-builder',
        loadComponent: () => import('./pages/portfolio-builder/portfolio-builder').then((m) => m.PortfolioBuilderComponent),
        title: 'Portfolio Builder',
      },
      {
        path: 'optimization-studio',
        loadComponent: () => import('./pages/optimization-studio/optimization-studio').then((m) => m.OptimizationStudioComponent),
        title: 'Optimization Studio',
      },
      {
        path: 'backtesting',
        loadComponent: () => import('./backtesting/backtesting').then((m) => m.BacktestingComponent),
        title: 'Backtesting',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'risk-center',
        loadComponent: () => import('./pages/risk-center/risk-center').then((m) => m.RiskCenterComponent),
        title: 'Risk Center',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'factor-research',
        loadComponent: () => import('./pages/factor-research/factor-research').then((m) => m.FactorResearchComponent),
        title: 'Factor Research',
      },
      {
        path: 'rebalancing',
        loadComponent: () => import('./pages/rebalancing/rebalancing').then((m) => m.RebalancingComponent),
        title: 'Rebalancing',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'attribution',
        loadComponent: () => import('./pages/attribution/attribution').then((m) => m.AttributionComponent),
        title: 'Attribution',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'macro-intelligence',
        loadComponent: () => import('./macro-intelligence/macro-intelligence').then((m) => m.MacroIntelligenceComponent),
        title: 'Macro Intelligence',
      },
      {
        path: 'ai-control-room',
        loadComponent: () => import('./pages/ai-control-room/ai-control-room').then((m) => m.AiControlRoomComponent),
        title: 'AI Control Room',
      },
      {
        path: 'settings',
        loadComponent: () => import('./settings/settings').then((m) => m.SettingsComponent),
        title: 'Settings',
      },
      {
        path: 'portfolio/:name',
        loadComponent: () => import('./portfolio-detail/portfolio-detail').then((m) => m.PortfolioDetailComponent),
        title: 'Portfolio',
      },
      {
        path: 'instrument/:id',
        loadComponent: () => import('./instrument-detail/instrument-detail').then((m) => m.InstrumentDetailComponent),
        title: 'Instrument',
      },
      {
        path: 'optimize',
        redirectTo: 'optimization-studio',
      },
      {
        path: '**',
        loadComponent: () => import('./shared/not-found/page-not-found').then((m) => m.PageNotFoundComponent),
        title: 'Page Not Found',
      },
    ],
  },
];
