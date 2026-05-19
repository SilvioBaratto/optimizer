import { Routes } from '@angular/router';

import { PORTFOLIO_BUILDER_V2 } from './features/portfolio-builder/feature-flag';
import { portfolioRequiredGuard } from './guards/portfolio-required.guard';

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
        loadComponent: PORTFOLIO_BUILDER_V2
          ? () => import('./features/portfolio-builder/portfolio-builder').then((m) => m.PortfolioBuilderComponent)
          : () => import('./pages/pipeline-stepper/pipeline-stepper').then((m) => m.PipelineStepperComponent),
        title: 'Portfolio Builder',
      },
      {
        path: 'portfolio-builder-legacy',
        loadComponent: () => import('./pages/pipeline-stepper/pipeline-stepper').then((m) => m.PipelineStepperComponent),
        title: 'Portfolio Builder (Legacy)',
      },
      {
        path: 'optimization-studio',
        loadComponent: () => import('./pages/optimization-studio/optimization-studio').then((m) => m.OptimizationStudioComponent),
        title: 'Optimization Studio',
      },
      {
        path: 'backtesting',
        loadComponent: () => import('./pages/backtesting/backtesting').then((m) => m.BacktestingComponent),
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
        loadComponent: () => import('./pages/macro-intelligence/macro-intelligence').then((m) => m.MacroIntelligenceComponent),
        title: 'Macro Intelligence',
      },
      {
        path: 'ai-control-room',
        loadComponent: () => import('./pages/ai-control-room/ai-control-room').then((m) => m.AiControlRoomComponent),
        title: 'AI Control Room',
      },
      {
        path: 'settings',
        loadComponent: () => import('./pages/settings/settings').then((m) => m.SettingsComponent),
        title: 'Settings',
      },
      {
        path: 'portfolio/:name',
        loadComponent: () => import('./pages/portfolio-detail/portfolio-detail').then((m) => m.PortfolioDetailComponent),
        title: 'Portfolio',
      },
      {
        path: 'instrument/:id',
        loadComponent: () => import('./pages/instrument-detail/instrument-detail').then((m) => m.InstrumentDetailComponent),
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
