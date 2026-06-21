import { Routes } from '@angular/router';

import { portfolioRequiredGuard } from './core/guards/portfolio-required.guard';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./shared/layout/layout').then((m) => m.LayoutComponent),
    children: [
      {
        path: '',
        loadComponent: () => import('./dashboard/dashboard').then((m) => m.DashboardComponent),
        title: 'Dashboard',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'portfolio-builder',
        loadComponent: () => import('./portfolio-builder/portfolio-builder').then((m) => m.PortfolioBuilderComponent),
        title: 'Portfolio Builder',
      },
      {
        path: 'optimize',
        loadComponent: () => import('./optimization-studio/optimization-studio').then((m) => m.OptimizationStudioComponent),
        title: 'Optimize',
      },
      {
        path: 'backtesting',
        loadComponent: () => import('./backtesting/backtesting').then((m) => m.BacktestingComponent),
        title: 'Backtesting',
        canActivate: [portfolioRequiredGuard],
      },
      {
        path: 'macro-intelligence',
        loadComponent: () => import('./macro-intelligence/macro-intelligence').then((m) => m.MacroIntelligenceComponent),
        title: 'Macro Intelligence',
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
        path: 'dashboard',
        redirectTo: '',
        pathMatch: 'full',
      },
      {
        path: 'optimization-studio',
        redirectTo: 'optimize',
      },
      {
        path: '**',
        loadComponent: () => import('./shared/not-found/page-not-found').then((m) => m.PageNotFoundComponent),
        title: 'Page Not Found',
      },
    ],
  },
];
