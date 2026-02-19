import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./shared/layout/layout').then((m) => m.LayoutComponent),
    children: [
      {
        path: '',
        loadComponent: () => import('./pages/dashboard/dashboard').then((m) => m.DashboardComponent),
        title: 'Dashboard',
      },
      {
        path: 'universe',
        loadComponent: () => import('./pages/universe/universe').then((m) => m.UniverseComponent),
        title: 'Universe',
      },
      {
        path: 'data',
        loadComponent: () => import('./pages/data/data').then((m) => m.DataComponent),
        title: 'Data Management',
      },
      {
        path: 'optimize',
        loadComponent: () => import('./pages/optimize/optimize').then((m) => m.OptimizeComponent),
        title: 'Portfolio Optimizer',
      },
      {
        path: 'macro',
        loadComponent: () => import('./pages/macro/macro').then((m) => m.MacroComponent),
        title: 'Macro Overview',
      },
    ],
  },
  {
    path: '**',
    redirectTo: '',
  },
];
