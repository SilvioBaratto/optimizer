import {
  ApplicationConfig,
  provideBrowserGlobalErrorListeners,
  provideZonelessChangeDetection,
  APP_INITIALIZER,
} from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';

import { routes } from './app.routes';
import { registerPortfolioTheme } from './shared/charts/echarts-theme';
import { ICON_PROVIDER } from './icons';
import { LucideIconConfig } from 'lucide-angular';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection(),
    provideRouter(routes),
    provideHttpClient(),
    provideAnimations(),
    ICON_PROVIDER,
    {
      provide: LucideIconConfig,
      useFactory: () => {
        const cfg = new LucideIconConfig();
        cfg.size = 16;
        cfg.strokeWidth = 1.5;
        return cfg;
      },
    },
    {
      provide: APP_INITIALIZER,
      useFactory: () => () => registerPortfolioTheme(),
      multi: true,
    },
  ],
};
