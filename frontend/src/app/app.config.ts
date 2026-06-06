import {
  ApplicationConfig,
  ErrorHandler,
  provideBrowserGlobalErrorListeners,
  provideZonelessChangeDetection,
  APP_INITIALIZER,
} from '@angular/core';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
import { firstValueFrom } from 'rxjs';

import { routes } from './app.routes';
import { apiHttpInterceptor } from './core/interceptors/api-http.interceptor';
import { GlobalErrorHandler } from './core/error-handling/global-error-handler';
import { registerPortfolioTheme } from './shared/charts/echarts-theme';
import { ICON_PROVIDER } from './icons';
import { LucideIconConfig } from 'lucide-angular';
import { PortfolioApiService } from './core/services/portfolio-api.service';
import { PortfolioContextService } from './core/services/portfolio-context.service';

/**
 * Resolve the initial portfolio selection before the app renders.
 *
 * Order of precedence:
 *   1. localStorage value (already applied synchronously in the service
 *      constructor) — validated against the server.
 *   2. The first active portfolio returned by the API.
 *   3. Leave `currentPortfolioId` as null if the server returns zero
 *      portfolios; the dashboard renders an empty state in that case.
 *
 * Any API error is logged and swallowed so the app boots regardless.
 */
function bootstrapPortfolioSelection(
  api: PortfolioApiService,
  ctx: PortfolioContextService,
): () => Promise<void> {
  return async () => {
    try {
      const response = await firstValueFrom(api.list());
      const active = response.items.filter((p) => p.is_active);
      if (active.length === 0) {
        ctx.setPortfolio(null);
        return;
      }

      const stored = ctx.currentPortfolioId();
      const storedStillExists =
        stored !== null && active.some((p) => p.name === stored);
      if (!storedStillExists) {
        ctx.setPortfolio(active[0].name);
      }
    } catch (err) {
      console.warn(
        '[portfolio bootstrap] failed to load portfolio list:',
        err,
      );
    }
  };
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection(),
    provideRouter(routes, withComponentInputBinding()),
    provideHttpClient(withInterceptors([apiHttpInterceptor])),
    provideAnimations(),
    { provide: ErrorHandler, useClass: GlobalErrorHandler },
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
    {
      provide: APP_INITIALIZER,
      useFactory: bootstrapPortfolioSelection,
      deps: [PortfolioApiService, PortfolioContextService],
      multi: true,
    },
  ],
};
