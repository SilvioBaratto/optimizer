import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';

import { PortfolioContextService } from '../services/portfolio-context.service';
import { NotificationService } from '../../shared/notification/notification.service';

export const NO_PORTFOLIO_MESSAGE = 'Please select a portfolio first';

export const portfolioRequiredGuard: CanActivateFn = () => {
  const portfolio = inject(PortfolioContextService);
  if (portfolio.hasPortfolio()) {
    return true;
  }
  inject(NotificationService).info(NO_PORTFOLIO_MESSAGE);
  return inject(Router).createUrlTree(['/portfolio-builder']);
};
