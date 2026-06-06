import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal } from '@angular/core';
import {
  ActivatedRouteSnapshot,
  Router,
  RouterStateSnapshot,
  UrlTree,
} from '@angular/router';

import {
  portfolioRequiredGuard,
  NO_PORTFOLIO_MESSAGE,
} from './portfolio-required.guard';
import { PortfolioContextService } from '../services/portfolio-context.service';
import { NotificationService } from '../../shared/notification/notification.service';

describe('portfolioRequiredGuard', () => {
  let portfolioId: ReturnType<typeof signal<string | null>>;
  let router: jasmine.SpyObj<Router>;
  let notifications: jasmine.SpyObj<NotificationService>;

  const route = {} as ActivatedRouteSnapshot;
  const state = { url: '/' } as RouterStateSnapshot;

  beforeEach(() => {
    portfolioId = signal<string | null>(null);
    router = jasmine.createSpyObj<Router>('Router', ['createUrlTree']);
    router.createUrlTree.and.callFake(
      (commands) => ({ commands } as unknown as UrlTree),
    );
    notifications = jasmine.createSpyObj<NotificationService>(
      'NotificationService',
      ['success', 'error', 'warning', 'info'],
    );

    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        {
          provide: PortfolioContextService,
          useValue: {
            currentPortfolioId: portfolioId,
            hasPortfolio: () => portfolioId() !== null,
          },
        },
        { provide: Router, useValue: router },
        { provide: NotificationService, useValue: notifications },
      ],
    });
  });

  function runGuard(): boolean | UrlTree {
    return TestBed.runInInjectionContext(
      () =>
        portfolioRequiredGuard(route, state) as boolean | UrlTree,
    );
  }

  it('allows navigation when a portfolio is selected', () => {
    portfolioId.set('my-portfolio');

    const result = runGuard();

    expect(result).toBe(true);
    expect(router.createUrlTree).not.toHaveBeenCalled();
    expect(notifications.info).not.toHaveBeenCalled();
  });

  it('redirects to /portfolio-builder when no portfolio is selected', () => {
    portfolioId.set(null);

    const result = runGuard();

    expect(router.createUrlTree).toHaveBeenCalledWith(['/portfolio-builder']);
    expect(result).not.toBe(true);
  });

  it('dispatches info toast when no portfolio is selected', () => {
    portfolioId.set(null);

    runGuard();

    expect(notifications.info).toHaveBeenCalledWith(NO_PORTFOLIO_MESSAGE);
  });
});
