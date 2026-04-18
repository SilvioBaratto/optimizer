import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { DashboardService } from './dashboard.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

describe('DashboardService', () => {
  let svc: DashboardService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        DashboardService,
      ],
    });
    svc = TestBed.inject(DashboardService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('getPerformanceMetrics(name, period)', () => {
    it('encodes the period as a query parameter (issue #433)', () => {
      svc.getPerformanceMetrics('myport', '3Y').subscribe();

      const req = http.expectOne(
        (r) =>
          r.url ===
          `${API}portfolio-analytics/myport/performance-metrics`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('period')).toBe('3Y');
      req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
    });

    it('defaults the period to 1Y when omitted', () => {
      svc.getPerformanceMetrics('myport').subscribe();

      const req = http.expectOne(
        (r) =>
          r.url ===
          `${API}portfolio-analytics/myport/performance-metrics`,
      );
      expect(req.request.params.get('period')).toBe('1Y');
      req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
    });

    it('round-trips every supported period', () => {
      for (const period of ['1Y', '3Y', '5Y', 'MAX'] as const) {
        svc.getPerformanceMetrics('myport', period).subscribe();
        const req = http.expectOne(
          (r) =>
            r.url ===
              `${API}portfolio-analytics/myport/performance-metrics` &&
            r.params.get('period') === period,
        );
        expect(req.request.params.get('period')).toBe(period);
        req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
      }
    });
  });
});
