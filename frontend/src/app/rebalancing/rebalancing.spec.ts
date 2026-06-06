import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import {
  apiHttpInterceptor,
  RETRY_BACKOFF,
  SUPPRESS_TOAST_STATUSES,
  type BackoffFn,
} from '../core/interceptors/api-http.interceptor';
import { NotificationService } from '../shared/notification/notification.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const PORTFOLIO = 'trading212';

describe('RebalancingComponent — preview error handling (issue #438)', () => {
  let http: HttpTestingController;
  let notifications: jasmine.SpyObj<NotificationService>;

  beforeEach(() => {
    notifications = jasmine.createSpyObj<NotificationService>(
      'NotificationService',
      ['success', 'error', 'warning', 'info'],
    );
    const immediateBackoff: BackoffFn = () => of(0);

    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(withInterceptors([apiHttpInterceptor])),
        provideHttpClientTesting(),
        { provide: NotificationService, useValue: notifications },
        { provide: RETRY_BACKOFF, useValue: immediateBackoff },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function bootstrapWithSelectedPortfolio() {
    const fx = TestBed.createComponent(RebalancingComponent);
    fx.detectChanges();
    // Drain the portfolios bootstrap call (URL may carry a trailing slash)
    http
      .expectOne((r) => r.url.startsWith(`${API}portfolio`) && !r.url.includes('/snapshot') && !r.url.includes('/rebalance'))
      .flush({ items: [] });
    fx.componentInstance.onPortfolioSelect(PORTFOLIO);
    fx.detectChanges();
    return fx;
  }

  function flushNonPreviewRequests(): void {
    // Selecting a portfolio fires drift, policies, snapshots, preview.
    // We only care about the preview branch, so flush the others as success.
    const matches = http.match(
      (r) => !r.url.includes('/rebalance/preview/'),
    );
    for (const req of matches) {
      const url = req.request.url;
      if (url.includes('/drift')) req.flush({ entries: [], breachedCount: 0 });
      else if (url.includes('/rebalance-policy')) req.flush({ items: [] });
      else if (url.includes('/snapshots')) req.flush({ items: [] });
      else req.flush({});
    }
  }

  it('renders a friendly panel message on a 404 from /rebalance/preview AND suppresses the toast', () => {
    const fx = bootstrapWithSelectedPortfolio();
    flushNonPreviewRequests();

    const previewReq = http.expectOne(
      (r) => r.url === `${API}rebalance/preview/${PORTFOLIO}`,
    );
    previewReq.flush(
      { detail: `Portfolio '${PORTFOLIO}' not found` },
      { status: 404, statusText: 'Not Found' },
    );

    const message = fx.componentInstance.panelErrors()['preview'];
    expect(message).toBeTruthy();
    expect(message).not.toContain('Http failure response');
    expect(message?.toLowerCase()).toContain('preview');
    expect(notifications.error).not.toHaveBeenCalled();
  });

  it('renders a friendly panel message on a non-404 error from /rebalance/preview', () => {
    // 422 isn't retried by the interceptor and isn't suppressed by SUPPRESS_TOAST_STATUSES,
    // so it exercises the same code path as a 500 (toast + friendly panel) without
    // needing to flush 4 retry attempts.
    const fx = bootstrapWithSelectedPortfolio();
    flushNonPreviewRequests();

    const previewReq = http.expectOne(
      (r) => r.url === `${API}rebalance/preview/${PORTFOLIO}`,
    );
    previewReq.flush(
      { detail: 'Snapshot mismatch' },
      { status: 422, statusText: 'Unprocessable Entity' },
    );

    const message = fx.componentInstance.panelErrors()['preview'];
    expect(message).toBeTruthy();
    expect(message).not.toContain('Http failure response');
    expect(message).toContain('Snapshot mismatch');
  });

  it('opts the preview request out of the 404 toast via HttpContext', () => {
    const fx = bootstrapWithSelectedPortfolio();
    flushNonPreviewRequests();

    const previewReq = http.expectOne(
      (r) => r.url === `${API}rebalance/preview/${PORTFOLIO}`,
    );
    // The request must carry the suppress-toast context so the interceptor
    // skips notify.error() for 404. We assert the request context contains
    // 404 (the contract enforced by the rebalancing service).
    const suppressed = previewReq.request.context.get(SUPPRESS_TOAST_STATUSES);
    expect(suppressed).toContain(404);

    previewReq.flush(
      { detail: 'gone' },
      { status: 404, statusText: 'Not Found' },
    );
    void fx;
  });
});
