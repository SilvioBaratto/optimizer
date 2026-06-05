import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PipelineStatusComponent } from './pipeline-status';
import { environment } from '../../../../environments/environment';
import { ICON_PROVIDER } from '../../../icons';

const JOBS_URL = `${environment.apiUrl}jobs`;

describe('PipelineStatusComponent — state transitions (issue #440)', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('starts in the loading state before the first response', () => {
    const fx = TestBed.createComponent(PipelineStatusComponent);
    fx.detectChanges();
    expect(fx.componentInstance.isLoading()).toBe(true);
    // Drain the bootstrap request so afterEach http.verify() passes
    http.expectOne((r) => r.url === JOBS_URL).flush({
      jobs: [], total: 0, limit: 100, offset: 0,
    });
  });

  it('transitions to the loaded state with one card per real job (issue #440)', () => {
    const fx = TestBed.createComponent(PipelineStatusComponent);
    fx.detectChanges();

    http.expectOne((r) => r.url === JOBS_URL).flush({
      jobs: [
        {
          id: 'a', domain: 'optimize', status: 'completed',
          current: 100, total: 100, error: null, errors_count: 0,
          started_at: '2026-04-17T00:00:00Z',
          finished_at: '2026-04-17T00:01:00Z', duration_seconds: 60,
        },
      ],
      total: 1, limit: 100, offset: 0,
    });

    expect(fx.componentInstance.isLoading()).toBe(false);
    expect(fx.componentInstance.domainStatuses().length).toBe(1);
    expect(fx.componentInstance.domainStatuses()[0].meta.domain).toBe('optimize');
    expect(fx.componentInstance.pollError()).toBeNull();
  });

  it('renders an empty state when the API returns no jobs (issue #440)', () => {
    const fx = TestBed.createComponent(PipelineStatusComponent);
    fx.detectChanges();

    http.expectOne((r) => r.url === JOBS_URL).flush({
      jobs: [], total: 0, limit: 100, offset: 0,
    });

    expect(fx.componentInstance.isLoading()).toBe(false);
    expect(fx.componentInstance.domainStatuses()).toEqual([]);
    expect(fx.componentInstance.hasNoData()).toBe(true);
    expect(fx.componentInstance.pollError()).toBeNull();
  });

  it('transitions to the error state when the API call fails (issue #440)', () => {
    const fx = TestBed.createComponent(PipelineStatusComponent);
    fx.detectChanges();

    http
      .expectOne((r) => r.url === JOBS_URL)
      .error(new ProgressEvent('network'), { status: 500 });

    expect(fx.componentInstance.isLoading()).toBe(false);
    expect(fx.componentInstance.pollError()).toBeTruthy();
    expect(fx.componentInstance.domainStatuses()).toEqual([]);
  });

  it('polling survives a transient 5xx: pollError is set, then a later poll recovers (issue #851)', () => {
    jasmine.clock().install();
    try {
      const fx = TestBed.createComponent(PipelineStatusComponent);
      fx.detectChanges();
      const ok = { jobs: [], total: 0, limit: 100, offset: 0 };

      // Initial load succeeds.
      http.expectOne((r) => r.url === JOBS_URL).flush(ok);

      // First poll tick errors with a transient 5xx → catchError keeps it alive.
      jasmine.clock().tick(30_000);
      http
        .expectOne((r) => r.url === JOBS_URL)
        .error(new ProgressEvent('network'), { status: 503 });
      expect(fx.componentInstance.pollError()).toBeTruthy();

      // The interval survived: a later tick fires another poll that recovers.
      jasmine.clock().tick(30_000);
      http.expectOne((r) => r.url === JOBS_URL).flush(ok);
      expect(fx.componentInstance.pollError()).toBeNull();

      fx.destroy(); // tear down the periodic timer
    } finally {
      jasmine.clock().uninstall();
    }
  });

  it('exposes hasRunning and hasFailures from the loaded jobs', () => {
    const fx = TestBed.createComponent(PipelineStatusComponent);
    fx.detectChanges();

    http.expectOne((r) => r.url === JOBS_URL).flush({
      jobs: [
        {
          id: 'r1', domain: 'optimize', status: 'running',
          current: 5, total: 10, error: null, errors_count: 0,
          started_at: '2026-04-18T00:00:00Z', finished_at: null, duration_seconds: null,
        },
        {
          id: 'f1', domain: 'macro_fetch', status: 'failed',
          current: 0, total: 1, error: 'boom', errors_count: 1,
          started_at: '2026-04-18T00:00:00Z',
          finished_at: '2026-04-18T00:00:30Z', duration_seconds: 30,
        },
      ],
      total: 2, limit: 100, offset: 0,
    });

    expect(fx.componentInstance.hasRunning()).toBe(true);
    expect(fx.componentInstance.runningJobs().length).toBe(1);
    expect(fx.componentInstance.hasFailures()).toBe(true);
    expect(fx.componentInstance.failedJobs().length).toBe(1);
  });
});
