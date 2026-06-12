import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { ActivatedRoute, Router, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { JobsPanelComponent } from './jobs-panel';
import { ICON_PROVIDER } from '../../icons';
import { environment } from '../../../environments/environment';
import type { JobListResponse, JobSummary } from '../../core/models/jobs.model';

const JOBS_URL = `${environment.apiUrl}jobs`;

function job(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id: 'job-1',
    domain: 'yfinance_fetch',
    status: 'completed',
    current: 100,
    total: 100,
    error: null,
    errors_count: 0,
    started_at: '2026-04-17T00:00:00Z',
    finished_at: '2026-04-17T00:05:00Z',
    duration_seconds: 300,
    ...overrides,
  };
}

function response(
  jobs: JobSummary[] = [],
  total = 0,
  limit = 25,
  offset = 0,
): JobListResponse {
  return { jobs, total, limit, offset };
}

describe('JobsPanelComponent', () => {
  let fixture: ComponentFixture<JobsPanelComponent>;
  let component: JobsPanelComponent;
  let http: HttpTestingController;

  function configure(queryParams: Record<string, unknown> = {}): void {
    TestBed.configureTestingModule({
      imports: [JobsPanelComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        {
          provide: ActivatedRoute,
          useValue: { queryParams: of(queryParams) },
        },
      ],
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(JobsPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  afterEach(() => http.verify());

  it('fires an initial GET /jobs with default limit=25, offset=0 and no filters', () => {
    configure();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.method).toBe('GET');
    expect(req.request.params.get('limit')).toBe('25');
    expect(req.request.params.get('offset')).toBe('0');
    expect(req.request.params.has('domain')).toBe(false);
    expect(req.request.params.has('status')).toBe(false);
    req.flush(response());
  });

  it('changing domainFilter triggers one new GET /jobs with repeated domain params', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());

    component.domainFilter.set(['yfinance_fetch', 'optimize']);
    fixture.detectChanges();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.getAll('domain')).toEqual(['yfinance_fetch', 'optimize']);
    req.flush(response());
  });

  it('changing statusFilter triggers one new GET /jobs with repeated status params', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());

    component.statusFilter.set(['running', 'failed']);
    fixture.detectChanges();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.getAll('status')).toEqual(['running', 'failed']);
    req.flush(response());
  });

  it('changing page triggers a new request with updated offset', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));

    component.page.set(2);
    fixture.detectChanges();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.get('offset')).toBe('50');
    expect(req.request.params.get('limit')).toBe('25');
    req.flush(response([], 100));
  });

  it('changing pageSize triggers a new request with updated limit (offset still 0)', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());

    component.pageSize.set(50);
    fixture.detectChanges();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.get('limit')).toBe('50');
    req.flush(response());
  });

  it('hydrates signals from ActivatedRoute.queryParams on init', () => {
    configure({
      domain: ['yfinance_fetch', 'optimize'],
      status: ['running'],
      page: '3',
      pageSize: '50',
    });

    expect(component.domainFilter()).toEqual(['yfinance_fetch', 'optimize']);
    expect(component.statusFilter()).toEqual(['running']);
    expect(component.page()).toBe(3);
    expect(component.pageSize()).toBe(50);

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.get('offset')).toBe('150');
    expect(req.request.params.get('limit')).toBe('50');
    req.flush(response([], 1000));
  });

  it('writes queryParams back via router.navigate when a signal changes', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());

    const router = TestBed.inject(Router);
    const navSpy = spyOn(router, 'navigate').and.callThrough();

    component.domainFilter.set(['optimize']);
    fixture.detectChanges();

    expect(navSpy).toHaveBeenCalled();
    const [, extras] = navSpy.calls.mostRecent().args;
    expect(extras?.queryParamsHandling).toBe('merge');
    expect(extras?.replaceUrl).toBe(true);
    expect(extras?.queryParams).toEqual(
      jasmine.objectContaining({ domain: ['optimize'] }),
    );
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
  });

  it('populates jobs and total signals from the response', () => {
    configure();
    const jobs = [job({ id: 'a' }), job({ id: 'b', status: 'running' })];
    http.expectOne((r) => r.url === JOBS_URL).flush(response(jobs, 42));

    expect(component.jobs()).toEqual(jobs);
    expect(component.total()).toBe(42);
  });

  it('disables Prev on page 0 and Next when (page+1)*pageSize >= total', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 25));
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    const prev = root.querySelector<HTMLButtonElement>('[data-testid="page-prev"]');
    const next = root.querySelector<HTMLButtonElement>('[data-testid="page-next"]');
    expect(prev!.disabled).toBe(true); // page=0
    expect(next!.disabled).toBe(true); // 1*25 >= 25
  });

  it('row click invokes getJob(id) and sets the selected job', () => {
    configure();
    const first = job({ id: 'abc' });
    http.expectOne((r) => r.url === JOBS_URL).flush(response([first], 1));

    component.openJob(first);
    const detailReq = http.expectOne((r) => r.url === `${JOBS_URL}/abc`);
    expect(detailReq.request.method).toBe('GET');
    detailReq.flush(first);
    fixture.detectChanges();

    expect(component.selected()).toEqual(first);
  });

  it('renders <app-job-progress-tracker> inside the drawer when selected job is running', () => {
    configure();
    const running = job({ id: 'abc', status: 'running' });
    http.expectOne((r) => r.url === JOBS_URL).flush(response([running], 1));

    component.openJob(running);
    http.expectOne((r) => r.url === `${JOBS_URL}/abc`).flush(running);
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelectorAll('app-job-progress-tracker').length).toBe(1);
  });

  it('renders an inline error banner on list-fetch failure and sets error signal', () => {
    configure();
    http
      .expectOne((r) => r.url === JOBS_URL)
      .flush({ detail: 'boom' }, { status: 500, statusText: 'err' });
    fixture.detectChanges();

    expect(component.error()).toContain('boom');
    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('[role="alert"]')).not.toBeNull();
  });

  it('shows an empty-state message when total === 0 after the first successful fetch', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 0));
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('[data-testid="jobs-empty"]')).not.toBeNull();
  });

  it('clicking the retry button in the error banner re-fires GET /jobs and clears the error', () => {
    configure();
    http
      .expectOne((r) => r.url === JOBS_URL)
      .flush({ detail: 'boom' }, { status: 500, statusText: 'err' });
    fixture.detectChanges();

    expect(component.error()).toBeTruthy();
    const root = fixture.nativeElement as HTMLElement;
    const retryBtn = root.querySelector<HTMLButtonElement>('[role="alert"] button');
    retryBtn!.click();
    fixture.detectChanges();

    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    fixture.detectChanges();

    expect(component.error()).toBeNull();
  });

  it('when getJob fails, drawerError signal is set', () => {
    configure();
    const first = job({ id: 'abc' });
    http.expectOne((r) => r.url === JOBS_URL).flush(response([first], 1));
    fixture.detectChanges();

    component.openJob(first);
    http
      .expectOne((r) => r.url === `${JOBS_URL}/abc`)
      .flush({ detail: 'fetch failed' }, { status: 500, statusText: 'err' });
    fixture.detectChanges();

    expect(component.drawerError()).toBeTruthy();
    expect(component.selected()).toEqual(first); // summary preserved on error
  });
});
