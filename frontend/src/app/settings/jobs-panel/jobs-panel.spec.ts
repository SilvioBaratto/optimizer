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

/** Build a multi-select change event with the given option values selected. */
function makeSelectEvent(values: string[]): Event {
  const select = document.createElement('select');
  select.multiple = true;
  for (const v of values) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.selected = true;
    select.appendChild(opt);
  }
  return { target: select } as unknown as Event;
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

  // ── Filter controls reset to page 0 and refetch (issue #986) ───────────────
  it('when setDomainFilter is called off page 0, page resets to 0 and refetch uses offset 0', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));
    component.page.set(2);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));

    component.setDomainFilter(['optimize']);
    expect(component.page()).toBe(0);
    fixture.detectChanges();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.get('offset')).toBe('0');
    expect(req.request.params.getAll('domain')).toEqual(['optimize']);
    req.flush(response());
  });

  it('when setStatusFilter is called off page 0, page resets to 0', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));
    component.page.set(2);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));

    component.setStatusFilter(['failed']);
    expect(component.page()).toBe(0);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
  });

  it('when setPageSize is called off page 0, page resets to 0 and limit updates', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));
    component.page.set(2);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));

    component.setPageSize(50);
    expect(component.page()).toBe(0);
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.get('limit')).toBe('50');
    expect(req.request.params.get('offset')).toBe('0');
    req.flush(response());
  });

  // ── Pagination bounds (issue #986) ─────────────────────────────────────────
  it('when nextPage is called below the last page, the page advances by one', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 100));
    component.nextPage();
    expect(component.page()).toBe(1);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 100));
  });

  it('when nextPage is called on the last page, the page does not change', () => {
    configure({ page: '3' }); // (3+1)*25 = 100 >= total 100 → next disabled
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 100));
    expect(component.nextDisabled()).toBe(true);
    component.nextPage();
    expect(component.page()).toBe(3);
  });

  it('when prevPage is called above page 0, the page decreases by one', () => {
    configure({ page: '2' });
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 100));
    component.prevPage();
    expect(component.page()).toBe(1);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 100));
  });

  it('when prevPage is called on page 0, the page stays at 0', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 100));
    component.prevPage();
    expect(component.page()).toBe(0);
  });

  // ── Drawer interactions (issue #986) ───────────────────────────────────────
  it('when onRowClick is called, the row raw job opens the detail request', () => {
    configure();
    const first = job({ id: 'abc' });
    http.expectOne((r) => r.url === JOBS_URL).flush(response([first], 1));

    component.onRowClick({ raw: first } as unknown as Record<string, unknown>);
    http.expectOne((r) => r.url === `${JOBS_URL}/abc`).flush(first);
    expect(component.selected()).toEqual(first);
  });

  it('when closeDrawer is called, the selected job is cleared', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.selected.set(job());
    component.closeDrawer();
    expect(component.selected()).toBeNull();
  });

  // ── Multi-select event parsing (issue #986) ────────────────────────────────
  it('when selectedOptions reads a multi-select event, all selected values are returned', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    expect(component.selectedOptions(makeSelectEvent(['a', 'b']))).toEqual(['a', 'b']);
  });

  it('when selectedStatuses reads a multi-select event, only valid statuses are kept', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    expect(component.selectedStatuses(makeSelectEvent(['running', 'bogus']))).toEqual(['running']);
  });

  // ── Row formatting branches (issue #986) ───────────────────────────────────
  it('when a job has a null duration, the duration cell shows the dash placeholder', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ duration_seconds: null })]);
    expect(component.rows()[0]['duration']).toBe('—');
  });

  it('when a job duration is under a minute, the duration cell shows seconds', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ duration_seconds: 30.5 })]);
    expect(component.rows()[0]['duration']).toBe('30.5s');
  });

  it('when a running job has a falsy total, progress falls back to zero denominator', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ status: 'running', current: 5, total: 0 })]);
    expect(component.rows()[0]['progress']).toBe('5/0');
  });

  it('when a failed job error exceeds 80 chars, the error cell is truncated with an ellipsis', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    const longError = 'x'.repeat(120);
    component.jobs.set([job({ status: 'failed', error: longError })]);
    const cell = component.rows()[0]['error'] as string;
    expect(cell.endsWith('…')).toBe(true);
    expect(cell.length).toBe(78); // 77 chars + ellipsis
  });

  it('when a failed job error is short, the error cell passes it through unchanged', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ status: 'failed', error: 'boom' })]);
    expect(component.rows()[0]['error']).toBe('boom');
  });

  it('when a job has a null started_at, the started cell shows the dash placeholder', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ started_at: null })]);
    expect(component.rows()[0]['startedAt']).toBe('—');
  });

  it('when a job domain is unknown, the domain label falls back to the raw domain', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ domain: 'totally_unknown_domain' })]);
    expect(component.rows()[0]['domainLabel']).toBe('totally_unknown_domain');
  });

  it('when jobs are present, the data table renders the rows', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([job()], 1));
    fixture.detectChanges();
    expect(component.hasRows()).toBe(true);
  });

  // ── Query-param hydration edge cases (issue #986) ──────────────────────────
  it('when the domain query param is a single string, it hydrates to a one-element array', () => {
    configure({ domain: 'yfinance_fetch' });
    expect(component.domainFilter()).toEqual(['yfinance_fetch']);
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
  });

  it('when the page query param is negative, page falls back to 0', () => {
    configure({ page: '-5' });
    expect(component.page()).toBe(0);
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
  });

  it('when the page query param is non-numeric, page falls back to 0', () => {
    configure({ page: 'abc' });
    expect(component.page()).toBe(0);
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
  });

  it('when status query param contains an invalid status value, the invalid value is filtered out', () => {
    configure({ status: ['running', 'not_a_real_status'] });
    expect(component.statusFilter()).toEqual(['running']);
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
  });

  it('when status, page, and a non-default pageSize are set, syncQueryParams emits all three', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 1000));
    const router = TestBed.inject(Router);
    const navSpy = spyOn(router, 'navigate').and.callThrough();

    component.statusFilter.set(['running']);
    component.page.set(2);
    component.pageSize.set(50);
    fixture.detectChanges();

    const [, extras] = navSpy.calls.mostRecent().args;
    expect(extras?.queryParams).toEqual(
      jasmine.objectContaining({ status: ['running'], page: 2, pageSize: 50 }),
    );
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 1000));
  });

  // ── syncQueryParams null-collapse (issue #1026) ────────────────────────────
  it('when all filters are cleared and page resets to 0, syncQueryParams writes null for domain, status, page, and pageSize', () => {
    configure({ domain: ['optimize'], page: '1' });
    http.expectOne((r) => r.url === JOBS_URL).flush(response([], 100));

    const router = TestBed.inject(Router);
    const navSpy = spyOn(router, 'navigate').and.callThrough();

    // setDomainFilter([]) resets page to 0 and clears domain
    component.setDomainFilter([]);
    fixture.detectChanges();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());

    const [, extras] = navSpy.calls.mostRecent().args;
    const qp = extras?.queryParams as Record<string, unknown>;
    expect(qp?.['domain']).toBeNull();
    expect(qp?.['status']).toBeNull();
    expect(qp?.['page']).toBeNull();
    expect(qp?.['pageSize']).toBeNull();
  });

  // ── formatDuration ≥60s branch (issue #1026) ──────────────────────────────
  it('when a job duration is exactly 60 seconds, the duration cell shows "1m 0s"', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ duration_seconds: 60 })]);
    expect(component.rows()[0]['duration']).toBe('1m 0s');
  });

  it('when a job duration is 90 seconds, the duration cell shows "1m 30s"', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ duration_seconds: 90 })]);
    expect(component.rows()[0]['duration']).toBe('1m 30s');
  });

  // ── formatError failed+null branch (issue #1026) ──────────────────────────
  it('when a failed job has no error message (null), the error cell shows the dash placeholder', () => {
    configure();
    http.expectOne((r) => r.url === JOBS_URL).flush(response());
    component.jobs.set([job({ status: 'failed', error: null })]);
    expect(component.rows()[0]['error']).toBe('—');
  });
});
