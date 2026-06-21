import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { SchedulerStatusPanelComponent } from './scheduler-status-panel';
import { environment } from '../../../environments/environment';

const API = environment.apiUrl;

describe('SchedulerStatusPanelComponent', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('renders jobs when scheduler is running', () => {
    const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
    fx.detectChanges();

    http.expectOne(`${API}scheduler/status`).flush({
      schedulerRunning: true,
      jobs: [
        {
          jobId: 'daily_pipeline',
          name: 'Daily pipeline',
          nextRunTime: '2026-04-18T07:00:00Z',
          lastRunTime: '2026-04-17T07:00:00Z',
          lastStatus: 'completed',
          trigger: 'cron[hour=7]',
        },
      ],
    });

    expect(fx.componentInstance.schedulerRunning()).toBe(true);
    expect(fx.componentInstance.jobs().length).toBe(1);
    expect(fx.componentInstance.isEmpty()).toBe(false);
  });

  it('renders each scheduled job exactly once — not twice (issue #437)', () => {
    const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
    fx.detectChanges();

    const jobs = [
      { jobId: 'a', name: 'Daily pipeline', nextRunTime: '2026-04-18T07:00:00Z',
        lastRunTime: '2026-04-17T07:00:00Z', lastStatus: 'completed', trigger: 'cron[hour=7]' },
      { jobId: 'b', name: 'Weekly refetch', nextRunTime: null,
        lastRunTime: null, lastStatus: null, trigger: 'cron[day_of_week=sun]' },
      { jobId: 'c', name: 'Midday news', nextRunTime: '2026-04-18T14:00:00Z',
        lastRunTime: '2026-04-17T14:00:00Z', lastStatus: 'completed', trigger: 'cron[hour=14]' },
    ];
    http.expectOne(`${API}scheduler/status`).flush({ schedulerRunning: true, jobs });
    fx.detectChanges();

    const host: HTMLElement = fx.nativeElement;
    // Count rendered <tr> rows that carry a job name in the data-table body.
    // A leftover `<ul>`/`<li>` duplicate would push this count past jobs.length.
    const renderedNameCells = host.querySelectorAll('tbody tr');
    expect(renderedNameCells.length).toBe(jobs.length);

    // Belt-and-braces: ensure no orphan <ul> with per-job <li> survived.
    const orphanList = host.querySelectorAll('ul li');
    expect(orphanList.length)
      .withContext('the duplicated <ul> rendering must be removed')
      .toBe(0);
  });

  it('formats nextRunTime / lastRunTime in the data-table cells (issue #437)', () => {
    const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
    fx.detectChanges();

    http.expectOne(`${API}scheduler/status`).flush({
      schedulerRunning: true,
      jobs: [{
        jobId: 'a', name: 'Daily pipeline', nextRunTime: '2026-04-18T07:00:00Z',
        lastRunTime: '2026-04-17T07:00:00Z', lastStatus: 'completed', trigger: 'cron[hour=7]',
      }],
    });

    // The chosen presentation must keep the timestamp columns visible.
    const labels = fx.componentInstance.tableColumns.map((c) => c.label);
    expect(labels).toContain('Next run');
    expect(labels).toContain('Last run');

    const row = fx.componentInstance.tableRows()[0];
    expect(row['next_run_time']).toBe('2026-04-18T07:00:00Z');
    expect(row['last_run_time']).toBe('2026-04-17T07:00:00Z');
  });

  it('shows the empty state when scheduler is not running', () => {
    const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
    fx.detectChanges();

    http.expectOne(`${API}scheduler/status`).flush({
      schedulerRunning: false,
      jobs: [],
    });

    expect(fx.componentInstance.schedulerRunning()).toBe(false);
    expect(fx.componentInstance.isEmpty()).toBe(true);
  });

  it('shows the empty state when scheduler is running with zero jobs', () => {
    const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
    fx.detectChanges();

    http.expectOne(`${API}scheduler/status`).flush({
      schedulerRunning: true,
      jobs: [],
    });

    expect(fx.componentInstance.isEmpty()).toBe(true);
  });

  it('surfaces the error when the scheduler endpoint fails', () => {
    const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
    fx.detectChanges();

    http
      .expectOne(`${API}scheduler/status`)
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });

    expect(fx.componentInstance.loadError()).toBeTruthy();
    expect(fx.componentInstance.loading()).toBe(false);
  });

  describe('error banner (issue #964)', () => {
    it('when scheduler endpoint returns 500, loadError is set and app-page-error-banner is in the DOM', () => {
      const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
      fx.detectChanges();
      http
        .expectOne(`${API}scheduler/status`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeTruthy();
      expect(fx.nativeElement.querySelector('app-page-error-banner')).not.toBeNull();
    });

    it('clicking the retry button in the error banner re-fires GET /scheduler/status', () => {
      const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
      fx.detectChanges();
      http
        .expectOne(`${API}scheduler/status`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      fx.detectChanges();

      const banner: HTMLElement = fx.nativeElement.querySelector('app-page-error-banner');
      banner.querySelector<HTMLButtonElement>('button')!.click();
      fx.detectChanges();

      http
        .expectOne(`${API}scheduler/status`)
        .flush({ schedulerRunning: true, jobs: [] });
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeNull();
      expect(fx.componentInstance.schedulerRunning()).toBe(true);
    });
  });

  // ── formatTimestamp branches via column format helpers (issue #1026) ───────
  describe('formatTimestamp — column format callbacks', () => {
    const DASH = '—';

    function formatFn(
      comp: SchedulerStatusPanelComponent,
      columnKey: string,
    ): (v: unknown) => string {
      const col = comp.tableColumns.find((c) => c.key === columnKey);
      return col!.format as (v: unknown) => string;
    }

    function createAndFlush() {
      const fx = TestBed.createComponent(SchedulerStatusPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}scheduler/status`).flush({ schedulerRunning: true, jobs: [] });
      return fx.componentInstance;
    }

    it('when next_run_time column format is called with null, it returns the dash placeholder', () => {
      const comp = createAndFlush();
      expect(formatFn(comp, 'next_run_time')(null)).toBe(DASH);
    });

    it('when next_run_time column format is called with empty string, it returns the dash placeholder', () => {
      const comp = createAndFlush();
      expect(formatFn(comp, 'next_run_time')('')).toBe(DASH);
    });

    it('when next_run_time column format is called with a valid ISO string, it returns a non-empty date', () => {
      const comp = createAndFlush();
      const result = formatFn(comp, 'next_run_time')('2026-04-18T07:00:00Z');
      expect(result).not.toBe(DASH);
      expect(result.length).toBeGreaterThan(0);
    });

    it('when last_run_time column format is called with null, it returns the dash placeholder', () => {
      const comp = createAndFlush();
      expect(formatFn(comp, 'last_run_time')(null)).toBe(DASH);
    });
  });
});
