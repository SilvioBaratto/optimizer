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
});
