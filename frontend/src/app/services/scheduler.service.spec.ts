import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { SchedulerService } from './scheduler.service';
import { environment } from '../../environments/environment';
import type { SchedulerStatusResponse } from '../models/scheduler.model';

const API = environment.apiUrl;

describe('SchedulerService', () => {
  let svc: SchedulerService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        SchedulerService,
      ],
    });
    svc = TestBed.inject(SchedulerService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('getStatus()', () => {
    it('GETs /scheduler/status and returns the running flag + jobs list', () => {
      let result: SchedulerStatusResponse | undefined;
      svc.getStatus().subscribe((r) => (result = r));

      const req = http.expectOne(`${API}scheduler/status`);
      expect(req.request.method).toBe('GET');

      req.flush({
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

      expect(result?.schedulerRunning).toBe(true);
      expect(result?.jobs.length).toBe(1);
      expect(result?.jobs[0].jobId).toBe('daily_pipeline');
    });

    it('returns schedulerRunning=false + empty jobs when the scheduler is stopped', () => {
      let result: SchedulerStatusResponse | undefined;
      svc.getStatus().subscribe((r) => (result = r));
      http.expectOne(`${API}scheduler/status`).flush({
        schedulerRunning: false,
        jobs: [],
      });
      expect(result?.schedulerRunning).toBe(false);
      expect(result?.jobs).toEqual([]);
    });

    it('propagates 500 on scheduler errors', () => {
      let error: HttpErrorResponse | undefined;
      svc.getStatus().subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}scheduler/status`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      expect(error?.status).toBe(500);
    });
  });
});
