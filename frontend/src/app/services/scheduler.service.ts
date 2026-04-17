import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type { SchedulerStatusResponse } from '../models/scheduler.model';

/**
 * HTTP client for the read-only scheduler status endpoint.
 *
 *   GET /api/v1/scheduler/status → SchedulerStatusResponse
 *     { schedulerRunning: bool, jobs: SchedulerJobStatusDto[] }
 *
 * The endpoint is a pure read surface; there are no write operations against
 * the scheduler from the frontend (APScheduler runs inside the FastAPI process).
 */
@Injectable({ providedIn: 'root' })
export class SchedulerService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  getStatus(): Observable<SchedulerStatusResponse> {
    return this.http.get<SchedulerStatusResponse>(`${this.api}scheduler/status`);
  }
}
