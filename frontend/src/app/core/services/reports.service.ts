import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import type {
  ReportGenerateRequest,
  ReportJobCreateResponse,
  ReportJobProgress,
} from '../models/report.model';

/**
 * HTTP client for the report-generation endpoints.
 *
 *   POST /api/v1/reports/generate        → 202 + ReportJobCreateResponse
 *   GET  /api/v1/reports/jobs/{job_id}   → ReportJobProgress (module-specific path)
 *   GET  /api/v1/reports/{report_id}/download → PDF stream (client navigates to URL)
 *
 * The polling path is module-specific, so this service exposes its own
 * pollJob() method rather than routing through the generic JobsService.
 */
@Injectable({ providedIn: 'root' })
export class ReportsService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  generate(body: ReportGenerateRequest): Observable<ReportJobCreateResponse> {
    return this.http.post<ReportJobCreateResponse>(`${this.api}reports/generate`, body);
  }

  pollJob(jobId: string): Observable<ReportJobProgress> {
    return this.http.get<ReportJobProgress>(
      `${this.api}reports/jobs/${encodeURIComponent(jobId)}`,
    );
  }

  downloadUrl(reportId: string): string {
    return `${this.api}reports/${encodeURIComponent(reportId)}/download`;
  }
}
