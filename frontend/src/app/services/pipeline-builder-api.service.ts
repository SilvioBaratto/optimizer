import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import type {
  CreateSessionResponse,
  RunLevelConfig,
  StepPollResponse,
  StepRunResponse,
} from '../models/pipeline-builder.model';

interface WireStepRunResponse {
  job_id: string;
  status: string;
  message: string;
}

@Injectable({ providedIn: 'root' })
export class PipelineBuilderApiService {
  private readonly http = inject(HttpClient);
  private readonly base = `${environment.apiUrl}pipeline-builder`;

  createSession(config: RunLevelConfig): Observable<CreateSessionResponse> {
    return this.http.post<CreateSessionResponse>(`${this.base}/sessions`, config);
  }

  runStep(
    sessionId: string,
    stepId: string,
    params: Record<string, unknown>,
  ): Observable<StepRunResponse> {
    const url = `${this.base}/sessions/${encode(sessionId)}/steps/${encode(stepId)}`;
    return this.http
      .post<WireStepRunResponse>(url, params)
      .pipe(map(toStepRunResponse));
  }

  pollStep(sessionId: string, stepId: string): Observable<StepPollResponse> {
    const url = `${this.base}/sessions/${encode(sessionId)}/steps/${encode(stepId)}`;
    return this.http.get<StepPollResponse>(url);
  }

  deleteSession(sessionId: string): Observable<void> {
    const url = `${this.base}/sessions/${encode(sessionId)}`;
    return this.http.delete<void>(url);
  }

  getArtifactUrl(sessionId: string, name: string): string {
    return `${this.base}/sessions/${encode(sessionId)}/artifacts/${name}`;
  }
}

function encode(segment: string): string {
  return encodeURIComponent(segment);
}

function toStepRunResponse(wire: WireStepRunResponse): StepRunResponse {
  return { jobId: wire.job_id, status: wire.status, message: wire.message };
}
