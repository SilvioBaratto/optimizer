import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, throwError, timer } from 'rxjs';
import { catchError, switchMap, takeWhile } from 'rxjs/operators';

import { environment } from '../../../environments/environment';

export interface ReferenceIndexSeedRequest {
  tickers: string[];
}

export interface ReferenceIndexSeedJobResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message?: string;
}

export interface ReferenceIndexSeedProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  errors: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface ReferenceIndexStatusItem {
  ticker: string;
  instrumentExists: boolean;
  priceRows: number;
  latestPriceDate: string | null;
  isStale: boolean;
}

export interface ReferenceIndexStatusResponse {
  items: ReferenceIndexStatusItem[];
  total: number;
  healthyCount: number;
  missingCount: number;
  staleCount: number;
  staleDays: number;
}

export interface ReferenceIndexConfiguredResponse {
  tickers: string[];
  settingsTickers: string[];
  portfolioTickers: string[];
}

@Injectable({ providedIn: 'root' })
export class ReferenceIndexService {
  private readonly http = inject(HttpClient);
  private readonly base = `${environment.apiUrl}reference-indices`;

  startSeed(tickers: string[]): Observable<ReferenceIndexSeedJobResponse> {
    return this.http
      .post<ReferenceIndexSeedJobResponse>(`${this.base}/seed`, { tickers })
      .pipe(
        catchError((err) =>
          throwError(() => new Error(err.error?.detail ?? 'Failed to start seed')),
        ),
      );
  }

  getJob(jobId: string): Observable<ReferenceIndexSeedProgress> {
    return this.http
      .get<ReferenceIndexSeedProgress>(`${this.base}/seed/${jobId}`)
      .pipe(
        catchError((err) =>
          throwError(() => new Error(err.error?.detail ?? 'Failed to fetch job')),
        ),
      );
  }

  getStatus(): Observable<ReferenceIndexStatusResponse> {
    return this.http
      .get<ReferenceIndexStatusResponse>(`${this.base}/status`)
      .pipe(
        catchError((err) =>
          throwError(() => new Error(err.error?.detail ?? 'Failed to load status')),
        ),
      );
  }

  getConfigured(): Observable<ReferenceIndexConfiguredResponse> {
    return this.http
      .get<ReferenceIndexConfiguredResponse>(`${this.base}/configured`)
      .pipe(
        catchError((err) =>
          throwError(() => new Error(err.error?.detail ?? 'Failed to load configured tickers')),
        ),
      );
  }

  /** Poll a seed job every `intervalMs` until status is terminal. */
  pollUntilDone(jobId: string, intervalMs = 2000): Observable<ReferenceIndexSeedProgress> {
    return timer(0, intervalMs).pipe(
      switchMap(() => this.getJob(jobId)),
      takeWhile((j) => j.status !== 'completed' && j.status !== 'failed', true),
    );
  }
}
