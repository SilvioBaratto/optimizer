import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import {
  BrokerAccountDto,
  BrokerPositionDto,
  CreatePortfolioDto,
  CreateSnapshotDto,
  PortfolioDto,
  PortfolioListResponseDto,
  SnapshotDto,
  SnapshotListResponseDto,
  SyncJobResponseDto,
  SyncProgressResponseDto,
} from '../models/portfolio-api.model';

function wrapError(fallback: string) {
  return (err: { error?: { detail?: string } }) =>
    throwError(() => new Error(err?.error?.detail ?? fallback));
}

/**
 * HTTP client for /api/v1/portfolio/*.
 */
@Injectable({ providedIn: 'root' })
export class PortfolioApiService {
  private readonly http = inject(HttpClient);
  private readonly base = `${environment.apiUrl}portfolio`;

  list(): Observable<PortfolioListResponseDto> {
    return this.http
      .get<PortfolioListResponseDto>(`${this.base}/`)
      .pipe(catchError(wrapError('Failed to load portfolio list')));
  }

  get(name: string): Observable<PortfolioDto> {
    return this.http
      .get<PortfolioDto>(this.forName(name))
      .pipe(catchError(wrapError(`Failed to load portfolio '${name}'`)));
  }

  create(payload: CreatePortfolioDto): Observable<PortfolioDto> {
    return this.http
      .post<PortfolioDto>(`${this.base}/`, payload)
      .pipe(catchError(wrapError('Failed to create portfolio')));
  }

  getSnapshots(name: string): Observable<SnapshotListResponseDto> {
    return this.http
      .get<SnapshotListResponseDto>(`${this.forName(name)}/snapshots`)
      .pipe(catchError(wrapError(`Failed to load snapshots for '${name}'`)));
  }

  createSnapshot(name: string, payload: CreateSnapshotDto): Observable<SnapshotDto> {
    return this.http
      .post<SnapshotDto>(`${this.forName(name)}/snapshots`, payload)
      .pipe(catchError(wrapError(`Failed to create snapshot for '${name}'`)));
  }

  getLatestSnapshot(name: string): Observable<SnapshotDto> {
    return this.http
      .get<SnapshotDto>(`${this.forName(name)}/snapshots/latest`)
      .pipe(catchError(wrapError(`Failed to load latest snapshot for '${name}'`)));
  }

  syncPortfolio(name: string): Observable<SyncJobResponseDto> {
    return this.http
      .post<SyncJobResponseDto>(`${this.forName(name)}/sync`, {})
      .pipe(catchError(wrapError(`Failed to start sync for '${name}'`)));
  }

  getSyncProgress(name: string, jobId: string): Observable<SyncProgressResponseDto> {
    return this.http
      .get<SyncProgressResponseDto>(
        `${this.forName(name)}/sync/${encodeURIComponent(jobId)}`,
      )
      .pipe(catchError(wrapError(`Failed to poll sync progress for '${name}'`)));
  }

  getPositions(name: string): Observable<BrokerPositionDto[]> {
    return this.http
      .get<BrokerPositionDto[]>(`${this.forName(name)}/positions`)
      .pipe(catchError(wrapError(`Failed to load positions for '${name}'`)));
  }

  getAccount(name: string): Observable<BrokerAccountDto> {
    return this.http
      .get<BrokerAccountDto>(`${this.forName(name)}/account`)
      .pipe(catchError(wrapError(`Failed to load account for '${name}'`)));
  }

  private forName(name: string): string {
    return `${this.base}/${encodeURIComponent(name)}`;
  }
}
