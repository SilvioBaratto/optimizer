import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface PortfolioDto {
  id: string;
  name: string;
  description: string | null;
  currency: string;
  benchmark_ticker: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface PortfolioListResponseDto {
  items: PortfolioDto[];
  total: number;
}

/**
 * Real HTTP client for /api/v1/portfolio/ CRUD.
 *
 * Separate from `PortfolioService` (which holds mock optimization flows)
 * so that real portfolio-CRUD concerns stay isolated.
 */
@Injectable({ providedIn: 'root' })
export class PortfolioApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;

  list(): Observable<PortfolioListResponseDto> {
    return this.http
      .get<PortfolioListResponseDto>(`${this.base}portfolio/`)
      .pipe(
        catchError((err) =>
          throwError(
            () =>
              new Error(
                err?.error?.detail ?? 'Failed to load portfolio list',
              ),
          ),
        ),
      );
  }

  get(name: string): Observable<PortfolioDto> {
    return this.http
      .get<PortfolioDto>(
        `${this.base}portfolio/${encodeURIComponent(name)}`,
      )
      .pipe(
        catchError((err) =>
          throwError(
            () =>
              new Error(
                err?.error?.detail ?? `Failed to load portfolio '${name}'`,
              ),
          ),
        ),
      );
  }
}
