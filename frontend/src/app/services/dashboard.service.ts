import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import type {
  ApiPerformanceMetricsResponse,
  ApiEquityCurveResponse,
  ApiAllocationResponse,
  ApiMarketSnapshotResponse,
  ApiAssetClassReturnsResponse,
  ApiRollingMetricsResponse,
} from '../models/dashboard-api.model';

@Injectable({ providedIn: 'root' })
export class DashboardService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;

  getPerformanceMetrics(
    name: string,
    period: '1Y' | '3Y' | '5Y' | 'MAX' = '1Y',
  ): Observable<ApiPerformanceMetricsResponse> {
    const params = new HttpParams().set('period', period);
    return this.http
      .get<ApiPerformanceMetricsResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/performance-metrics`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load performance metrics',
      ))));
  }

  getRollingMetrics(
    name: string,
    period: '1Y' | '3Y' | '5Y' | 'MAX' = '3Y',
    window?: number,
  ): Observable<ApiRollingMetricsResponse> {
    let params = new HttpParams().set('period', period);
    if (window !== undefined) {
      params = params.set('window', String(window));
    }
    return this.http
      .get<ApiRollingMetricsResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/rolling-metrics`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load rolling metrics',
      ))));
  }

  getEquityCurve(
    name: string,
    benchmark = 'SPY',
    period: '1Y' | '3Y' | '5Y' | 'MAX' = '3Y',
  ): Observable<ApiEquityCurveResponse> {
    const params = new HttpParams()
      .set('benchmark', benchmark)
      .set('period', period);
    return this.http
      .get<ApiEquityCurveResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/equity-curve`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load equity curve',
      ))));
  }

  getAllocation(name: string): Observable<ApiAllocationResponse> {
    return this.http
      .get<ApiAllocationResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/allocation`,
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load allocation',
      ))));
  }

  getMarketSnapshot(): Observable<ApiMarketSnapshotResponse> {
    return this.http
      .get<ApiMarketSnapshotResponse>(`${this.base}market/snapshot`)
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load market snapshot',
      ))));
  }

  getAssetClassReturns(name: string): Observable<ApiAssetClassReturnsResponse> {
    return this.http
      .get<ApiAssetClassReturnsResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/asset-class-returns`,
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load asset class returns',
      ))));
  }
}
