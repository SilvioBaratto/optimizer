import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import type {
  ApiPerformanceMetricsResponse,
  ApiEquityCurveResponse,
  ApiAllocationResponse,
  ApiDriftResponse,
  ApiActivityFeedResponse,
  ApiMarketSnapshotResponse,
  ApiMarketRegimeResponse,
  ApiAssetClassReturnsResponse,
} from '../models/dashboard-api.model';

@Injectable({ providedIn: 'root' })
export class DashboardService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;

  getPerformanceMetrics(name: string): Observable<ApiPerformanceMetricsResponse> {
    return this.http
      .get<ApiPerformanceMetricsResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/performance-metrics`,
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load performance metrics',
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

  getDrift(name: string, threshold = 0.05): Observable<ApiDriftResponse> {
    const params = new HttpParams().set('threshold', threshold.toString());
    return this.http
      .get<ApiDriftResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/drift`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load drift analysis',
      ))));
  }

  getActivity(
    name: string,
    limit = 20,
    offset = 0,
    type?: string,
  ): Observable<ApiActivityFeedResponse> {
    let params = new HttpParams()
      .set('limit', limit.toString())
      .set('offset', offset.toString());
    if (type) params = params.set('type', type);
    return this.http
      .get<ApiActivityFeedResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/activity`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load activity feed',
      ))));
  }

  getMarketSnapshot(): Observable<ApiMarketSnapshotResponse> {
    return this.http
      .get<ApiMarketSnapshotResponse>(`${this.base}market/snapshot`)
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load market snapshot',
      ))));
  }

  getRegimeState(): Observable<ApiMarketRegimeResponse> {
    return this.http
      .get<ApiMarketRegimeResponse>(`${this.base}market/regime`)
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load regime state',
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
