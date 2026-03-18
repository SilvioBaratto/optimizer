import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
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
import {
  MOCK_DASHBOARD_KPIS,
  MOCK_EQUITY_CURVE,
  MOCK_ALLOCATION_SUNBURST,
  MOCK_DRIFT_TABLE,
  MOCK_ACTIVITY_FEED,
  MOCK_MARKET_CONTEXT,
  MOCK_REGIME_INFO,
  MOCK_ASSET_CLASS_RETURNS,
} from '../mocks/dashboard-mocks';

@Injectable({ providedIn: 'root' })
export class DashboardService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;

  getPerformanceMetrics(name: string): Observable<ApiPerformanceMetricsResponse> {
    if (environment.useMocks) {
      return of({
        kpis: MOCK_DASHBOARD_KPIS,
        nav: 12_847_320,
        navChangePct: 0.0181,
      });
    }
    return this.http
      .get<ApiPerformanceMetricsResponse>(
        `${this.base}portfolio/${encodeURIComponent(name)}/performance-metrics`,
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
    if (environment.useMocks) {
      return of({
        points: MOCK_EQUITY_CURVE,
        portfolioTotalReturn: 0.2847,
        benchmarkTotalReturn: 0.18,
      });
    }
    const params = new HttpParams()
      .set('benchmark', benchmark)
      .set('period', period);
    return this.http
      .get<ApiEquityCurveResponse>(
        `${this.base}portfolio/${encodeURIComponent(name)}/equity-curve`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load equity curve',
      ))));
  }

  getAllocation(name: string): Observable<ApiAllocationResponse> {
    if (environment.useMocks) {
      return of({
        nodes: MOCK_ALLOCATION_SUNBURST as ApiAllocationResponse['nodes'],
        totalPositions: 19,
        totalSectors: 6,
      });
    }
    return this.http
      .get<ApiAllocationResponse>(
        `${this.base}portfolio/${encodeURIComponent(name)}/allocation`,
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load allocation',
      ))));
  }

  getDrift(name: string, threshold = 0.05): Observable<ApiDriftResponse> {
    if (environment.useMocks) {
      return of({
        entries: MOCK_DRIFT_TABLE,
        totalDrift: 0.031,
        breachedCount: 3,
        threshold,
      });
    }
    const params = new HttpParams().set('threshold', threshold.toString());
    return this.http
      .get<ApiDriftResponse>(
        `${this.base}portfolio/${encodeURIComponent(name)}/drift`,
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
    if (environment.useMocks) {
      return of({
        items: MOCK_ACTIVITY_FEED,
        total: MOCK_ACTIVITY_FEED.length,
      });
    }
    let params = new HttpParams()
      .set('limit', limit.toString())
      .set('offset', offset.toString());
    if (type) params = params.set('type', type);
    return this.http
      .get<ApiActivityFeedResponse>(
        `${this.base}portfolio/${encodeURIComponent(name)}/activity`,
        { params },
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load activity feed',
      ))));
  }

  getMarketSnapshot(): Observable<ApiMarketSnapshotResponse> {
    if (environment.useMocks) {
      return of({
        ...MOCK_MARKET_CONTEXT,
        asOf: new Date().toISOString(),
      });
    }
    return this.http
      .get<ApiMarketSnapshotResponse>(`${this.base}market/snapshot`)
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load market snapshot',
      ))));
  }

  getRegimeState(): Observable<ApiMarketRegimeResponse> {
    if (environment.useMocks) {
      return of({
        current: MOCK_REGIME_INFO.current,
        probability: MOCK_REGIME_INFO.probability,
        since: MOCK_REGIME_INFO.since,
        hmmStates: MOCK_REGIME_INFO.hmmStates,
        modelInfo: { nStates: 4, lastFitted: new Date().toISOString() },
      });
    }
    return this.http
      .get<ApiMarketRegimeResponse>(`${this.base}market/regime`)
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load regime state',
      ))));
  }

  getAssetClassReturns(name: string): Observable<ApiAssetClassReturnsResponse> {
    if (environment.useMocks) {
      return of({
        returns: MOCK_ASSET_CLASS_RETURNS,
        asOf: new Date().toISOString().slice(0, 10),
      });
    }
    return this.http
      .get<ApiAssetClassReturnsResponse>(
        `${this.base}portfolio/${encodeURIComponent(name)}/asset-class-returns`,
      )
      .pipe(catchError(err => throwError(() => new Error(
        err.error?.detail ?? 'Failed to load asset class returns',
      ))));
  }
}
