import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, forkJoin, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { mapHttpError } from '../core/http-error';
import { FormatService } from '../core/services/format.service';
import type {
  ApiPerformanceMetricsResponse,
  ApiEquityCurveResponse,
  ApiAllocationResponse,
  ApiMarketSnapshotResponse,
  ApiAssetClassReturnsResponse,
  ApiRollingMetricsResponse,
} from '../core/models/dashboard-api.model';
import type {
  DashboardKPI,
  EquityCurvePoint,
  AllocationNode,
  AssetClassReturn,
  PortfolioDataSnapshot,
} from './dashboard.model';

export function returnCellBg(value: number): string {
  if (value > 0.05) return 'background-color: var(--color-heatmap-7); color: white';
  if (value > 0.02) return 'background-color: var(--color-heatmap-6); color: white';
  if (value > 0) return 'background-color: var(--color-heatmap-5)';
  if (value === 0) return 'background-color: var(--color-heatmap-4)';
  if (value > -0.02) return 'background-color: var(--color-heatmap-3)';
  if (value > -0.05) return 'background-color: var(--color-heatmap-2); color: white';
  return 'background-color: var(--color-heatmap-1); color: white';
}

@Injectable({ providedIn: 'root' })
export class DashboardService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiUrl;
  private readonly fmt = inject(FormatService);

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
      .pipe(catchError(mapHttpError()));
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
      .pipe(catchError(mapHttpError()));
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
      .pipe(catchError(mapHttpError()));
  }

  getAllocation(name: string): Observable<ApiAllocationResponse> {
    return this.http
      .get<ApiAllocationResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/allocation`,
      )
      .pipe(catchError(mapHttpError()));
  }

  getMarketSnapshot(): Observable<ApiMarketSnapshotResponse> {
    return this.http
      .get<ApiMarketSnapshotResponse>(`${this.base}market/snapshot`)
      .pipe(catchError(mapHttpError()));
  }

  getAssetClassReturns(name: string): Observable<ApiAssetClassReturnsResponse> {
    return this.http
      .get<ApiAssetClassReturnsResponse>(
        `${this.base}portfolio-analytics/${encodeURIComponent(name)}/asset-class-returns`,
      )
      .pipe(catchError(mapHttpError()));
  }

  loadPortfolioData(
    name: string,
    period: '1Y' | '3Y' | '5Y' | 'MAX',
    benchmark: string,
  ): Observable<PortfolioDataSnapshot> {
    return forkJoin({
      performance: this.getPerformanceMetrics(name, period).pipe(catchError(() => of(null))),
      equity: this.getEquityCurve(name, benchmark, period).pipe(catchError(() => of(null))),
      allocation: this.getAllocation(name).pipe(catchError(() => of(null))),
      assetClass: this.getAssetClassReturns(name).pipe(catchError(() => of(null))),
    }).pipe(
      map(({ performance, equity, allocation, assetClass }) => ({
        kpis: (performance?.kpis ?? []) as DashboardKPI[],
        nav: performance?.nav ?? 0,
        navChangePct: performance?.navChangePct ?? 0,
        currency: performance?.currency ?? 'EUR',
        equityCurvePoints: (equity?.points ?? []) as EquityCurvePoint[],
        allocationNodes: (allocation?.nodes ?? []) as AllocationNode[],
        assetClassReturns: (assetClass?.returns ?? []) as AssetClassReturn[],
      })),
    );
  }

  formatKpiValue(kpi: DashboardKPI, currency: string): string {
    switch (kpi.format) {
      case 'percent': return this.fmt.formatPercent(kpi.value);
      case 'currency': return this.fmt.formatCurrencyCompact(kpi.value, currency);
      case 'ratio': return this.fmt.formatRatio(kpi.value);
      default: return kpi.value.toLocaleString();
    }
  }

  kpiTrend(kpi: DashboardKPI): 'up' | 'down' | 'flat' {
    if (kpi.change > 0) return 'up';
    if (kpi.change < 0) return 'down';
    return 'flat';
  }

  kpiDelta(kpi: DashboardKPI, nav: number): number {
    if (kpi.format === 'currency') return nav ? kpi.change / nav : 0;
    return kpi.change;
  }

  kpiDeltaFormat(kpi: DashboardKPI): 'percent' | 'absolute' {
    return kpi.format === 'percent' || kpi.format === 'currency' ? 'percent' : 'absolute';
  }
}
