import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type {
  ConcentrationApiResponse,
  CorrelationApiResponse,
  FactorExposureApiResponse,
  LiquidityApiResponse,
  RiskLimitCreatePayload,
  RiskLimitDto,
  RiskLimitListResponse,
  RiskLimitUpdatePayload,
  StressScenarioApiResponse,
  StressScenarioRequest,
  VarApiResponse,
} from './risk.model';

interface VarQuery {
  lookback?: number;
  method?: 'historical' | 'parametric';
}

interface LiquidityQuery {
  lookbackDays?: number;
  participationRate?: number;
}

function toVarParams(query?: VarQuery): HttpParams | undefined {
  if (!query) return undefined;
  let params = new HttpParams();
  if (query.lookback !== undefined) params = params.set('lookback', String(query.lookback));
  if (query.method) params = params.set('method', query.method);
  return params.keys().length ? params : undefined;
}

function toLiquidityParams(query?: LiquidityQuery): HttpParams | undefined {
  if (!query) return undefined;
  let params = new HttpParams();
  if (query.lookbackDays !== undefined)
    params = params.set('lookback_days', String(query.lookbackDays));
  if (query.participationRate !== undefined)
    params = params.set('participation_rate', String(query.participationRate));
  return params.keys().length ? params : undefined;
}

/**
 * HTTP client for risk analytics, stress scenarios and risk-limits CRUD.
 *
 * Endpoint groups:
 *   - GET  /portfolio/{name}/risk/{var|correlation|factor-exposure|concentration|liquidity}
 *   - POST /risk/stress-scenarios     (synchronous LLM call)
 *   - GET/POST    /risk/{portfolio_name}/limits
 *   - PUT/DELETE  /risk/{portfolio_name}/limits/{limit_id}
 */
@Injectable({ providedIn: 'root' })
export class RiskService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  // ── Read-only analytics ──────────────────────────────────────────────────
  getVar(name: string, query?: VarQuery): Observable<VarApiResponse> {
    const params = toVarParams(query);
    return this.http.get<VarApiResponse>(
      `${this.api}portfolio/${encodeURIComponent(name)}/risk/var`,
      params ? { params } : {},
    );
  }

  getCorrelation(name: string, lookback?: number): Observable<CorrelationApiResponse> {
    const opts = lookback !== undefined
      ? { params: new HttpParams().set('lookback', String(lookback)) }
      : {};
    return this.http.get<CorrelationApiResponse>(
      `${this.api}portfolio/${encodeURIComponent(name)}/risk/correlation`,
      opts,
    );
  }

  getFactorExposure(name: string): Observable<FactorExposureApiResponse> {
    return this.http.get<FactorExposureApiResponse>(
      `${this.api}portfolio/${encodeURIComponent(name)}/risk/factor-exposure`,
    );
  }

  getConcentration(name: string, n?: number): Observable<ConcentrationApiResponse> {
    const opts = n !== undefined
      ? { params: new HttpParams().set('n', String(n)) }
      : {};
    return this.http.get<ConcentrationApiResponse>(
      `${this.api}portfolio/${encodeURIComponent(name)}/risk/concentration`,
      opts,
    );
  }

  getLiquidity(name: string, query?: LiquidityQuery): Observable<LiquidityApiResponse> {
    const params = toLiquidityParams(query);
    return this.http.get<LiquidityApiResponse>(
      `${this.api}portfolio/${encodeURIComponent(name)}/risk/liquidity`,
      params ? { params } : {},
    );
  }

  // ── LLM-driven POSTs ─────────────────────────────────────────────────────
  generateStressScenarios(
    body: StressScenarioRequest,
  ): Observable<StressScenarioApiResponse> {
    return this.http.post<StressScenarioApiResponse>(
      `${this.api}risk/stress-scenarios`,
      body,
    );
  }

  // ── Risk-limits CRUD ─────────────────────────────────────────────────────
  listLimits(portfolioName: string): Observable<RiskLimitListResponse> {
    return this.http.get<RiskLimitListResponse>(this.limitsBase(portfolioName));
  }

  createLimit(
    portfolioName: string,
    body: RiskLimitCreatePayload,
  ): Observable<RiskLimitDto> {
    return this.http.post<RiskLimitDto>(this.limitsBase(portfolioName), body);
  }

  updateLimit(
    portfolioName: string,
    limitId: string,
    body: RiskLimitUpdatePayload,
  ): Observable<RiskLimitDto> {
    return this.http.put<RiskLimitDto>(this.limitUrl(portfolioName, limitId), body);
  }

  deleteLimit(portfolioName: string, limitId: string): Observable<void> {
    return this.http.delete<void>(this.limitUrl(portfolioName, limitId));
  }

  private limitsBase(portfolioName: string): string {
    return `${this.api}risk/${encodeURIComponent(portfolioName)}/limits`;
  }

  private limitUrl(portfolioName: string, limitId: string): string {
    return `${this.limitsBase(portfolioName)}/${encodeURIComponent(limitId)}`;
  }
}
