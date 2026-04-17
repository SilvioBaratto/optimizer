import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type {
  DriftApiResponse,
  RebalanceDecideApiResponse,
  RebalanceDecideRequest,
  RebalancePreviewApiResponse,
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
  RebalancingPolicyListApiResponse,
} from '../models/rebalancing.model';
import type { SnapshotListResponseDto } from '../models/portfolio-api.model';

/**
 * HTTP client for rebalancing workflow endpoints.
 *
 *   GET  /portfolio-analytics/{name}/drift
 *   GET  /portfolio/{name}/rebalance-policy
 *   POST /portfolio/{name}/rebalance-policy
 *   POST /portfolio/{name}/activate-policy/{policy_id}
 *   GET  /rebalance/preview/{portfolio_name}
 *   GET  /portfolio/{name}/snapshots
 *   POST /rebalance/decide
 *
 * Policy backend exposes list + create + activate only (no PUT/DELETE).
 */
@Injectable({ providedIn: 'root' })
export class RebalancingService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  getDrift(portfolioName: string, threshold?: number): Observable<DriftApiResponse> {
    const url = `${this.api}portfolio-analytics/${encodeURIComponent(portfolioName)}/drift`;
    if (threshold === undefined) return this.http.get<DriftApiResponse>(url);
    const params = new HttpParams().set('threshold', String(threshold));
    return this.http.get<DriftApiResponse>(url, { params });
  }

  listPolicies(portfolioName: string): Observable<RebalancingPolicyListApiResponse> {
    return this.http.get<RebalancingPolicyListApiResponse>(this.policyUrl(portfolioName));
  }

  createPolicy(
    portfolioName: string,
    body: RebalancingPolicyCreatePayload,
  ): Observable<RebalancingPolicyDto> {
    return this.http.post<RebalancingPolicyDto>(this.policyUrl(portfolioName), body);
  }

  activatePolicy(
    portfolioName: string,
    policyId: string,
  ): Observable<RebalancingPolicyDto> {
    const url =
      `${this.api}portfolio/${encodeURIComponent(portfolioName)}/activate-policy/${encodeURIComponent(policyId)}`;
    return this.http.post<RebalancingPolicyDto>(url, {});
  }

  getPreview(portfolioName: string): Observable<RebalancePreviewApiResponse> {
    return this.http.get<RebalancePreviewApiResponse>(
      `${this.api}rebalance/preview/${encodeURIComponent(portfolioName)}`,
    );
  }

  getSnapshots(portfolioName: string): Observable<SnapshotListResponseDto> {
    return this.http.get<SnapshotListResponseDto>(
      `${this.api}portfolio/${encodeURIComponent(portfolioName)}/snapshots`,
    );
  }

  decide(body: RebalanceDecideRequest): Observable<RebalanceDecideApiResponse> {
    return this.http.post<RebalanceDecideApiResponse>(`${this.api}rebalance/decide`, body);
  }

  private policyUrl(portfolioName: string): string {
    return `${this.api}portfolio/${encodeURIComponent(portfolioName)}/rebalance-policy`;
  }
}
