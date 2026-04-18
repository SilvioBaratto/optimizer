import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import type {
  DomainMeta,
  DomainStatus,
  FreshnessLevel,
  JobListResponse,
  JobSummary,
} from '../models/jobs.model';
import { DOMAIN_META } from '../models/jobs.model';

const FALLBACK_META: Omit<DomainMeta, 'domain' | 'label'> = {
  icon: 'activity',
  staleThresholdHours: 168,
  criticalThresholdHours: 720,
};

const KNOWN_META: ReadonlyMap<string, DomainMeta> = new Map(
  DOMAIN_META.map((meta) => [meta.domain, meta]),
);

function titleCase(raw: string): string {
  return raw
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function syntheticMeta(domain: string): DomainMeta {
  return { domain, label: titleCase(domain), ...FALLBACK_META };
}

function metaFor(domain: string): DomainMeta {
  return KNOWN_META.get(domain) ?? syntheticMeta(domain);
}

@Injectable({ providedIn: 'root' })
export class JobsService {
  private readonly http = inject(HttpClient);
  private readonly apiBase = environment.apiUrl;

  /**
   * Fetch domain-level status entries for the Pipeline Health grid.
   *
   * Issue #440: returns only the domains that have jobs in the API
   * response. An empty response yields an empty array — the page renders
   * a single "no pipeline activity yet" tile rather than fanning out a
   * placeholder card per known domain. HTTP errors are propagated so
   * the caller can render a banner instead of pretending to have data.
   */
  getDomainStatuses(): Observable<DomainStatus[]> {
    return this.http
      .get<JobListResponse>(`${this.apiBase}jobs`, { params: { limit: '100' } })
      .pipe(map((res) => this.buildDomainStatuses(res.jobs)));
  }

  getJob(id: string): Observable<JobSummary> {
    return this.http.get<JobSummary>(
      `${this.apiBase}jobs/${encodeURIComponent(id)}`,
    );
  }

  private buildDomainStatuses(jobs: JobSummary[]): DomainStatus[] {
    const byDomain = this.groupByDomain(jobs);
    return [...byDomain.entries()].map(([domain, domainJobs]) =>
      this.statusFor(metaFor(domain), domainJobs),
    );
  }

  private groupByDomain(jobs: JobSummary[]): Map<string, JobSummary[]> {
    const byDomain = new Map<string, JobSummary[]>();
    for (const job of jobs) {
      const group = byDomain.get(job.domain) ?? [];
      group.push(job);
      byDomain.set(job.domain, group);
    }
    return byDomain;
  }

  private statusFor(meta: DomainMeta, domainJobs: JobSummary[]): DomainStatus {
    const lastSuccess =
      domainJobs
        .filter((j) => j.status === 'completed')
        .sort((a, b) => (b.finished_at ?? '').localeCompare(a.finished_at ?? ''))[0] ??
      null;
    const running =
      domainJobs.find((j) => j.status === 'running' || j.status === 'pending') ?? null;
    const recentFailures = domainJobs
      .filter((j) => j.status === 'failed')
      .sort((a, b) => (b.started_at ?? '').localeCompare(a.started_at ?? ''))
      .slice(0, 5);
    const { freshness, ageHours } = this.computeFreshness(meta, lastSuccess);
    return {
      meta,
      lastSuccess,
      running,
      recentFailures,
      freshness,
      lastSuccessAgeHours: ageHours,
    };
  }

  private computeFreshness(
    meta: DomainMeta,
    lastSuccess: JobSummary | null,
  ): { freshness: FreshnessLevel; ageHours: number | null } {
    if (!lastSuccess?.finished_at) {
      return { freshness: 'unknown', ageHours: null };
    }
    const ageHours = (Date.now() - new Date(lastSuccess.finished_at).getTime()) / 3_600_000;
    if (ageHours >= meta.criticalThresholdHours) return { freshness: 'critical', ageHours };
    if (ageHours >= meta.staleThresholdHours) return { freshness: 'stale', ageHours };
    return { freshness: 'fresh', ageHours };
  }
}
