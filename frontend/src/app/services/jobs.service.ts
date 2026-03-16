import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import type {
  JobListResponse,
  JobSummary,
  DomainMeta,
  DomainStatus,
  FreshnessLevel,
} from '../models/jobs.model';
import { DOMAIN_META } from '../models/jobs.model';

@Injectable({ providedIn: 'root' })
export class JobsService {
  private readonly http = inject(HttpClient);
  private readonly apiBase = environment.apiUrl;

  getDomainStatuses(): Observable<DomainStatus[]> {
    return this.http
      .get<JobListResponse>(`${this.apiBase}jobs`, { params: { limit: '100' } })
      .pipe(
        map(res => this.buildDomainStatuses(res.jobs)),
        catchError(() => of(DOMAIN_META.map(meta => this.emptyDomainStatus(meta)))),
      );
  }

  private buildDomainStatuses(jobs: JobSummary[]): DomainStatus[] {
    const byDomain = new Map<string, JobSummary[]>();
    for (const job of jobs) {
      const group = byDomain.get(job.domain) ?? [];
      group.push(job);
      byDomain.set(job.domain, group);
    }

    return DOMAIN_META.map(meta => {
      const domainJobs = byDomain.get(meta.domain) ?? [];

      const lastSuccess = domainJobs
        .filter(j => j.status === 'completed')
        .sort((a, b) => (b.finished_at ?? '').localeCompare(a.finished_at ?? ''))[0] ?? null;

      const running = domainJobs
        .find(j => j.status === 'running' || j.status === 'pending') ?? null;

      const recentFailures = domainJobs
        .filter(j => j.status === 'failed')
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
    });
  }

  private computeFreshness(
    meta: DomainMeta,
    lastSuccess: JobSummary | null,
  ): { freshness: FreshnessLevel; ageHours: number | null } {
    if (!lastSuccess?.finished_at) {
      return { freshness: 'unknown', ageHours: null };
    }

    const ageMs = Date.now() - new Date(lastSuccess.finished_at).getTime();
    const ageHours = ageMs / 3_600_000;

    if (ageHours >= meta.criticalThresholdHours) {
      return { freshness: 'critical', ageHours };
    }
    if (ageHours >= meta.staleThresholdHours) {
      return { freshness: 'stale', ageHours };
    }
    return { freshness: 'fresh', ageHours };
  }

  private emptyDomainStatus(meta: DomainMeta): DomainStatus {
    return {
      meta,
      lastSuccess: null,
      running: null,
      recentFailures: [],
      freshness: 'unknown',
      lastSuccessAgeHours: null,
    };
  }
}
