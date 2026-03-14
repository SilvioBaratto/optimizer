import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, from, forkJoin, interval, of } from 'rxjs';
import { switchMap, takeWhile, catchError, map } from 'rxjs/operators';
import { MockFetchService } from './mock-fetch.service';
import type {
  MacroCalibrationResponse,
  FredObservationPoint,
  MacroNewsItem,
  MacroNewsTheme,
  CountryMacroData,
  BondYieldSnapshot,
  MacroJobCreateResponse,
  MacroJobProgress,
  CompositeScorePoint,
} from '../models/macro-intelligence.model';
import {
  MOCK_MACRO_CALIBRATION,
  MOCK_FRED_PMI,
  MOCK_FRED_YIELD_SPREAD,
  MOCK_FRED_HY_OAS,
  MOCK_FRED_VIX,
  MOCK_FRED_IG_OAS,
  MOCK_COUNTRY_DATA,
  MOCK_BOND_YIELDS_TODAY,
  MOCK_BOND_YIELDS_1Y_AGO,
  MOCK_NEWS_THEMES,
  MOCK_NEWS_ITEMS,
  MOCK_COMPOSITE_HISTORY,
} from '../mocks/macro-intelligence-mocks';
import { COUNTRY_CODE_MAP } from '../constants/macro-intelligence.constants';
import { environment } from '../../environments/environment';

@Injectable({ providedIn: 'root' })
export class MacroIntelligenceService {
  private readonly mockFetch = inject(MockFetchService);
  private readonly http = inject(HttpClient);
  private readonly apiBase = environment.apiUrl;

  getMacroCalibration(): Observable<MacroCalibrationResponse> {
    return from(this.mockFetch.fetch(MOCK_MACRO_CALIBRATION));
  }

  getFredPmi(): Observable<FredObservationPoint[]> {
    return from(this.mockFetch.fetch(MOCK_FRED_PMI));
  }

  getFredYieldSpread(): Observable<FredObservationPoint[]> {
    return from(this.mockFetch.fetch(MOCK_FRED_YIELD_SPREAD));
  }

  getFredHyOas(): Observable<FredObservationPoint[]> {
    return from(this.mockFetch.fetch(MOCK_FRED_HY_OAS));
  }

  getFredVix(): Observable<FredObservationPoint[]> {
    return from(this.mockFetch.fetch(MOCK_FRED_VIX));
  }

  getFredIgOas(): Observable<FredObservationPoint[]> {
    return from(this.mockFetch.fetch(MOCK_FRED_IG_OAS));
  }

  getCountryData(): Observable<CountryMacroData[]> {
    return from(this.mockFetch.fetch(MOCK_COUNTRY_DATA));
  }

  getBondYieldsToday(): Observable<BondYieldSnapshot[]> {
    return from(this.mockFetch.fetch(MOCK_BOND_YIELDS_TODAY)).pipe(
      map(snapshots => this.normalizeBondYields(snapshots)),
    );
  }

  getBondYields1YAgo(): Observable<BondYieldSnapshot[]> {
    return from(this.mockFetch.fetch(MOCK_BOND_YIELDS_1Y_AGO)).pipe(
      map(snapshots => this.normalizeBondYields(snapshots)),
    );
  }

  getNewsThemes(): Observable<MacroNewsTheme[]> {
    return from(this.mockFetch.fetch(MOCK_NEWS_THEMES));
  }

  getNews(theme?: string): Observable<MacroNewsItem[]> {
    const items = theme
      ? MOCK_NEWS_ITEMS.filter(n => n.theme === theme)
      : MOCK_NEWS_ITEMS;
    return from(this.mockFetch.fetch(items));
  }

  getCompositeHistory(): Observable<CompositeScorePoint[]> {
    return from(this.mockFetch.fetch(MOCK_COMPOSITE_HISTORY));
  }

  triggerRefresh(): Observable<void> {
    const jobPaths = [
      'macro-data/fetch',
      'macro-data/fred/fetch',
      'macro-data/news/fetch',
    ];

    const posts$ = jobPaths.map(path =>
      this.http.post<MacroJobCreateResponse>(`${this.apiBase}${path}`, {}).pipe(
        map(r => ({ path, jobId: r.job_id })),
        catchError(() => of(null)),
      ),
    );

    return forkJoin(posts$).pipe(
      switchMap(responses => {
        const polls$ = responses
          .filter((r): r is { path: string; jobId: string } => r !== null)
          .map(({ path, jobId }) =>
            interval(3000).pipe(
              switchMap(() => this.http.get<MacroJobProgress>(`${this.apiBase}${path}/${jobId}`)),
              takeWhile(p => !this.isTerminalStatus(p.status), true),
            ),
          );

        if (polls$.length === 0) return of(undefined as void);
        return forkJoin(polls$).pipe(map(() => undefined as void));
      }),
    );
  }

  private normalizeBondYields(snapshots: BondYieldSnapshot[]): BondYieldSnapshot[] {
    return snapshots.map(s => ({
      ...s,
      country: COUNTRY_CODE_MAP[s.country] ?? s.country,
    }));
  }

  private isTerminalStatus(status: string): boolean {
    return status === 'completed' || status === 'failed';
  }
}
