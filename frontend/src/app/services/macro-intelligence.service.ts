import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, forkJoin, interval, of } from 'rxjs';
import { switchMap, takeWhile, catchError, map } from 'rxjs/operators';
import type {
  MacroCalibrationResponse,
  BlackLittermanBlConfig,
  FredObservationPoint,
  MacroNewsItem,
  MacroNewsTheme,
  CountryMacroData,
  BondYieldSnapshot,
  MacroJobCreateResponse,
  MacroJobProgress,
  CompositeScorePoint,
  BusinessCyclePhase,
} from '../models/macro-intelligence.model';
import {
  COUNTRY_CODE_MAP,
  DB_COUNTRIES,
  COMPOSITE_SCORE_THRESHOLDS,
} from '../constants/macro-intelligence.constants';
import { environment } from '../../environments/environment';

// ---------------------------------------------------------------------------
// Private API response interfaces (raw backend shapes)
// ---------------------------------------------------------------------------

interface ApiMacroCalibration {
  phase: string;
  delta: number;
  tau: number;
  confidence: number;
  rationale: string;
  macro_summary: string;
  bl_config: BlackLittermanBlConfig;
}

interface ApiFredObservation {
  id: string;
  series_id: string;
  date: string;
  value: number | null;
  created_at: string;
  updated_at: string;
}

interface ApiTeObservation {
  id: string;
  country: string;
  indicator_key: string;
  date: string;
  value: number | null;
  created_at: string;
  updated_at: string;
}

interface ApiBondYieldRow {
  id: string;
  country: string;
  maturity: string;
  yield_value: number | null;
  day_change: number | null;
  month_change: number | null;
  year_change: number | null;
  reference_date: string | null;
  created_at: string;
  updated_at: string;
}

interface ApiBondYieldObsRow {
  id: string;
  country: string;
  maturity: string;
  date: string;
  yield_value: number | null;
  created_at: string;
  updated_at: string;
}

interface ApiNewsTheme {
  value: string;
  label: string;
}

interface ApiNewsItem {
  id: string;
  news_id: string;
  title: string | null;
  publisher: string | null;
  link: string | null;
  publish_time: string | null;
  source_ticker: string | null;
  source_query: string | null;
  themes: string | null;
  snippet: string | null;
  full_content: string | null;
  created_at: string;
  updated_at: string;
}

interface ApiCountrySummary {
  country: string;
  economic_indicators: Array<Record<string, unknown>>;
  te_indicators: ApiTeObservation[];
  bond_yields: ApiBondYieldRow[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MATURITY_ORDER = [
  '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y',
];

const VALID_BUSINESS_CYCLE_PHASES = new Set<string>([
  'EARLY_EXPANSION',
  'MID_EXPANSION',
  'LATE_EXPANSION',
  'CONTRACTION',
]);

@Injectable({ providedIn: 'root' })
export class MacroIntelligenceService {
  private readonly http = inject(HttpClient);
  private readonly apiBase = environment.apiUrl;

  // ── Macro calibration ──

  getMacroCalibration(): Observable<MacroCalibrationResponse | null> {
    return this.http.get<ApiMacroCalibration>(
      `${this.apiBase}views/macro-calibration`,
      { params: { country: 'USA' } },
    ).pipe(
      map(raw => {
        const normalized = raw.phase === 'RECESSION' ? 'CONTRACTION' : raw.phase;
        const phase = VALID_BUSINESS_CYCLE_PHASES.has(normalized)
          ? (normalized as BusinessCyclePhase)
          : ('CONTRACTION' as BusinessCyclePhase);
        return {
          phase,
          delta: raw.delta,
          tau: raw.tau,
          confidence: raw.confidence,
          rationale: raw.rationale,
          macro_summary: raw.macro_summary,
          timestamp: new Date().toISOString(),
          bl_config: raw.bl_config,
        };
      }),
      catchError(() => of(null)),
    );
  }

  // ── FRED / TE indicator series ──

  getFredPmi(): Observable<FredObservationPoint[]> {
    return this.http.get<ApiTeObservation[]>(
      `${this.apiBase}macro-data/te-observations`,
      { params: { country: 'USA', indicator_keys: 'manufacturing_pmi', limit: '500' } },
    ).pipe(
      map(rows => rows
        .filter(r => r.value !== null)
        .map(r => ({ date: r.date, value: r.value! })),
      ),
      catchError(() => of([])),
    );
  }

  getFredYieldSpread(): Observable<FredObservationPoint[]> {
    return this.fetchFredSeries('T10Y2Y');
  }

  getFredHyOas(): Observable<FredObservationPoint[]> {
    return this.fetchFredSeries('BAMLH0A0HYM2');
  }

  getFredVix(): Observable<FredObservationPoint[]> {
    return this.fetchFredSeries('VIXCLS');
  }

  getFredIgOas(): Observable<FredObservationPoint[]> {
    return this.fetchFredSeries('BAMLC0A0CM');
  }

  // ── Country macro data ──

  getCountryData(): Observable<CountryMacroData[]> {
    const requests = DB_COUNTRIES.map(country =>
      this.http.get<ApiCountrySummary>(
        `${this.apiBase}macro-data/countries/${country}`,
      ).pipe(
        map(summary => this.mapCountrySummary(summary)),
        catchError(() => of(null)),
      ),
    );

    return forkJoin(requests).pipe(
      map(results => results.filter((r): r is CountryMacroData => r !== null)),
      catchError(() => of([])),
    );
  }

  // ── Bond yields ──

  getBondYieldsToday(): Observable<BondYieldSnapshot[]> {
    return this.http.get<ApiBondYieldRow[]>(
      `${this.apiBase}macro-data/bond-yields`,
    ).pipe(
      map(rows => this.groupBondYields(rows, new Date().toISOString().split('T')[0])),
      catchError(() => of([])),
    );
  }

  getBondYields1YAgo(): Observable<BondYieldSnapshot[]> {
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    const start = new Date(oneYearAgo);
    start.setDate(start.getDate() - 3);
    const end = new Date(oneYearAgo);
    end.setDate(end.getDate() + 3);

    return this.http.get<ApiBondYieldObsRow[]>(
      `${this.apiBase}macro-data/bond-yield-observations`,
      {
        params: {
          start_date: this.formatDate(start),
          end_date: this.formatDate(end),
        },
      },
    ).pipe(
      map(rows => {
        // Deduplicate: keep the latest observation per (country, maturity)
        const latest = new Map<string, ApiBondYieldObsRow>();
        for (const row of rows) {
          const key = `${row.country}|${row.maturity}`;
          const existing = latest.get(key);
          if (!existing || row.date > existing.date) {
            latest.set(key, row);
          }
        }
        const dedupedRows = [...latest.values()];

        // Group by country
        const byCountry = new Map<string, ApiBondYieldObsRow[]>();
        for (const row of dedupedRows) {
          const group = byCountry.get(row.country) ?? [];
          group.push(row);
          byCountry.set(row.country, group);
        }

        const snapshots: BondYieldSnapshot[] = [];
        for (const [country, countryRows] of byCountry) {
          const sorted = countryRows
            .filter(r => MATURITY_ORDER.includes(r.maturity))
            .sort((a, b) => MATURITY_ORDER.indexOf(a.maturity) - MATURITY_ORDER.indexOf(b.maturity));
          snapshots.push({
            country: COUNTRY_CODE_MAP[country] ?? country,
            date: countryRows[0]?.date ?? '',
            curve: sorted.map(r => ({
              maturity: r.maturity,
              yield_pct: r.yield_value ?? 0,
            })),
          });
        }
        return snapshots;
      }),
      catchError(() => of([])),
    );
  }

  // ── News ──

  getNewsThemes(): Observable<MacroNewsTheme[]> {
    return this.http.get<ApiNewsTheme[]>(
      `${this.apiBase}macro-data/news/themes`,
    ).pipe(
      map(items => items.map(i => ({ id: i.value, label: i.label, count: 0 }))),
      catchError(() => of([])),
    );
  }

  getNews(theme?: string): Observable<MacroNewsItem[]> {
    let params = new HttpParams().set('limit', '50');
    if (theme) {
      params = params.set('theme', theme);
    }

    return this.http.get<ApiNewsItem[]>(
      `${this.apiBase}macro-data/news`,
      { params },
    ).pipe(
      map(items => items.map(n => ({
        id: n.news_id,
        headline: n.title ?? '',
        snippet: n.snippet ?? '',
        url: n.link ?? '#',
        publisher: n.publisher ?? '',
        published_at: n.publish_time ?? '',
        theme: n.themes ?? '',
      }))),
      catchError(() => of([])),
    );
  }

  // ── Composite history (computed client-side) ──

  getCompositeHistory(): Observable<CompositeScorePoint[]> {
    return forkJoin([this.getFredYieldSpread(), this.getFredHyOas()]).pipe(
      map(([spreadData, hyData]) => {
        const spreadByMonth = this.groupByMonth(spreadData);
        const hyByMonth = this.groupByMonth(hyData);

        const months = [...spreadByMonth.keys()]
          .filter(m => hyByMonth.has(m))
          .sort();

        return months.map(month => {
          const spreadVal = spreadByMonth.get(month)!;
          const hyVal = hyByMonth.get(month)!;

          let score = 0;
          if (spreadVal > COMPOSITE_SCORE_THRESHOLDS.YIELD_SPREAD_BULL) score += 1;
          else if (spreadVal < COMPOSITE_SCORE_THRESHOLDS.YIELD_SPREAD_BEAR) score -= 1;

          if (hyVal < COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BULL) score += 1;
          else if (hyVal > COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BEAR) score -= 1;

          return { month, score };
        });
      }),
      catchError(() => of([])),
    );
  }

  // ── Refresh (already wired to real API) ──

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

  // ── Private helpers ──

  private fetchFredSeries(seriesId: string): Observable<FredObservationPoint[]> {
    return this.http.get<ApiFredObservation[]>(
      `${this.apiBase}macro-data/fred/series`,
      { params: { series_id: seriesId, start_date: this.oneYearAgo(), limit: '500' } },
    ).pipe(
      map(rows => rows
        .filter(r => r.value !== null)
        .map(r => ({ date: r.date, value: r.value! })),
      ),
      catchError(() => of([])),
    );
  }

  private oneYearAgo(): string {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return this.formatDate(d);
  }

  private formatDate(d: Date): string {
    return d.toISOString().split('T')[0];
  }

  private mapCountrySummary(summary: ApiCountrySummary): CountryMacroData {
    const te = new Map(summary.te_indicators.map(i => [i.indicator_key, i]));
    const bonds = new Map(summary.bond_yields.map(b => [b.maturity, b]));

    const yield10y = bonds.get('10Y')?.yield_value ?? 0;
    const yield2y = bonds.get('2Y')?.yield_value ?? 0;

    // Build sparkline from yield curve across maturities
    const sparkline = MATURITY_ORDER
      .filter(m => bonds.has(m))
      .map(m => bonds.get(m)!.yield_value ?? 0);

    return {
      country: summary.country,
      country_code: COUNTRY_CODE_MAP[summary.country] ?? summary.country,
      gdp_growth: te.get('gdp_growth_rate')?.value ?? 0,
      inflation: te.get('inflation_rate')?.value ?? 0,
      unemployment: te.get('unemployment_rate')?.value ?? 0,
      interest_rate: te.get('interest_rate')?.value ?? 0,
      yield_10y: yield10y,
      yield_2y: yield2y,
      yield_spread_bps: Math.round((yield10y - yield2y) * 100),
      yield_sparkline: sparkline,
    };
  }

  private groupBondYields(rows: ApiBondYieldRow[], refDate: string): BondYieldSnapshot[] {
    const byCountry = new Map<string, ApiBondYieldRow[]>();
    for (const row of rows) {
      const group = byCountry.get(row.country) ?? [];
      group.push(row);
      byCountry.set(row.country, group);
    }

    const snapshots: BondYieldSnapshot[] = [];
    for (const [country, countryRows] of byCountry) {
      const sorted = countryRows
        .filter(r => MATURITY_ORDER.includes(r.maturity))
        .sort((a, b) => MATURITY_ORDER.indexOf(a.maturity) - MATURITY_ORDER.indexOf(b.maturity));
      snapshots.push({
        country: COUNTRY_CODE_MAP[country] ?? country,
        date: refDate,
        curve: sorted.map(r => ({
          maturity: r.maturity,
          yield_pct: r.yield_value ?? 0,
        })),
      });
    }
    return snapshots;
  }

  private groupByMonth(data: FredObservationPoint[]): Map<string, number> {
    const byMonth = new Map<string, FredObservationPoint>();
    for (const point of data) {
      const month = point.date.substring(0, 7); // YYYY-MM
      const existing = byMonth.get(month);
      if (!existing || point.date > existing.date) {
        byMonth.set(month, point);
      }
    }
    const result = new Map<string, number>();
    for (const [month, point] of byMonth) {
      result.set(month, point.value);
    }
    return result;
  }

  private isTerminalStatus(status: string): boolean {
    return status === 'completed' || status === 'failed';
  }
}
