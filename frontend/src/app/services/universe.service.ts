import { Injectable, InjectionToken, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import {
  BuildProgress,
  Exchange,
  InstrumentList,
  UniverseStats,
} from '../models/universe.model';

export const SEARCH_DEBOUNCE_MS = new InjectionToken<number>('SEARCH_DEBOUNCE_MS', {
  providedIn: 'root',
  factory: () => 300,
});

interface InstrumentQuery {
  page?: number;
  page_size?: number;
  search?: string;
  exchange?: string;
}

interface BuildJobResponse {
  job_id: string;
  build_id: string;
  status: string;
  message: string;
}

const EMPTY_STATS: UniverseStats = {
  total_exchanges: 0,
  total_instruments: 0,
  last_updated: '',
};

function emptyPage(query: InstrumentQuery): InstrumentList {
  return {
    items: [],
    total: 0,
    page: query.page ?? 1,
    page_size: query.page_size ?? 15,
  };
}

function toParams(query: InstrumentQuery): HttpParams {
  let params = new HttpParams();
  if (query.page !== undefined) params = params.set('page', String(query.page));
  if (query.page_size !== undefined) params = params.set('page_size', String(query.page_size));
  if (query.search) params = params.set('search', query.search);
  if (query.exchange) params = params.set('exchange', query.exchange);
  return params;
}

@Injectable({ providedIn: 'root' })
export class UniverseService {
  private readonly http = inject(HttpClient);
  private readonly apiBase = environment.apiUrl;
  private readonly debounceMs = inject(SEARCH_DEBOUNCE_MS);

  getStats(): Observable<UniverseStats> {
    return this.http
      .get<UniverseStats>(`${this.apiBase}universe/stats`)
      .pipe(catchError(() => of(EMPTY_STATS)));
  }

  getExchanges(): Observable<Exchange[]> {
    return this.http
      .get<Exchange[]>(`${this.apiBase}universe/exchanges`)
      .pipe(catchError(() => of<Exchange[]>([])));
  }

  getInstruments(query: InstrumentQuery): Observable<InstrumentList> {
    return this.http
      .get<InstrumentList>(`${this.apiBase}universe/instruments`, {
        params: toParams(query),
      })
      .pipe(catchError(() => of(emptyPage(query))));
  }

  buildUniverse(): Observable<{ job_id: string }> {
    return this.http.post<BuildJobResponse>(`${this.apiBase}universe/build`, {});
  }

  getBuildProgress(id: string): Observable<BuildProgress> {
    return this.http.get<BuildProgress>(
      `${this.apiBase}universe/build/${encodeURIComponent(id)}`,
    );
  }

  searchTickers(queries$: Observable<string>): Observable<InstrumentList> {
    return queries$.pipe(
      debounceTime(this.debounceMs),
      distinctUntilChanged(),
      switchMap((search) => this.getInstruments({ search })),
    );
  }
}
