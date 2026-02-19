import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { UniverseStats, Exchange, InstrumentList } from '../models/universe.model';
import { MOCK_STATS, MOCK_EXCHANGES, MOCK_INSTRUMENTS } from '../mocks/mock-data';

@Injectable({ providedIn: 'root' })
export class UniverseService {
  getStats(): Observable<UniverseStats> {
    return of(MOCK_STATS).pipe(delay(300));
  }

  getExchanges(): Observable<Exchange[]> {
    return of(MOCK_EXCHANGES).pipe(delay(300));
  }

  getInstruments(params: { page?: number; page_size?: number; search?: string; exchange?: string }): Observable<InstrumentList> {
    let filtered = [...MOCK_INSTRUMENTS];

    if (params.search) {
      const q = params.search.toLowerCase();
      filtered = filtered.filter(i => i.ticker.toLowerCase().includes(q) || i.name.toLowerCase().includes(q));
    }
    if (params.exchange) {
      filtered = filtered.filter(i => i.exchange === params.exchange);
    }

    const page = params.page ?? 1;
    const pageSize = params.page_size ?? 15;
    const start = (page - 1) * pageSize;

    return of({
      items: filtered.slice(start, start + pageSize),
      total: filtered.length,
      page,
      page_size: pageSize,
    }).pipe(delay(300));
  }
}
