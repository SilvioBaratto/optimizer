import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { TickerProfile, PriceHistory, AnalystRecommendation, InsiderTransaction, TickerNews } from '../models/yfinance.model';
import { MOCK_PROFILES, MOCK_PRICES, MOCK_RECOMMENDATIONS, MOCK_INSIDER_TRANSACTIONS, MOCK_NEWS } from '../mocks/mock-data';

@Injectable({ providedIn: 'root' })
export class YfinanceService {
  getProfile(id: string): Observable<TickerProfile | null> {
    return of(MOCK_PROFILES[id] ?? null).pipe(delay(300));
  }

  getPrices(id: string): Observable<PriceHistory[]> {
    void id;
    return of(MOCK_PRICES).pipe(delay(500));
  }

  getRecommendations(id: string): Observable<AnalystRecommendation[]> {
    void id;
    return of(MOCK_RECOMMENDATIONS).pipe(delay(300));
  }

  getInsiderTransactions(id: string): Observable<InsiderTransaction[]> {
    void id;
    return of(MOCK_INSIDER_TRANSACTIONS).pipe(delay(300));
  }

  getNews(id: string): Observable<TickerNews[]> {
    void id;
    return of(MOCK_NEWS).pipe(delay(300));
  }
}
