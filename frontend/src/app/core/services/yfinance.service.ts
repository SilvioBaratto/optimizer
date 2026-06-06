import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';

import { environment } from '../../../environments/environment';
import {
  AnalystPriceTarget,
  AnalystRecommendation,
  Dividend,
  FinancialStatement,
  InsiderTransaction,
  InstitutionalHolder,
  MutualFundHolder,
  PriceHistory,
  StockSplit,
  TickerNews,
  TickerProfile,
} from '../models/yfinance.model';

export interface PriceQuery {
  startDate?: string;
  endDate?: string;
  limit?: number;
}

export interface FinancialsQuery {
  statementType?: string;
  periodType?: string;
}

function priceParams(query?: PriceQuery): HttpParams | undefined {
  if (!query) return undefined;
  let params = new HttpParams();
  if (query.startDate) params = params.set('start_date', query.startDate);
  if (query.endDate) params = params.set('end_date', query.endDate);
  if (query.limit !== undefined) params = params.set('limit', String(query.limit));
  return params.keys().length ? params : undefined;
}

function financialsParams(query?: FinancialsQuery): HttpParams | undefined {
  if (!query) return undefined;
  let params = new HttpParams();
  if (query.statementType) params = params.set('statement_type', query.statementType);
  if (query.periodType) params = params.set('period_type', query.periodType);
  return params.keys().length ? params : undefined;
}

@Injectable({ providedIn: 'root' })
export class YfinanceService {
  private readonly http = inject(HttpClient);
  private readonly base = `${environment.apiUrl}yfinance-data/instruments`;

  getProfile = (id: string): Observable<TickerProfile | null> => this.one<TickerProfile>(id, 'profile');
  getPrices = (id: string, q?: PriceQuery): Observable<PriceHistory[]> => this.list<PriceHistory>(id, 'prices', priceParams(q));
  getFinancials = (id: string, q?: FinancialsQuery): Observable<FinancialStatement[]> => this.list<FinancialStatement>(id, 'financials', financialsParams(q));
  getDividends = (id: string): Observable<Dividend[]> => this.list<Dividend>(id, 'dividends');
  getSplits = (id: string): Observable<StockSplit[]> => this.list<StockSplit>(id, 'splits');
  getRecommendations = (id: string): Observable<AnalystRecommendation[]> => this.list<AnalystRecommendation>(id, 'recommendations');
  getPriceTargets = (id: string): Observable<AnalystPriceTarget | null> => this.one<AnalystPriceTarget>(id, 'price-targets');
  getInstitutionalHolders = (id: string): Observable<InstitutionalHolder[]> => this.list<InstitutionalHolder>(id, 'institutional-holders');
  getMutualfundHolders = (id: string): Observable<MutualFundHolder[]> => this.list<MutualFundHolder>(id, 'mutualfund-holders');
  getInsiderTransactions = (id: string): Observable<InsiderTransaction[]> => this.list<InsiderTransaction>(id, 'insider-transactions');
  getNews = (id: string): Observable<TickerNews[]> => this.list<TickerNews>(id, 'news');

  private one<T>(id: string, path: string): Observable<T | null> {
    return this.http
      .get<T>(`${this.base}/${encodeURIComponent(id)}/${path}`)
      .pipe(catchError(() => of<T | null>(null)));
  }

  private list<T>(id: string, path: string, params?: HttpParams): Observable<T[]> {
    return this.http
      .get<T[]>(`${this.base}/${encodeURIComponent(id)}/${path}`, params ? { params } : {})
      .pipe(catchError(() => of<T[]>([])));
  }
}
