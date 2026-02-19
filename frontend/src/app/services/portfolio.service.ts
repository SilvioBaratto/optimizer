import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { StrategyInfo, PortfolioOptimizeRequest, PortfolioResult } from '../models/portfolio.model';
import { MOCK_STRATEGIES, MOCK_PORTFOLIO_RESULT } from '../mocks/mock-data';

@Injectable({ providedIn: 'root' })
export class PortfolioService {
  getStrategies(): Observable<StrategyInfo[]> {
    return of(MOCK_STRATEGIES).pipe(delay(200));
  }

  runOptimization(_req: PortfolioOptimizeRequest): Observable<PortfolioResult> {
    return of(MOCK_PORTFOLIO_RESULT).pipe(delay(1500));
  }
}
