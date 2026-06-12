import { Injectable, inject } from '@angular/core';
import { Observable, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

import { PortfolioApiService } from './portfolio-api.service';

@Injectable({ providedIn: 'root' })
export class TickerSeedingService {
  private readonly portfolioApi = inject(PortfolioApiService);

  seedFromPortfolio(
    portfolioName: string | null,
    defaultTickers: string[],
  ): Observable<string[]> {
    if (portfolioName == null) {
      return of(defaultTickers);
    }
    return this.portfolioApi.getLatestSnapshot(portfolioName).pipe(
      map((snapshot) => {
        const keys = Object.keys(snapshot.weights ?? {});
        return keys.length ? keys : defaultTickers;
      }),
      catchError(() => of(defaultTickers)),
    );
  }
}
