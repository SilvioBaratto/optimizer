import { Injectable, inject } from '@angular/core';
import { Observable, catchError, map, of } from 'rxjs';

import { PortfolioApiService } from '../core/services/portfolio-api.service';

export interface WeightResolution {
  source: 'stored' | 'equal-weight';
  weights: Record<string, number>;
}

const EQUAL_WEIGHT_FALLBACK: WeightResolution = { source: 'equal-weight', weights: {} };

/**
 * Resolves the weight map for a backtest run from the latest optimization
 * snapshot. Uses the portfolio ID directly as the lookup key so it resolves
 * without needing the portfolio list to be loaded.
 *
 * Equal-weight fallback (weights: {}) is used when:
 *  - portfolioId is null
 *  - the snapshot endpoint returns an error (including 404)
 *  - the snapshot type is not 'optimization'
 *  - the snapshot weights object is null or empty
 */
@Injectable({ providedIn: 'root' })
export class BacktestWeightResolverService {
  private readonly portfolioApi = inject(PortfolioApiService);

  resolve(portfolioId: string | null): Observable<WeightResolution> {
    if (!portfolioId) return of(EQUAL_WEIGHT_FALLBACK);

    return this.portfolioApi.getLatestSnapshot(portfolioId).pipe(
      map((snapshot): WeightResolution => {
        if (snapshot.snapshot_type !== 'optimization') return EQUAL_WEIGHT_FALLBACK;
        const weights = snapshot.weights ?? {};
        return Object.keys(weights).length > 0
          ? { source: 'stored', weights }
          : EQUAL_WEIGHT_FALLBACK;
      }),
      catchError(() => of(EQUAL_WEIGHT_FALLBACK)),
    );
  }
}
