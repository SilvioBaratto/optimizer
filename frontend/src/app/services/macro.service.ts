import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { EconomicIndicator, BondYield, CountryMacroSummary } from '../models/macro.model';
import { MOCK_INDICATORS, MOCK_BOND_YIELDS, MOCK_COUNTRY_SUMMARIES } from '../mocks/mock-data';

@Injectable({ providedIn: 'root' })
export class MacroService {
  getEconomicIndicators(): Observable<EconomicIndicator[]> {
    return of(MOCK_INDICATORS).pipe(delay(400));
  }

  getBondYields(): Observable<BondYield[]> {
    return of(MOCK_BOND_YIELDS).pipe(delay(350));
  }

  getCountrySummary(country: string): Observable<CountryMacroSummary | undefined> {
    return of(MOCK_COUNTRY_SUMMARIES.find(c => c.country === country)).pipe(delay(300));
  }

  getCountrySummaries(): Observable<CountryMacroSummary[]> {
    return of(MOCK_COUNTRY_SUMMARIES).pipe(delay(350));
  }
}
