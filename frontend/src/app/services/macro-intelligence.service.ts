import { Injectable, inject } from '@angular/core';
import { Observable, from } from 'rxjs';
import { MockFetchService } from './mock-fetch.service';
import type {
  MacroCalibrationResponse,
  FredObservationPoint,
  MacroNewsItem,
  MacroNewsTheme,
  CountryMacroData,
  BondYieldSnapshot,
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

@Injectable({ providedIn: 'root' })
export class MacroIntelligenceService {
  private readonly mockFetch = inject(MockFetchService);

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
    return from(this.mockFetch.fetch(MOCK_BOND_YIELDS_TODAY));
  }

  getBondYields1YAgo(): Observable<BondYieldSnapshot[]> {
    return from(this.mockFetch.fetch(MOCK_BOND_YIELDS_1Y_AGO));
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

  getCompositeHistory(): Observable<{ month: string; score: number }[]> {
    return from(this.mockFetch.fetch(MOCK_COMPOSITE_HISTORY));
  }
}
