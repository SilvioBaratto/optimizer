import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { MacroIntelligenceComponent } from './macro-intelligence';

describe('MacroIntelligenceComponent.parseMacroSummary (issue #439)', () => {
  let component: MacroIntelligenceComponent;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
    component = TestBed.createComponent(MacroIntelligenceComponent).componentInstance;
  });

  afterEach(() => {
    // Drain any in-flight requests fired by the constructor's loadData()
    for (const req of http.match(() => true)) req.flush({});
    http.verify();
  });

  describe('null-safety guards', () => {
    it('returns [] when summary is undefined (does not throw)', () => {
      expect(() => component.parseMacroSummary(undefined)).not.toThrow();
      expect(component.parseMacroSummary(undefined)).toEqual([]);
    });

    it('returns [] when summary is null (does not throw)', () => {
      expect(() => component.parseMacroSummary(null)).not.toThrow();
      expect(component.parseMacroSummary(null)).toEqual([]);
    });

    it('returns [] when summary is the empty string', () => {
      expect(component.parseMacroSummary('')).toEqual([]);
    });

    it('returns [] when summary is whitespace-only', () => {
      expect(component.parseMacroSummary('   \n\t  ')).toEqual([]);
    });
  });

  describe('error handling (issue #851)', () => {
    it('when every data fetch fails, panels reset to empty and the page is not stuck loading', () => {
      // The constructor fired loadData(); fail all in-flight requests.
      for (const req of http.match(() => true)) {
        req.error(new ProgressEvent('network'), { status: 500 });
      }
      expect(component.fredPmi()).toEqual([]);
      expect(component.newsItems()).toEqual([]);
      expect(component.countryData()).toEqual([]);
      expect(component.macroCalibration()).toBeNull();
      // getNews owns the skeleton — a failed news call must clear isLoading.
      expect(component.isLoading()).toBe(false);
    });
  });

  describe('parsing of well-formed and malformed segments', () => {
    it('parses a single label:value segment', () => {
      const rows = component.parseMacroSummary('PMI: 51.2');
      expect(rows).toEqual([{ label: 'PMI', value: '51.2' }]);
    });

    it('parses multiple pipe-separated segments', () => {
      const rows = component.parseMacroSummary('PMI: 51.2 | 2s10s: -0.34 | HY: 350bp');
      expect(rows).toEqual([
        { label: 'PMI', value: '51.2' },
        { label: '2s10s', value: '-0.34' },
        { label: 'HY', value: '350bp' },
      ]);
    });

    it('treats a malformed segment lacking a colon as a label-only entry with empty value', () => {
      const rows = component.parseMacroSummary('OnlyLabel');
      expect(rows).toEqual([{ label: 'OnlyLabel', value: '' }]);
    });

    it('mixes well-formed and malformed segments without throwing', () => {
      const rows = component.parseMacroSummary('PMI: 51.2 | NoColonHere | YIELD: 4.25%');
      expect(rows).toEqual([
        { label: 'PMI', value: '51.2' },
        { label: 'NoColonHere', value: '' },
        { label: 'YIELD', value: '4.25%' },
      ]);
    });
  });
});
