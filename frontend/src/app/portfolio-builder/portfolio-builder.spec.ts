import { TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { UniverseService } from '../core/services/universe.service';
import { YfinanceService } from '../core/services/yfinance.service';
import { MacroIntelligenceService } from '../macro-intelligence/macro-intelligence.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';

function makeUniverseSpy(): jasmine.SpyObj<UniverseService> {
  const spy = jasmine.createSpyObj<UniverseService>('UniverseService', ['searchTickers']);
  spy.searchTickers.and.returnValue(of({ items: [], total: 0, page: 1, page_size: 15 }));
  return spy;
}

function makeYfinanceSpy(): jasmine.SpyObj<YfinanceService> {
  const spy = jasmine.createSpyObj<YfinanceService>('YfinanceService', ['getProfile']);
  spy.getProfile.and.returnValue(of(null));
  return spy;
}

function makeMacroSpy(): jasmine.SpyObj<MacroIntelligenceService> {
  const spy = jasmine.createSpyObj<MacroIntelligenceService>(
    'MacroIntelligenceService', ['getMacroCalibration'],
  );
  spy.getMacroCalibration.and.returnValue(of(null));
  return spy;
}

function makeApiSpy(): jasmine.SpyObj<PortfolioApiService> {
  return jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', ['create']);
}

function setup(): HTMLElement {
  TestBed.configureTestingModule({
    imports: [PortfolioBuilderComponent],
    schemas: [NO_ERRORS_SCHEMA],
    providers: [
      provideZonelessChangeDetection(),
      { provide: UniverseService,          useValue: makeUniverseSpy() },
      { provide: YfinanceService,          useValue: makeYfinanceSpy() },
      { provide: MacroIntelligenceService, useValue: makeMacroSpy()    },
      { provide: PortfolioApiService,      useValue: makeApiSpy()       },
    ],
  });
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('PortfolioBuilderComponent', () => {
  it('when instantiated, component creates without error', () => {
    TestBed.configureTestingModule({
      imports: [PortfolioBuilderComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        { provide: UniverseService,          useValue: makeUniverseSpy() },
        { provide: YfinanceService,          useValue: makeYfinanceSpy() },
        { provide: MacroIntelligenceService, useValue: makeMacroSpy()    },
        { provide: PortfolioApiService,      useValue: makeApiSpy()       },
      ],
    });
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when rendered, the search region is present in the DOM', () => {
    const host = setup();
    expect(host.querySelector('[data-region="search"]')).not.toBeNull();
  });

  it('when rendered, the list region is present in the DOM', () => {
    const host = setup();
    expect(host.querySelector('[data-region="list"]')).not.toBeNull();
  });

  it('when rendered, app-asset-search is mounted inside the search region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="search"]');
    expect(region?.querySelector('app-asset-search')).not.toBeNull();
  });

  it('when rendered, app-asset-list is mounted inside the list region', () => {
    const host = setup();
    const region = host.querySelector('[data-region="list"]');
    expect(region?.querySelector('app-asset-list')).not.toBeNull();
  });

  it('when rendered, no BuilderStore-coupled components (app-stage-strip, app-inputs-pane, app-canvas-pane) appear in the DOM', () => {
    const host = setup();
    expect(host.querySelector('app-stage-strip')).toBeNull();
    expect(host.querySelector('app-inputs-pane')).toBeNull();
    expect(host.querySelector('app-canvas-pane')).toBeNull();
  });

  it('when rendered, no app-pipeline-stepper element is present', () => {
    const host = setup();
    expect(host.querySelector('app-pipeline-stepper')).toBeNull();
  });
});
