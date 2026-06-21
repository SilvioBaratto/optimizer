/**
 * Contract tests — issue #1049, criterion 1
 *
 * "A Save control creates a named portfolio via PortfolioApiService.create;
 *  success/error surfaced; duplicate/empty-name handled."
 *
 * Tests are source-blind: derived from the acceptance criterion only, not from
 * reading the implementation.  Every test is RED today and will turn GREEN once
 * the implementation is complete.
 *
 * Assumption: the component's public surface exposes
 *   savePortfolio(name: string): void
 *   saveSuccess: Signal<boolean>
 *   saveError:   Signal<string | null>
 *   nameError:   Signal<string | null>
 *
 * These are the minimal signal-based names consistent with Angular 21 conventions
 * and the criterion text.  If the implementation uses different names the tests
 * will fail at compile time (red), guiding the implementer to honour the contract.
 */

import { NO_ERRORS_SCHEMA } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { PortfolioDto } from '../app/core/models/portfolio-api.model';
import { PortfolioBuilderComponent } from '../app/portfolio-builder/portfolio-builder';
import { PortfolioApiService } from '../app/core/services/portfolio-api.service';

function makePortfolioDto(name = 'My Portfolio'): PortfolioDto {
  return {
    id: '1',
    name,
    description: null,
    currency: 'USD',
    benchmark_ticker: 'SPY',
    is_active: true,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };
}

describe('PortfolioBuilder — Save Control', () => {
  let apiSpy: jasmine.SpyObj<PortfolioApiService>;

  beforeEach(() => {
    apiSpy = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', ['create']);

    TestBed.configureTestingModule({
      imports: [PortfolioBuilderComponent],
      providers: [{ provide: PortfolioApiService, useValue: apiSpy }],
      schemas: [NO_ERRORS_SCHEMA],
    });
  });

  // ── Happy path ────────────────────────────────────────────────────────────

  it('when a valid name is submitted, PortfolioApiService.create is called with that name', () => {
    apiSpy.create.and.returnValue(of(makePortfolioDto()));
    const comp = TestBed.createComponent(PortfolioBuilderComponent).componentInstance;

    comp.savePortfolio('My Portfolio');

    expect(apiSpy.create).toHaveBeenCalledOnceWith(
      jasmine.objectContaining({ name: 'My Portfolio' })
    );
  });

  it('when PortfolioApiService.create resolves, saveSuccess signal becomes true', () => {
    apiSpy.create.and.returnValue(of(makePortfolioDto()));
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;

    comp.savePortfolio('My Portfolio');
    fixture.detectChanges();

    expect(comp.saveSuccess()).toBeTrue();
  });

  // ── Error path ────────────────────────────────────────────────────────────

  it('when PortfolioApiService.create throws a generic error, saveError signal is set', () => {
    apiSpy.create.and.returnValue(throwError(() => new Error('Internal Server Error')));
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;

    comp.savePortfolio('My Portfolio');
    fixture.detectChanges();

    expect(comp.saveError()).toBeTruthy();
  });

  it('when API returns HTTP 409, duplicate-name error message is surfaced', () => {
    /** 409 Conflict = portfolio name already taken; the word "duplicate" must appear. */
    apiSpy.create.and.returnValue(throwError(() => ({ status: 409 })));
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;

    comp.savePortfolio('Taken Name');
    fixture.detectChanges();

    const error: string = comp.saveError() ?? '';
    expect(error.toLowerCase()).toContain('duplicate');
  });

  // ── Validation path ───────────────────────────────────────────────────────

  it('when name is empty string, PortfolioApiService.create is never called', () => {
    const comp = TestBed.createComponent(PortfolioBuilderComponent).componentInstance;

    comp.savePortfolio('');

    expect(apiSpy.create).not.toHaveBeenCalled();
  });

  it('when name is empty string, nameError signal is set', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;

    comp.savePortfolio('');
    fixture.detectChanges();

    expect(comp.nameError()).toBeTruthy();
  });

  it('when name is whitespace only, PortfolioApiService.create is never called', () => {
    const comp = TestBed.createComponent(PortfolioBuilderComponent).componentInstance;

    comp.savePortfolio('   ');

    expect(apiSpy.create).not.toHaveBeenCalled();
  });

  it('when name is whitespace only, nameError signal is set', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;

    comp.savePortfolio('   ');
    fixture.detectChanges();

    expect(comp.nameError()).toBeTruthy();
  });
});
