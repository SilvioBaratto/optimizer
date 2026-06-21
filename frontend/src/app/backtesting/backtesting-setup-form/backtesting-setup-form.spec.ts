/**
 * BacktestingSetupFormComponent — contract specification tests
 * Issue #1050: feat(backtesting): local setup form — portfolio picker + date-range + benchmark
 *
 * Criteria covered:
 *  [UNIT] FormGroup with portfolio, period, benchmark controls (criterion 1)
 *  [UNIT] Emits run config via runConfig output (criterion 2)
 *  [UNIT] No context-bar dependency; defaults to period='1Y', benchmark='SPY' (criterion 3)
 */

import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import { BacktestingSetupFormComponent } from './backtesting-setup-form';
import { PortfolioApiService } from '../../core/services/portfolio-api.service';
import type { PortfolioListResponseDto } from '../../core/models/portfolio-api.model';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SAVED_PORTFOLIOS: PortfolioListResponseDto = {
  items: [
    {
      id: 'p1', name: 'Growth Portfolio', description: null,
      currency: 'USD', benchmark_ticker: 'SPY', is_active: true,
      created_at: '', updated_at: '',
    },
    {
      id: 'p2', name: 'Balanced Portfolio', description: null,
      currency: 'USD', benchmark_ticker: 'SPY', is_active: true,
      created_at: '', updated_at: '',
    },
  ],
  total: 2,
};

const portfolioApiSpy = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', {
  list: of(SAVED_PORTFOLIOS),
});

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('BacktestingSetupFormComponent', () => {
  let fixture: ComponentFixture<BacktestingSetupFormComponent>;
  let component: BacktestingSetupFormComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BacktestingSetupFormComponent],
      providers: [
        provideZonelessChangeDetection(),
        { provide: PortfolioApiService, useValue: portfolioApiSpy },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(BacktestingSetupFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  // ── Criterion 1: FormGroup with portfolio picker, period/date-range, benchmark ─────

  it('when setup form is initialized, it exposes a FormGroup via setupForm', () => {
    expect(component.setupForm).toBeTruthy();
  });

  it('when setup form is initialized, form group contains a portfolio control', () => {
    expect(component.setupForm.get('portfolio')).not.toBeNull();
  });

  it('when setup form is initialized, form group contains a period control for the date-range picker', () => {
    expect(component.setupForm.get('period')).not.toBeNull();
  });

  it('when setup form is initialized, form group contains a benchmark control', () => {
    expect(component.setupForm.get('benchmark')).not.toBeNull();
  });

  it('when setup form is initialized, it loads saved portfolios so the picker can list them', () => {
    expect(component.portfolios().length).toBeGreaterThan(0);
  });

  // ── Criterion 2: Form emits run config consumed by the run wiring ─────────────────

  it('when run is triggered with valid form values, run config containing portfolio period and benchmark is emitted', () => {
    const emitted: unknown[] = [];
    component.runConfig.subscribe((cfg) => emitted.push(cfg));

    component.setupForm.setValue({ portfolio: 'p1', period: '6M', benchmark: 'QQQ' });
    component.onRun();

    expect(emitted.length).toBe(1);
    expect(emitted[0]).toEqual(
      jasmine.objectContaining({ portfolio: 'p1', period: '6M', benchmark: 'QQQ' }),
    );
  });

  it('when run is triggered a second time with different values, the updated run config is emitted', () => {
    const emitted: unknown[] = [];
    component.runConfig.subscribe((cfg) => emitted.push(cfg));

    component.setupForm.setValue({ portfolio: 'p1', period: '1Y', benchmark: 'SPY' });
    component.onRun();

    component.setupForm.setValue({ portfolio: 'p2', period: '3Y', benchmark: 'QQQ' });
    component.onRun();

    expect(emitted.length).toBe(2);
    expect(emitted[1]).toEqual(
      jasmine.objectContaining({ portfolio: 'p2', period: '3Y', benchmark: 'QQQ' }),
    );
  });

  it('when portfolio is empty, onRun does not emit', () => {
    const emitted: unknown[] = [];
    component.runConfig.subscribe((cfg) => emitted.push(cfg));

    component.setupForm.patchValue({ portfolio: '', period: '1Y', benchmark: 'SPY' });
    component.onRun();

    expect(emitted.length).toBe(0);
  });

  // ── Criterion 3: No global context-bar dependency; sensible defaults ──────────────

  it('when setup form is created without any global context data, period defaults to 1Y', () => {
    expect(component.setupForm.get('period')?.value).toBe('1Y');
  });

  it('when setup form is created without any global context data, benchmark defaults to SPY', () => {
    expect(component.setupForm.get('benchmark')?.value).toBe('SPY');
  });

  it('when setup form is created without any global context data, portfolio control starts empty', () => {
    const portfolioValue = component.setupForm.get('portfolio')?.value;
    expect(portfolioValue == null || portfolioValue === '').toBeTrue();
  });
});
