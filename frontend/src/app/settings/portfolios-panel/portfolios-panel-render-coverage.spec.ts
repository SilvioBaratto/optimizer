/**
 * portfolios-panel-render-coverage.spec.ts — issue #942
 *
 * Template-branch coverage for PortfoliosPanelComponent. Services are spied so
 * the create / seed-and-create flows are deterministic (the seed poll is
 * otherwise interval-driven). Drives the list three-way, the create form, the
 * isFormValid boundary and every submit branch.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { configureTestBed, makePortfolioDto } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { PortfoliosPanelComponent } from './portfolios-panel';
import { PortfolioApiService } from '../../core/services/portfolio-api.service';
import { MarketService } from '../../core/services/market.service';
import { ReferenceIndexService } from '../../core/services/reference-index.service';

interface ApiSpy { list: jasmine.Spy; create: jasmine.Spy; }
interface MarketSpy { getIndices: jasmine.Spy; }
interface RefSpy { startSeed: jasmine.Spy; pollUntilDone: jasmine.Spy; }

function makeSpies(): { api: ApiSpy; market: MarketSpy; refIndex: RefSpy } {
  const api = jasmine.createSpyObj<ApiSpy>('PortfolioApiService', ['list', 'create']);
  api.list.and.returnValue(of({ items: [] }));
  api.create.and.returnValue(of(makePortfolioDto()));
  const market = jasmine.createSpyObj<MarketSpy>('MarketService', ['getIndices']);
  market.getIndices.and.returnValue(of({ indices: [{ ticker: 'SPY', name: 'S&P 500' }], total: 1 }));
  const refIndex = jasmine.createSpyObj<RefSpy>('ReferenceIndexService', ['startSeed', 'pollUntilDone']);
  refIndex.startSeed.and.returnValue(of({ job_id: 's1' }));
  refIndex.pollUntilDone.and.returnValue(of({ status: 'completed', current: 1, total: 1, error: null }));
  return { api, market, refIndex };
}

describe('PortfoliosPanelComponent — render-coverage (#942)', () => {
  let fixture: ComponentFixture<PortfoliosPanelComponent>;
  let comp: PortfoliosPanelComponent;
  let el: HTMLElement;
  let spies: { api: ApiSpy; market: MarketSpy; refIndex: RefSpy };

  async function mount(): Promise<void> {
    await configureTestBed({
      imports: [PortfoliosPanelComponent],
      withHttp: false,
      providers: [
        ICON_PROVIDER,
        { provide: PortfolioApiService, useValue: spies.api },
        { provide: MarketService, useValue: spies.market },
        { provide: ReferenceIndexService, useValue: spies.refIndex },
      ],
    });
    fixture = TestBed.createComponent(PortfoliosPanelComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  }

  beforeEach(() => {
    spies = makeSpies();
  });

  function validForm(over: Partial<{ name: string; currency: string; benchmark_ticker: string }> = {}) {
    return { name: 'My Portfolio', description: '', currency: 'USD', benchmark_ticker: 'SPY', ...over };
  }

  describe('list three-way', () => {
    it('when loading with no portfolios, the loading message is shown', async () => {
      await mount();
      comp.loading.set(true);
      comp.portfolios.set([]);
      fixture.detectChanges();
      expect(el.textContent).toContain('Loading portfolios');
    });

    it('when there are no portfolios, the empty message is shown', async () => {
      await mount();
      expect(el.textContent).toContain('No portfolios yet');
    });

    it('when portfolios exist, the data table renders', async () => {
      await mount();
      comp.portfolios.set([makePortfolioDto()]);
      fixture.detectChanges();
      expect(el.querySelector('app-data-table')).not.toBeNull();
    });
  });

  describe('create form and validity', () => {
    it('when showCreate is toggled on, the form and currency options render', async () => {
      await mount();
      comp.toggleCreate();
      fixture.detectChanges();
      expect(el.querySelector('[data-testid="create-portfolio-form"]')).not.toBeNull();
      expect(el.querySelectorAll('select[name="currency"] option').length).toBe(7);
    });

    it('isFormValid is true at the 2-char name boundary and false just below', async () => {
      await mount();
      comp.form.set(validForm({ name: 'ab' }));
      expect(comp.isFormValid()).toBe(true);
      comp.form.set(validForm({ name: 'a' }));
      expect(comp.isFormValid()).toBe(false);
      comp.form.set(validForm({ currency: 'US' }));
      expect(comp.isFormValid()).toBe(false);
    });

    it('when creating, the submit button reads Creating and is disabled', async () => {
      await mount();
      comp.showCreate.set(true);
      comp.creating.set(true);
      fixture.detectChanges();
      const btn = el.querySelector('[data-testid="submit-portfolio"]') as HTMLButtonElement;
      expect(btn.disabled).toBe(true);
      expect(btn.textContent).toContain('Creating');
    });

    it('when createError or loadError are set, the banners are shown', async () => {
      await mount();
      comp.showCreate.set(true);
      comp.createError.set('create boom');
      comp.loadError.set('load boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('create boom');
      expect(el.textContent).toContain('load boom');
    });

    it('updateField patches a single form field', async () => {
      await mount();
      comp.updateField('name', 'Hello');
      expect(comp.form().name).toBe('Hello');
    });
  });

  describe('submit branches', () => {
    it('a valid submit with a known benchmark creates directly', async () => {
      await mount();
      comp.form.set(validForm()); // SPY is known
      comp.submit();
      expect(spies.api.create).toHaveBeenCalled();
      expect(spies.refIndex.startSeed).not.toHaveBeenCalled();
      expect(comp.creating()).toBe(false);
      expect(comp.showCreate()).toBe(false);
    });

    it('an invalid submit is a no-op', async () => {
      await mount();
      comp.form.set(validForm({ name: '' }));
      comp.submit();
      expect(spies.api.create).not.toHaveBeenCalled();
    });

    it('a create failure records the error', async () => {
      await mount();
      spies.api.create.and.returnValue(throwError(() => new Error('nope')));
      comp.form.set(validForm());
      comp.submit();
      expect(comp.createError()).toBe('nope');
      expect(comp.creating()).toBe(false);
    });

    it('an unknown benchmark seeds then creates on completion', async () => {
      await mount();
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(spies.refIndex.startSeed).toHaveBeenCalledWith(['ZZZZ']);
      expect(spies.api.create).toHaveBeenCalled();
    });

    it('a failed seed records the error', async () => {
      await mount();
      spies.refIndex.pollUntilDone.and.returnValue(
        of({ status: 'failed', current: 0, total: 0, error: 'seed boom' }),
      );
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(comp.createError()).toBe('seed boom');
      expect(spies.api.create).not.toHaveBeenCalled();
    });

    it('seed progress updates the seeding message', async () => {
      await mount();
      spies.refIndex.pollUntilDone.and.returnValue(
        of({ status: 'running', current: 2, total: 5, error: null }),
      );
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(comp.seedingMessage()).toContain('2/5');
    });

    it('a seed start failure records the error', async () => {
      await mount();
      spies.refIndex.startSeed.and.returnValue(throwError(() => new Error('start boom')));
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(comp.createError()).toBe('start boom');
    });
  });

  describe('load failures', () => {
    it('a refresh failure records the load error', async () => {
      await mount();
      spies.api.list.and.returnValue(throwError(() => new Error('list boom')));
      comp.refresh();
      expect(comp.loadError()).toBe('list boom');
      expect(comp.loading()).toBe(false);
    });

    it('an indices failure leaves all benchmarks treated as unknown', async () => {
      spies.market.getIndices.and.returnValue(throwError(() => new Error('idx boom')));
      await mount();
      // knownBenchmarks empty → even SPY seeds
      comp.form.set(validForm());
      comp.submit();
      expect(spies.refIndex.startSeed).toHaveBeenCalledWith(['SPY']);
    });
  });

  describe('edge formatting and default error messages', () => {
    it('an inactive portfolio renders "no" in the Active column', async () => {
      await mount();
      comp.portfolios.set([makePortfolioDto({ is_active: false })]);
      expect(comp.tableRows()[0]['is_active']).toBe('no');
    });

    it('an empty benchmark builds a payload without a benchmark and creates directly', async () => {
      await mount();
      comp.form.set(validForm({ benchmark_ticker: '' }));
      comp.submit();
      expect(spies.refIndex.startSeed).not.toHaveBeenCalled();
      const payload = spies.api.create.calls.mostRecent().args[0] as { benchmark_ticker?: string };
      expect(payload.benchmark_ticker).toBeUndefined();
    });

    it('a message-less refresh failure uses the default load error', async () => {
      await mount();
      spies.api.list.and.returnValue(throwError(() => ({})));
      comp.refresh();
      expect(comp.loadError()).toBe('Failed to load portfolios');
    });

    it('a message-less create failure uses the default create error', async () => {
      await mount();
      spies.api.create.and.returnValue(throwError(() => ({})));
      comp.form.set(validForm());
      comp.submit();
      expect(comp.createError()).toBe('Create failed');
    });

    it('a message-less seed start failure uses the default seed error', async () => {
      await mount();
      spies.refIndex.startSeed.and.returnValue(throwError(() => ({})));
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(comp.createError()).toBe('Failed to seed ZZZZ');
    });

    it('a failed seed without an error message uses the default seed-failed text', async () => {
      await mount();
      spies.refIndex.pollUntilDone.and.returnValue(of({ status: 'failed', current: 0, total: 0, error: null }));
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(comp.createError()).toBe('Seed failed for ZZZZ');
    });

    it('a seed polling failure uses the default polling error', async () => {
      await mount();
      spies.refIndex.pollUntilDone.and.returnValue(throwError(() => ({})));
      comp.form.set(validForm({ benchmark_ticker: 'ZZZZ' }));
      comp.submit();
      expect(comp.createError()).toBe('Seed polling failed');
    });
  });
});
