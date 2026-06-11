import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const OPTIMIZE_URL = `${environment.apiUrl}optimize`;
const RUN: { optimizerType: 'mean_risk'; config: Record<string, unknown> } = {
  optimizerType: 'mean_risk',
  config: {},
};

describe('OptimizationStudioComponent', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;
  let comp: OptimizationStudioComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({ imports: [OptimizationStudioComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(OptimizationStudioComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges(); // runs the dateRange effect; no HTTP fires on init
  });

  afterEach(() => {
    http.verify();
    localStorage.clear();
  });

  it('when a sync optimize resolves, the run result is set and polling is off', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ id: 'run-1', weights: {} });
    expect(comp.hasResult()).toBe(true);
    expect(comp.isPolling()).toBe(false);
  });

  it('when optimize returns a job_id, polling begins', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ job_id: 'j1', run_id: 'r1', status: 'pending' });
    expect(comp.isPolling()).toBe(true);
  });

  it('when the job completes, the run is fetched and polling stops', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ job_id: 'j1', run_id: 'r1', status: 'pending' });
    comp.onJobCompleted('run-1');
    http.expectOne(`${OPTIMIZE_URL}/run-1`).flush({ id: 'run-1', weights: {} });
    expect(comp.hasResult()).toBe(true);
    expect(comp.isPolling()).toBe(false);
  });

  it('when no portfolio is selected, applying weights errors without a request', () => {
    expect(ctx.currentPortfolioId()).toBeNull();
    comp.onApplyWeights({ AAPL: 1 });
    expect(comp.applyStatus()).toBe('error');
    expect(comp.applyError()).toContain('No active portfolio');
    http.expectNone(() => true);
  });

  it('when a pipeline is saved then loaded, its state round-trips via localStorage', () => {
    comp.savePipeline();
    expect(comp.pipelineStatus()).toBe('Pipeline saved.');
    comp.loadPipeline();
    expect(comp.pipelineStatus()).toContain('Pipeline loaded');
  });

  it('when optimize errors, the run error is recorded and running stops', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runError()).toBeTruthy();
    expect(comp.isRunning()).toBe(false);
  });

  it('onRunPipeline is a no-op while a run is already in flight', () => {
    comp.isRunning.set(true);
    comp.onRunPipeline(RUN);
    expect(http.match(OPTIMIZE_URL).length).toBe(0);
  });

  it('onJobFailed records the error with a default fallback', () => {
    comp.onJobFailed('boom');
    expect(comp.runError()).toBe('boom');
    expect(comp.isRunning()).toBe(false);
    comp.onJobFailed('');
    expect(comp.runError()).toBe('Job failed');
  });

  it('when fetching a completed run fails, the error is recorded', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ job_id: 'j1', run_id: 'r1', status: 'pending' });
    comp.onJobCompleted('r1');
    http
      .expectOne(`${OPTIMIZE_URL}/r1`)
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runError()).toBeTruthy();
  });

  it('applying weights to a selected portfolio saves a snapshot and reports success', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onApplyWeights({ AAPL: 1 });
    expect(comp.applyStatus()).toBe('saving');
    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush({ items: [{ id: 'pf-1', name: 'My Port' }] });
    http
      .expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'))
      .flush({ id: 'snap-1' });
    expect(comp.applyStatus()).toBe('success');
    expect(comp.appliedPortfolioName()).toBe('My Port');
  });

  it('applying weights surfaces an error when the snapshot fails', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onApplyWeights({ AAPL: 1 });
    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush({ items: [{ id: 'pf-1', name: 'My Port' }] });
    http
      .expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'))
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.applyStatus()).toBe('error');
    expect(comp.applyError()).toBeTruthy();
  });

  it('onNodeSelect toggles the active node', () => {
    comp.onNodeSelect('p2');
    expect(comp.activeNode()).toBe('p2');
    comp.onNodeSelect('p2');
    expect(comp.activeNode()).toBeNull();
  });

  it('retry clears the run error and stops loading', () => {
    comp.runError.set('x');
    comp.retry();
    expect(comp.runError()).toBeNull();
    expect(comp.isLoading()).toBe(false);
  });

  it('loadData clears the error and loading flags', () => {
    comp.hasError.set(true);
    comp.loadData();
    expect(comp.hasError()).toBe(false);
    expect(comp.isLoading()).toBe(false);
  });

  it('openReportModal does not throw', () => {
    expect(() => comp.openReportModal()).not.toThrow();
  });
});
