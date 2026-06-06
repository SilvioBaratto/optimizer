import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { PortfolioContextService } from '../../core/services/portfolio-context.service';
import { environment } from '../../../environments/environment';

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
});
