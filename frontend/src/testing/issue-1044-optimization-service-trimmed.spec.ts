/**
 * Issue #1044 — Criterion 5 [UNIT]
 *
 * "Now-orphaned service methods (LLM moments, views/pooling, load/save pipeline)
 *  trimmed from optimization.service.ts if unreferenced."
 *
 * Injects OptimizationService and asserts that the three categories of orphaned
 * methods are NOT present on the service instance. The exact method names are the
 * real ones from the original service (before deletion):
 *   - LLM moments: calibrateDelta, adaptFactorWeights, selectCovRegime
 *   - Views/pooling: generateViews, opinionPool, entropyPooling
 *   - Pipeline persistence: loadPipeline, savePipeline
 */

import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { OptimizationService } from '../app/optimization-studio/optimization.service';

function hasMethod(obj: unknown, name: string): boolean {
  return typeof (obj as Record<string, unknown>)[name] === 'function';
}

describe('OptimizationService — orphaned methods removed (issue #1044 criterion 5)', () => {
  let service: OptimizationService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        OptimizationService,
      ],
    });
    service = TestBed.inject(OptimizationService);
  });

  // ── LLM moments (moment-panel callers) ───────────────────────────────────

  it('when checking service, calibrateDelta is absent (LLM-moment method removed)', () => {
    expect(hasMethod(service, 'calibrateDelta'))
      .withContext('calibrateDelta was orphaned when moment-panel was deleted')
      .toBeFalse();
  });

  it('when checking service, adaptFactorWeights is absent (LLM-moment method removed)', () => {
    expect(hasMethod(service, 'adaptFactorWeights'))
      .withContext('adaptFactorWeights was orphaned when moment-panel was deleted')
      .toBeFalse();
  });

  it('when checking service, selectCovRegime is absent (LLM-moment method removed)', () => {
    expect(hasMethod(service, 'selectCovRegime'))
      .withContext('selectCovRegime was orphaned when moment-panel was deleted')
      .toBeFalse();
  });

  // ── Views / pooling (view-panel callers) ──────────────────────────────────

  it('when checking service, generateViews is absent (views/pooling method removed)', () => {
    expect(hasMethod(service, 'generateViews'))
      .withContext('generateViews was orphaned when view-panel was deleted')
      .toBeFalse();
  });

  it('when checking service, opinionPool is absent (views/pooling method removed)', () => {
    expect(hasMethod(service, 'opinionPool'))
      .withContext('opinionPool was orphaned when view-panel was deleted')
      .toBeFalse();
  });

  it('when checking service, entropyPooling is absent (views/pooling method removed)', () => {
    expect(hasMethod(service, 'entropyPooling'))
      .withContext('entropyPooling was orphaned when view-panel was deleted')
      .toBeFalse();
  });

  // ── Pipeline persistence (pipeline-builder callers) ───────────────────────

  it('when checking service, loadPipeline is absent (pipeline-load method removed)', () => {
    expect(hasMethod(service, 'loadPipeline'))
      .withContext('loadPipeline was orphaned when pipeline-builder was deleted')
      .toBeFalse();
  });

  it('when checking service, savePipeline is absent (pipeline-save method removed)', () => {
    expect(hasMethod(service, 'savePipeline'))
      .withContext('savePipeline was orphaned when pipeline-builder was deleted')
      .toBeFalse();
  });

  // ── Core methods that MUST remain ────────────────────────────────────────

  it('when checking service, optimize method is present (core run functionality)', () => {
    expect(hasMethod(service, 'optimize'))
      .withContext('optimize must remain — it is the core POST /optimize caller')
      .toBeTrue();
  });

  it('when checking service, applyWeightsToPortfolio is present (save-weights flow)', () => {
    expect(hasMethod(service, 'applyWeightsToPortfolio'))
      .withContext('applyWeightsToPortfolio must remain — it is called from Apply Weights button')
      .toBeTrue();
  });
});
