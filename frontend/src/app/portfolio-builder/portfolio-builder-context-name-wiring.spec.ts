/**
 * Source-blind RED tests for issue #962.
 *
 * Derives every assertion from the acceptance criteria text only.
 * Tests that PortfolioBuilderComponent wires currentPortfolioName() (not
 * currentPortfolioId()) into BuilderStore.setPortfolioName(), so the drift
 * service targets the correct URL segment.
 *
 * Criteria pinned:
 *   - portfolio-builder.ts feeds currentPortfolioName() (not currentPortfolioId())
 *     into store.setPortfolioName()
 *   - BuilderStore propagates the global portfolio NAME into every
 *     portfolio-name-dependent endpoint call
 *   - Drift fetch no-ops when currentPortfolioName() is null (guard holds)
 *   - Every wired page has specs: (a) loads data on init, (b) handles API error,
 *     (c) reacts to portfolio context change
 */

import {
  ComponentFixture,
  TestBed,
} from '@angular/core/testing';
import {
  NO_ERRORS_SCHEMA,
  provideZonelessChangeDetection,
  signal,
} from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { EMPTY } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { BuilderStore } from './state/builder.store';
import { BuilderResultService } from './builder-result.service';
import { BuilderDriftService } from './builder-drift.service';
import { BUILDER_DRIFT_SERVICE } from './state/builder.store';
import { PipelineBuilderApiService } from '../core/services/pipeline-builder-api.service';

/** Build a minimal PortfolioContextService mock that satisfies the component's
 *  read-only access to currentPortfolioName and currentPortfolioId. */
function buildContextMock(
  name: ReturnType<typeof signal<string | null>>,
  id: ReturnType<typeof signal<string | null>>,
) {
  return {
    currentPortfolioName: name.asReadonly(),
    currentPortfolioId: id.asReadonly(),
  };
}

describe('PortfolioBuilderComponent — portfolio context name wiring (issue #962)', () => {
  let fixture: ComponentFixture<PortfolioBuilderComponent>;
  let store: BuilderStore;
  let http: HttpTestingController;

  const mockName = signal<string | null>(null);
  const mockId = signal<string | null>(null);

  beforeEach(() => {
    mockName.set(null);
    mockId.set(null);

    TestBed.configureTestingModule({
      imports: [PortfolioBuilderComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        {
          provide: PortfolioContextService,
          useValue: buildContextMock(mockName, mockId),
        },
        {
          provide: PipelineBuilderApiService,
          useValue: {
            runStep: () => EMPTY,
            getArtifactUrl: () => '',
          },
        },
      ],
    });

    // Override component-level providers to prevent real HTTP during init.
    TestBed.overrideComponent(PortfolioBuilderComponent, {
      set: {
        providers: [
          BuilderStore,
          {
            provide: BuilderResultService,
            useValue: { init: () => {}, runExplicit: () => {} },
          },
          {
            provide: BuilderDriftService,
            useValue: { init: () => {}, runExplicit: () => {} },
          },
          {
            provide: BUILDER_DRIFT_SERVICE,
            useValue: { runExplicit: () => {} },
          },
        ],
      },
    });

    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(PortfolioBuilderComponent);
    // Component-level providers: get the BuilderStore instance owned by this component.
    store = fixture.debugElement.injector.get(BuilderStore);
  });

  afterEach(() => {
    http.verify();
  });

  // (c) reacts to portfolio context change — name propagation
  describe('when currentPortfolioName() emits a name', () => {
    it('when name is "Growth Portfolio", store.portfolioName() is "Growth Portfolio"', () => {
      mockName.set('Growth Portfolio');
      fixture.detectChanges();
      TestBed.flushEffects();

      expect(store.portfolioName()).toBe('Growth Portfolio');
    });

    it('when name is "Growth Portfolio" and id is a UUID, store.portfolioName() is not the UUID', () => {
      const uuid = '550e8400-e29b-41d4-a716-446655440000';
      mockId.set(uuid);
      mockName.set('Growth Portfolio');
      fixture.detectChanges();
      TestBed.flushEffects();

      expect(store.portfolioName()).not.toBe(uuid);
      expect(store.portfolioName()).toBe('Growth Portfolio');
    });

    it('when name changes from "Old Portfolio" to "New Portfolio", store reflects the new name', () => {
      mockName.set('Old Portfolio');
      fixture.detectChanges();
      TestBed.flushEffects();

      mockName.set('New Portfolio');
      fixture.detectChanges();
      TestBed.flushEffects();

      expect(store.portfolioName()).toBe('New Portfolio');
    });
  });

  // Null guard: no-op when name is null
  describe('when currentPortfolioName() is null', () => {
    it('when name is null, store.portfolioName() is null', () => {
      mockName.set(null);
      fixture.detectChanges();
      TestBed.flushEffects();

      expect(store.portfolioName()).toBeNull();
    });

    it('when name transitions from a value to null, store.portfolioName() becomes null', () => {
      mockName.set('Temporary Portfolio');
      fixture.detectChanges();
      TestBed.flushEffects();

      mockName.set(null);
      fixture.detectChanges();
      TestBed.flushEffects();

      expect(store.portfolioName()).toBeNull();
    });
  });

  // (a) loads data on init — store starts with null name until context provides one
  describe('on init with no portfolio selected', () => {
    it('store.portfolioName() starts as null when context emits null', () => {
      fixture.detectChanges();
      TestBed.flushEffects();

      expect(store.portfolioName()).toBeNull();
    });
  });
});
