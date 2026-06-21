/**
 * Issue #1036 — GlobalSearchComponent with no query must return exactly
 * six results (one per kept nav item). Deleted routes must not appear.
 * Source-blind: derived from acceptance criteria only.
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';

import { configureTestBed } from '../../../testing';
import { GlobalSearchComponent } from './global-search';
import { GlobalSearchService } from './global-search.service';
import { ICON_PROVIDER } from '../../icons';

describe('GlobalSearchComponent — six results after nav trim (#1036)', () => {
  let fixture: ComponentFixture<GlobalSearchComponent>;
  let comp: GlobalSearchComponent;
  let router: Router;

  beforeEach(async () => {
    spyOn(HTMLInputElement.prototype, 'focus');
    await configureTestBed({
      imports: [GlobalSearchComponent],
      withRouter: true,
      withHttp: false,
      providers: [GlobalSearchService, ICON_PROVIDER],
    });

    fixture = TestBed.createComponent(GlobalSearchComponent);
    comp = fixture.componentInstance;
    router = TestBed.inject(Router);
    spyOn(router, 'navigateByUrl').and.returnValue(Promise.resolve(true));
    fixture.detectChanges();
  });

  it('when no query is set, filteredResults() returns exactly six items', () => {
    expect(comp.filteredResults().length).toBe(6);
  });

  it('when no query is set, /risk-center is not in results', () => {
    const routes = comp.filteredResults().map(r => r.route);
    expect(routes).not.toContain('/risk-center');
  });

  it('when no query is set, /ai-control-room is not in results', () => {
    const routes = comp.filteredResults().map(r => r.route);
    expect(routes).not.toContain('/ai-control-room');
  });

  it('when no query is set, /rebalancing is not in results', () => {
    const routes = comp.filteredResults().map(r => r.route);
    expect(routes).not.toContain('/rebalancing');
  });

  it('when no query is set, /attribution is not in results', () => {
    const routes = comp.filteredResults().map(r => r.route);
    expect(routes).not.toContain('/attribution');
  });

  it('when no query is set, /factor-research is not in results', () => {
    const routes = comp.filteredResults().map(r => r.route);
    expect(routes).not.toContain('/factor-research');
  });

  it('when a result with route /optimize is selected, router navigates to that kept route', () => {
    const optimizeResult = comp.filteredResults().find(r => r.route === '/optimize');
    expect(optimizeResult).withContext('Optimize entry must exist in results').toBeDefined();
    comp.selectResult(optimizeResult!);
    expect(router.navigateByUrl).toHaveBeenCalledWith('/optimize');
  });
});
