import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';

import { configureTestBed } from '../../../testing';
import { GlobalSearchComponent } from './global-search';
import { GlobalSearchService } from './global-search.service';
import { ICON_PROVIDER } from '../../icons';

describe('GlobalSearchComponent', () => {
  let fixture: ComponentFixture<GlobalSearchComponent>;
  let comp: GlobalSearchComponent;
  let router: Router;
  let search: GlobalSearchService;

  function dispatchKey(key: string): void {
    fixture.nativeElement.dispatchEvent(new KeyboardEvent('keydown', { key }));
    fixture.detectChanges();
  }

  beforeEach(async () => {
    // afterNextRender focuses the input; spy focus to keep the test DOM quiet.
    spyOn(HTMLInputElement.prototype, 'focus');
    await configureTestBed({
      imports: [GlobalSearchComponent],
      withRouter: true,
      withHttp: false,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(GlobalSearchComponent);
    comp = fixture.componentInstance;
    router = TestBed.inject(Router);
    search = TestBed.inject(GlobalSearchService);
    spyOn(router, 'navigateByUrl').and.returnValue(Promise.resolve(true));
    spyOn(search, 'close');
    fixture.detectChanges();
  });

  it('when no query is set, all nav items are listed', () => {
    expect(comp.filteredResults().length).toBe(11);
  });

  it('when a query is typed, results are filtered by label', () => {
    comp.onQueryChange('macro');
    expect(comp.filteredResults().length).toBe(1);
    expect(comp.filteredResults()[0].label).toBe('Macro Intelligence');
  });

  it('when ArrowDown is pressed, the active index advances', () => {
    expect(comp.activeIndex()).toBe(0);
    dispatchKey('ArrowDown');
    expect(comp.activeIndex()).toBe(1);
  });

  it('when Enter is pressed, the active route is navigated and the panel closes', () => {
    dispatchKey('Enter');
    expect(router.navigateByUrl).toHaveBeenCalledWith('/');
    expect(search.close).toHaveBeenCalled();
  });

  it('when Escape is pressed, the search service is closed', () => {
    dispatchKey('Escape');
    expect(search.close).toHaveBeenCalled();
  });

  it('when a result is clicked, it navigates and closes', () => {
    comp.selectResult({ label: 'Risk Center', route: '/risk-center', category: 'Risk & Research' });
    expect(router.navigateByUrl).toHaveBeenCalledWith('/risk-center');
    expect(search.close).toHaveBeenCalled();
  });
});
