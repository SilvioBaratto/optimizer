import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { SearchInputComponent } from './search-input';

describe('SearchInputComponent', () => {
  let fixture: ComponentFixture<SearchInputComponent>;
  let comp: SearchInputComponent;

  beforeEach(async () => {
    // Zoneless app: drive debounceTime with jasmine.clock (setTimeout-backed),
    // installed AFTER the async compile so it does not intercept it.
    await configureTestBed({ imports: [SearchInputComponent], withHttp: false, providers: [ICON_PROVIDER] });
    jasmine.clock().install();
    jasmine.clock().mockDate(); // debounceTime compares scheduler.now() to Date.now()
    fixture = TestBed.createComponent(SearchInputComponent);
    comp = fixture.componentInstance;
    fixture.detectChanges(); // runs ngOnInit → subscribes valueChanges
  });

  afterEach(() => jasmine.clock().uninstall());

  it('when a value is typed, searchChange emits only after the 300ms debounce', () => {
    let emitted: string | undefined;
    comp.searchChange.subscribe((v) => (emitted = v));
    comp.searchControl.setValue('abc');
    expect(emitted).toBeUndefined();
    jasmine.clock().tick(300);
    expect(emitted).toBe('abc');
  });

  it('when the same value is set twice, distinctUntilChanged emits once', () => {
    const seen: string[] = [];
    comp.searchChange.subscribe((v) => seen.push(v));
    comp.searchControl.setValue('abc');
    jasmine.clock().tick(300);
    comp.searchControl.setValue('abc');
    jasmine.clock().tick(300);
    expect(seen).toEqual(['abc']);
  });

  it('when cleared, an empty string is emitted', () => {
    let emitted: string | undefined;
    comp.searchControl.setValue('abc');
    jasmine.clock().tick(300);
    comp.searchChange.subscribe((v) => (emitted = v));
    comp.searchControl.setValue('');
    jasmine.clock().tick(300);
    expect(emitted).toBe('');
  });
});
