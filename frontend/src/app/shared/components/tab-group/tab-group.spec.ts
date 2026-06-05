import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../../testing';
import { TabGroupComponent, type Tab } from './tab-group';

const TABS: Tab[] = [
  { id: 'a', label: 'Alpha' },
  { id: 'b', label: 'Beta', badge: 3 },
  { id: 'c', label: 'Gamma' },
];

describe('TabGroupComponent', () => {
  let fixture: ComponentFixture<TabGroupComponent>;
  let comp: TabGroupComponent;

  // tabs AND activeTab are both required: set both before the first CD.
  function mount(active: string): void {
    fixture.componentRef.setInput('tabs', TABS);
    fixture.componentRef.setInput('activeTab', active);
    fixture.detectChanges();
  }

  beforeEach(async () => {
    await configureTestBed({ imports: [TabGroupComponent], withHttp: false });
    fixture = TestBed.createComponent(TabGroupComponent);
    comp = fixture.componentInstance;
  });

  it('when tabs are set, one button per tab is rendered with aria-selected on the active one', () => {
    mount('a');
    const btns = fixture.nativeElement.querySelectorAll('button[role="tab"]');
    expect(btns.length).toBe(3);
    expect(btns[0].getAttribute('aria-selected')).toBe('true');
    expect(btns[1].getAttribute('aria-selected')).toBe('false');
  });

  it('when a tab is clicked, tabChange emits its id', () => {
    mount('a');
    let emitted: string | undefined;
    comp.tabChange.subscribe((id) => (emitted = id));
    const btns = fixture.nativeElement.querySelectorAll('button[role="tab"]');
    btns[1].click();
    expect(emitted).toBe('b');
  });

  it('when ArrowRight is pressed, tabChange emits the next tab id', () => {
    mount('a');
    let emitted: string | undefined;
    comp.tabChange.subscribe((id) => (emitted = id));
    const list = fixture.nativeElement.querySelector('[role="tablist"]') as HTMLElement;
    list.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(emitted).toBe('b');
  });

  it('when ArrowLeft is pressed at the first tab, it wraps to the last', () => {
    mount('a');
    let emitted: string | undefined;
    comp.tabChange.subscribe((id) => (emitted = id));
    const list = fixture.nativeElement.querySelector('[role="tablist"]') as HTMLElement;
    list.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    expect(emitted).toBe('c');
  });
});
