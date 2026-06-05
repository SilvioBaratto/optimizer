import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../../testing';
import { PageHeaderComponent } from './page-header';

describe('PageHeaderComponent', () => {
  let fixture: ComponentFixture<PageHeaderComponent>;
  let comp: PageHeaderComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [PageHeaderComponent], withHttp: false });
    fixture = TestBed.createComponent(PageHeaderComponent);
    comp = fixture.componentInstance;
  });

  it('when a title is set, it renders in the heading', () => {
    setInput(fixture, 'title', 'Risk Center');
    expect(fixture.nativeElement.querySelector('h1').textContent).toContain('Risk Center');
  });

  it('when lastUpdated is null, no "Updated" line is shown', () => {
    setInput(fixture, 'title', 'Risk Center');
    setInput(fixture, 'lastUpdated', null);
    expect(comp.relativeTime()).toBe('');
    expect(fixture.nativeElement.textContent).not.toContain('Updated');
  });

  it('when lastUpdated is minutes ago, the relative time reads "Nm ago"', () => {
    setInput(fixture, 'title', 'Risk Center');
    setInput(fixture, 'lastUpdated', new Date(Date.now() - 5 * 60_000));
    expect(comp.relativeTime()).toBe('5m ago');
    expect(fixture.nativeElement.textContent).toContain('Updated');
  });

  it('when lastUpdated is days ago, the relative time reads "Nd ago"', () => {
    setInput(fixture, 'title', 'Risk Center');
    setInput(fixture, 'lastUpdated', new Date(Date.now() - 3 * 86_400_000));
    expect(comp.relativeTime()).toBe('3d ago');
  });

  it('when a subtitle is set, it renders', () => {
    setInput(fixture, 'title', 'Risk Center');
    setInput(fixture, 'subtitle', 'Exposure & limits');
    expect(fixture.nativeElement.textContent).toContain('Exposure & limits');
  });
});
