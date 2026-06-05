import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { EmptyStateComponent } from './empty-state';

describe('EmptyStateComponent', () => {
  let fixture: ComponentFixture<EmptyStateComponent>;

  beforeEach(async () => {
    await configureTestBed({ imports: [EmptyStateComponent], withHttp: false });
    fixture = TestBed.createComponent(EmptyStateComponent);
  });

  it('when a title is set, it renders in the heading', () => {
    setInput(fixture, 'title', 'Nothing here');
    expect(fixture.nativeElement.querySelector('h3').textContent).toContain('Nothing here');
  });

  it('when no description is given, no description paragraph is rendered', () => {
    setInput(fixture, 'title', 'Empty');
    expect(fixture.nativeElement.querySelector('p')).toBeNull();
  });

  it('when a description is given, it renders in a paragraph', () => {
    setInput(fixture, 'title', 'Empty');
    setInput(fixture, 'description', 'Try adjusting filters');
    expect(fixture.nativeElement.querySelector('p').textContent).toContain('Try adjusting filters');
  });
});
