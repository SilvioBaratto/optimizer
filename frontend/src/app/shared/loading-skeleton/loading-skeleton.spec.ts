import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { LoadingSkeletonComponent } from './loading-skeleton';

describe('LoadingSkeletonComponent', () => {
  let fixture: ComponentFixture<LoadingSkeletonComponent>;

  beforeEach(async () => {
    await configureTestBed({ imports: [LoadingSkeletonComponent], withHttp: false });
    fixture = TestBed.createComponent(LoadingSkeletonComponent);
  });

  it('when no dimensions are set, the default height and width apply', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement.querySelector('div') as HTMLElement;
    expect(el.style.height).toBe('1rem');
    expect(el.style.width).toBe('100%');
  });

  it('when custom dimensions are set, they override the defaults', () => {
    setInput(fixture, 'height', '2rem');
    setInput(fixture, 'width', '50%');
    const el = fixture.nativeElement.querySelector('div') as HTMLElement;
    expect(el.style.height).toBe('2rem');
    expect(el.style.width).toBe('50%');
  });
});
