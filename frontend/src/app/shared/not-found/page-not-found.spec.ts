import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { PageNotFoundComponent } from './page-not-found';

describe('PageNotFoundComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PageNotFoundComponent],
      providers: [provideZonelessChangeDetection(), provideRouter([])],
    }).compileComponents();
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(PageNotFoundComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the 404 heading', () => {
    const fixture = TestBed.createComponent(PageNotFoundComponent);
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('404');
    expect(text.toLowerCase()).toContain('page not found');
  });

  it('renders a back-to-dashboard link that points to /', () => {
    const fixture = TestBed.createComponent(PageNotFoundComponent);
    fixture.detectChanges();

    const anchor = fixture.nativeElement.querySelector(
      'a[routerLink="/"]',
    ) as HTMLAnchorElement | null;
    expect(anchor).not.toBeNull();
    expect(anchor!.textContent!.toLowerCase()).toContain('dashboard');
  });
});
