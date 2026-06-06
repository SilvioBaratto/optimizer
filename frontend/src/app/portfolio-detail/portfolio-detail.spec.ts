import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { PortfolioDetailComponent } from './portfolio-detail';

describe('PortfolioDetailComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PortfolioDetailComponent],
      providers: [provideZonelessChangeDetection(), provideRouter([])],
    }).compileComponents();
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(PortfolioDetailComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the portfolio name read from the route param input', () => {
    const fixture = TestBed.createComponent(PortfolioDetailComponent);
    fixture.componentRef.setInput('name', 'alpha');
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('alpha');
  });
});
