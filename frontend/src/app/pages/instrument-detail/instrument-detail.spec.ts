import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { InstrumentDetailComponent } from './instrument-detail';

describe('InstrumentDetailComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InstrumentDetailComponent],
      providers: [provideZonelessChangeDetection(), provideRouter([])],
    }).compileComponents();
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(InstrumentDetailComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the instrument id read from the route param input', () => {
    const fixture = TestBed.createComponent(InstrumentDetailComponent);
    fixture.componentRef.setInput('id', 'instr-123');
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('instr-123');
  });
});
