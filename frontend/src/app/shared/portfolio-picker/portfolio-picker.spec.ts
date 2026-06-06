import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject, of, throwError } from 'rxjs';

import { PortfolioPickerComponent } from './portfolio-picker';
import { PortfolioApiService } from '../../core/services/portfolio-api.service';
import {
  PortfolioDto,
  PortfolioListResponseDto,
} from '../../core/models/portfolio-api.model';
import { PortfolioContextService } from '../../core/services/portfolio-context.service';

function portfolioDto(overrides: Partial<PortfolioDto> = {}): PortfolioDto {
  return {
    id: 'id-1',
    name: 'Alpha',
    description: null,
    currency: 'USD',
    benchmark_ticker: 'SPY',
    is_active: true,
    created_at: '2026-01-01',
    updated_at: '2026-01-01',
    ...overrides,
  };
}

describe('PortfolioPickerComponent', () => {
  let api: jasmine.SpyObj<PortfolioApiService>;
  let ctx: PortfolioContextService;

  function setup(): PortfolioPickerComponent {
    const fixture = TestBed.createComponent(PortfolioPickerComponent);
    fixture.detectChanges();
    return fixture.componentInstance;
  }

  function configure(apiSpy: jasmine.SpyObj<PortfolioApiService>) {
    TestBed.configureTestingModule({
      imports: [PortfolioPickerComponent],
      providers: [
        provideZonelessChangeDetection(),
        { provide: PortfolioApiService, useValue: apiSpy },
      ],
    });
    ctx = TestBed.inject(PortfolioContextService);
  }

  beforeEach(() => {
    api = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', [
      'list',
      'get',
    ]);
  });

  it('renders a select with each portfolio name after loading', () => {
    const items = [
      portfolioDto({ id: 'id-a', name: 'Alpha' }),
      portfolioDto({ id: 'id-b', name: 'Beta' }),
    ];
    api.list.and.returnValue(of({ items, total: 2 } as PortfolioListResponseDto));
    configure(api);

    const fixture = TestBed.createComponent(PortfolioPickerComponent);
    fixture.detectChanges();

    const options = fixture.nativeElement.querySelectorAll(
      'select option',
    ) as NodeListOf<HTMLOptionElement>;
    const labels = Array.from(options).map((o) => o.textContent?.trim());
    expect(labels).toContain('Alpha');
    expect(labels).toContain('Beta');
  });

  it('calls PortfolioContextService.setPortfolio with id on change', () => {
    const items = [
      portfolioDto({ id: 'id-a', name: 'Alpha' }),
      portfolioDto({ id: 'id-b', name: 'Beta' }),
    ];
    api.list.and.returnValue(of({ items, total: 2 } as PortfolioListResponseDto));
    configure(api);
    const setSpy = spyOn(TestBed.inject(PortfolioContextService), 'setPortfolio');

    const fixture = TestBed.createComponent(PortfolioPickerComponent);
    fixture.detectChanges();

    const select = fixture.nativeElement.querySelector(
      'select',
    ) as HTMLSelectElement;
    select.value = 'id-b';
    select.dispatchEvent(new Event('change'));

    expect(setSpy).toHaveBeenCalledWith('id-b');
  });

  it('shows empty state when portfolio list is empty', () => {
    api.list.and.returnValue(of({ items: [], total: 0 } as PortfolioListResponseDto));
    configure(api);

    const fixture = TestBed.createComponent(PortfolioPickerComponent);
    fixture.detectChanges();

    const text = (fixture.nativeElement.textContent as string).toLowerCase();
    expect(text).toContain('no portfolios');
    expect(fixture.nativeElement.querySelector('select')).toBeNull();
  });

  it('shows loading state before the list resolves', () => {
    const pending = new Subject<PortfolioListResponseDto>();
    api.list.and.returnValue(pending.asObservable());
    configure(api);

    const fixture = TestBed.createComponent(PortfolioPickerComponent);
    fixture.detectChanges();

    const html = fixture.nativeElement.innerHTML as string;
    expect(html).toMatch(/loading|skeleton/i);
    expect(fixture.nativeElement.querySelector('select')).toBeNull();
  });

  it('shows error state with retry when the API call fails', () => {
    api.list.and.returnValue(throwError(() => new Error('boom')));
    configure(api);

    const fixture = TestBed.createComponent(PortfolioPickerComponent);
    fixture.detectChanges();

    const text = (fixture.nativeElement.textContent as string).toLowerCase();
    expect(text).toContain('failed');
    const retry = fixture.nativeElement.querySelector(
      'button',
    ) as HTMLButtonElement | null;
    expect(retry).not.toBeNull();

    // Second attempt succeeds when user clicks retry
    api.list.and.returnValue(
      of({ items: [portfolioDto()], total: 1 } as PortfolioListResponseDto),
    );
    retry!.click();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('select')).not.toBeNull();
  });
});
