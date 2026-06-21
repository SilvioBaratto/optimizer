// Structural layout spec for the new portfolio-builder shell (issue #1047).
//
// The shell renders a flex-column layout with two named data-regions:
//   - [data-region="search"]  → hosts app-asset-search
//   - [data-region="list"]    → hosts app-asset-list
//
// Pixel-level verification belongs in a Playwright/Cypress suite.
// This spec asserts structural contracts only.

import { TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { UniverseService } from '../core/services/universe.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';

function setup(): HTMLElement {
  TestBed.configureTestingModule({
    imports: [PortfolioBuilderComponent],
    schemas: [NO_ERRORS_SCHEMA],
    providers: [
      provideZonelessChangeDetection(),
      {
        provide: UniverseService,
        useValue: {
          searchTickers: () => of({ items: [], total: 0, page: 1, page_size: 15 }),
        },
      },
      {
        provide: PortfolioApiService,
        useValue: jasmine.createSpyObj('PortfolioApiService', ['create']),
      },
    ],
  });
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('PortfolioBuilderComponent — layout contract (R4 shell)', () => {
  it('when rendered, the [data-region="search"] section exists', () => {
    const host = setup();
    expect(host.querySelector('[data-region="search"]')).not.toBeNull();
  });

  it('when rendered, the [data-region="list"] section exists', () => {
    const host = setup();
    expect(host.querySelector('[data-region="list"]')).not.toBeNull();
  });

  it('when rendered, search and list regions are siblings under the same flex container', () => {
    const host = setup();
    const search = host.querySelector('[data-region="search"]');
    const list = host.querySelector('[data-region="list"]');
    expect(search?.parentElement).toBe(list?.parentElement);
  });

  it('when rendered, the search region appears before the list region in DOM order', () => {
    const host = setup();
    const regions = Array.from(host.querySelectorAll('[data-region]')).map((el) =>
      el.getAttribute('data-region'),
    );
    const searchIdx = regions.indexOf('search');
    const listIdx = regions.indexOf('list');
    expect(searchIdx).toBeGreaterThanOrEqual(0);
    expect(listIdx).toBeGreaterThan(searchIdx);
  });
});
