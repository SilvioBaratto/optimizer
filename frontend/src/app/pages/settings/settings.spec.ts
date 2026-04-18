import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { ActivatedRoute, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { SettingsComponent } from './settings';
import { ICON_PROVIDER } from '../../icons';

describe('SettingsComponent — Jobs tab (#457)', () => {
  let fixture: ComponentFixture<SettingsComponent>;
  let component: SettingsComponent;
  let http: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SettingsComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        {
          provide: ActivatedRoute,
          useValue: { queryParams: of({}) },
        },
      ],
    }).compileComponents();

    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(SettingsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  afterEach(() => {
    // Drain any pending list-jobs request the jobs panel may have fired when
    // the tab activated. Some may already be cancelled by the time afterEach
    // runs (takeUntilDestroyed), so swallow the flush error in that case.
    const open = http.match(() => true);
    for (const req of open) {
      if (req.cancelled) continue;
      try {
        req.flush({ jobs: [], total: 0, limit: 25, offset: 0 });
      } catch {
        /* request was cancelled between match() and flush — ignore */
      }
    }
    http.verify();
  });

  it("exposes a 'jobs' tab labelled 'Jobs'", () => {
    const tab = component.tabs.find((t) => t.id === 'jobs');
    expect(tab).toBeDefined();
    expect(tab!.label).toBe('Jobs');
  });

  it("mounts <app-jobs-panel> when activeTab() === 'jobs'", () => {
    component.activeTab.set('jobs');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-jobs-panel').length).toBe(1);
  });

  it("does NOT mount <app-jobs-panel> when a different tab is active", () => {
    component.activeTab.set('data');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-jobs-panel').length).toBe(0);
  });
});
