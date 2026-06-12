import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { DataManagementPanelComponent } from './data-management-panel';
import { environment } from '../../../environments/environment';
import type {
  DatabaseStatus,
  HealthCheck,
  TableInfo,
} from '../database.model';

const API = environment.apiUrl;

const HEALTH: HealthCheck = {
  healthy: true,
  latency_ms: 3,
  database_url: 'postgresql://postgres:***@localhost:54320/optimizer_db',
};

const UNHEALTHY: HealthCheck = {
  healthy: false,
  latency_ms: 9999,
  database_url: 'postgresql://postgres:***@localhost:54320/optimizer_db',
};

const STATUS: DatabaseStatus = {
  health: HEALTH,
  tables: [],
  total_size_pretty: '12 MB',
};

const TABLES: TableInfo[] = [
  { name: 'prices', schema: 'public', row_count: 1000, size_bytes: 10_000, size_pretty: '10 KB' },
  { name: 'news', schema: 'public', row_count: 50, size_bytes: 500, size_pretty: '500 B' },
];

describe('DataManagementPanelComponent', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function flushInitialLoad(): void {
    http.expectOne(`${API}database/health`).flush(HEALTH);
    http.expectOne(`${API}database/status`).flush(STATUS);
    http.expectOne(`${API}database/tables`).flush(TABLES);
  }

  it('loads health, status, and tables on construction', () => {
    const fx = TestBed.createComponent(DataManagementPanelComponent);
    fx.detectChanges();
    flushInitialLoad();

    expect(fx.componentInstance.health()?.healthy).toBe(true);
    expect(fx.componentInstance.status()?.total_size_pretty).toBe('12 MB');
    expect(fx.componentInstance.tables().length).toBe(2);
    expect(fx.componentInstance.healthIsHealthy()).toBe(true);
  });

  describe('healthLabel + healthIsHealthy (issue #436)', () => {
    function loadWithHealth(payload: HealthCheck) {
      const fx = TestBed.createComponent(DataManagementPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}database/health`).flush(payload);
      http.expectOne(`${API}database/status`).flush(STATUS);
      http.expectOne(`${API}database/tables`).flush([]);
      return fx.componentInstance;
    }

    it('renders "healthy" in green when /database/health returns healthy: true', () => {
      const c = loadWithHealth(HEALTH);
      expect(c.healthLabel()).toBe('healthy');
      expect(c.healthIsHealthy()).toBe(true);
    });

    it('renders "unhealthy" in red when /database/health returns healthy: false', () => {
      const c = loadWithHealth(UNHEALTHY);
      expect(c.healthLabel()).toBe('unhealthy');
      expect(c.healthIsHealthy()).toBe(false);
    });

    it('renders "unknown" while the health response is still null', () => {
      // Build the component but never flush health → health() stays null
      const fx = TestBed.createComponent(DataManagementPanelComponent);
      fx.detectChanges();
      // Drain the in-flight requests so afterEach http.verify() passes
      http.expectOne(`${API}database/health`).flush(HEALTH); // populate
      http.expectOne(`${API}database/status`).flush(STATUS);
      http.expectOne(`${API}database/tables`).flush([]);
      // Reset the signal to simulate the pre-load state
      fx.componentInstance.health.set(null);
      expect(fx.componentInstance.healthLabel()).toBe('unknown');
      expect(fx.componentInstance.healthIsHealthy()).toBe(false);
    });
  });

  it('populates name, schema, row_count, and size_pretty on every table row (issue #435)', () => {
    const fx = TestBed.createComponent(DataManagementPanelComponent);
    fx.detectChanges();
    flushInitialLoad();

    const rows = fx.componentInstance.tableRows();
    expect(rows.length).toBe(2);
    for (const row of rows) {
      expect(row['name']).toBeTruthy();
      expect(row['schema']).toBeTruthy();
      expect(typeof row['row_count']).toBe('number');
      expect(row['size_pretty']).toBeTruthy();
    }

    // Sanity: the first row's values come from the realistic API payload.
    const first = rows[0];
    expect(first['name']).toBe('prices');
    expect(first['schema']).toBe('public');
    expect(first['row_count']).toBe(1000);
    expect(first['size_pretty']).toBe('10 KB');
  });

  it('gates truncate behind a confirmation dialog and sends confirm=true', () => {
    const fx = TestBed.createComponent(DataManagementPanelComponent);
    fx.detectChanges();
    flushInitialLoad();

    fx.componentInstance.requestDelete(TABLES[0]);
    expect(fx.componentInstance.pendingDelete()?.name).toBe('prices');
    expect(fx.componentInstance.deleteMessage()).toContain('prices');

    http.expectNone((r) => r.method === 'DELETE');

    fx.componentInstance.confirmDelete();

    const del = http.expectOne(
      (r) => r.method === 'DELETE' && r.url === `${API}database/tables/prices`,
    );
    expect(del.request.params.get('confirm')).toBe('true');
    del.flush({ status: 'truncated', table: 'prices' });

    http.expectOne(`${API}database/health`).flush(HEALTH);
    http.expectOne(`${API}database/status`).flush(STATUS);
    http.expectOne(`${API}database/tables`).flush([TABLES[1]]);

    expect(fx.componentInstance.pendingDelete()).toBeNull();
    expect(fx.componentInstance.deleteSuccess()).toContain('prices');
    expect(fx.componentInstance.tables().length).toBe(1);
  });

  it('leaves state untouched when the user cancels the confirmation', () => {
    const fx = TestBed.createComponent(DataManagementPanelComponent);
    fx.detectChanges();
    flushInitialLoad();

    fx.componentInstance.requestDelete(TABLES[0]);
    fx.componentInstance.cancelDelete();

    expect(fx.componentInstance.pendingDelete()).toBeNull();
    http.expectNone((r) => r.method === 'DELETE');
  });

  it('surfaces the error when truncate fails', () => {
    const fx = TestBed.createComponent(DataManagementPanelComponent);
    fx.detectChanges();
    flushInitialLoad();

    fx.componentInstance.requestDelete(TABLES[0]);
    fx.componentInstance.confirmDelete();

    http
      .expectOne((r) => r.method === 'DELETE' && r.url === `${API}database/tables/prices`)
      .flush({ detail: 'locked' }, { status: 500, statusText: 'Server Error' });

    expect(fx.componentInstance.deleting()).toBe(false);
    expect(fx.componentInstance.deleteError()).toBeTruthy();
    expect(fx.componentInstance.pendingDelete()?.name).toBe('prices');
  });

  describe('error banner (issue #964)', () => {
    it('when health endpoint returns 500, loadError is set and app-page-error-banner is in the DOM', () => {
      const fx = TestBed.createComponent(DataManagementPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}database/health`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      http.expectOne(`${API}database/status`).flush(STATUS);
      http.expectOne(`${API}database/tables`).flush(TABLES);
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeTruthy();
      expect(fx.nativeElement.querySelector('app-page-error-banner')).not.toBeNull();
    });

    it('when tables endpoint returns 500, loadError is set and app-page-error-banner is in the DOM', () => {
      const fx = TestBed.createComponent(DataManagementPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}database/health`).flush(HEALTH);
      http.expectOne(`${API}database/status`).flush(STATUS);
      http.expectOne(`${API}database/tables`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeTruthy();
      expect(fx.nativeElement.querySelector('app-page-error-banner')).not.toBeNull();
    });

    it('when status endpoint returns 500, loadError is set (previously silent failure)', () => {
      const fx = TestBed.createComponent(DataManagementPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}database/health`).flush(HEALTH);
      http.expectOne(`${API}database/status`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      http.expectOne(`${API}database/tables`).flush(TABLES);
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeTruthy();
      expect(fx.nativeElement.querySelector('app-page-error-banner')).not.toBeNull();
    });

    it('clicking the retry button in the banner re-fires all load requests', () => {
      const fx = TestBed.createComponent(DataManagementPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}database/health`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      http.expectOne(`${API}database/status`).flush(STATUS);
      http.expectOne(`${API}database/tables`).flush(TABLES);
      fx.detectChanges();

      const banner: HTMLElement = fx.nativeElement.querySelector('app-page-error-banner');
      banner.querySelector<HTMLButtonElement>('button')!.click();
      fx.detectChanges();

      http.expectOne(`${API}database/health`).flush(HEALTH);
      http.expectOne(`${API}database/status`).flush(STATUS);
      http.expectOne(`${API}database/tables`).flush(TABLES);

      fx.detectChanges();
      expect(fx.componentInstance.loadError()).toBeNull();
    });
  });
});
