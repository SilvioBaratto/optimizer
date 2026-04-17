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
} from '../../models/database.model';

const API = environment.apiUrl;

const HEALTH: HealthCheck = {
  status: 'healthy',
  latency_ms: 3,
  database: 'optimizer_db',
  version: 'PostgreSQL 16',
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

    expect(fx.componentInstance.health()?.status).toBe('healthy');
    expect(fx.componentInstance.status()?.total_size_pretty).toBe('12 MB');
    expect(fx.componentInstance.tables().length).toBe(2);
    expect(fx.componentInstance.healthIsHealthy()).toBe(true);
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
});
