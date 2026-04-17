import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import {
  DatabaseStatus,
  HealthCheck,
  TableInfo,
  TruncateResponse,
} from '../models/database.model';

@Injectable({ providedIn: 'root' })
export class DatabaseService {
  private readonly http = inject(HttpClient);
  private readonly base = `${environment.apiUrl}database`;

  getHealth(): Observable<HealthCheck> {
    return this.http.get<HealthCheck>(`${this.base}/health`);
  }

  getTables(): Observable<TableInfo[]> {
    return this.http.get<TableInfo[]>(`${this.base}/tables`);
  }

  getStatus(): Observable<DatabaseStatus> {
    return this.http.get<DatabaseStatus>(`${this.base}/status`);
  }

  truncateTable(name: string): Observable<TruncateResponse> {
    return this.http.delete<TruncateResponse>(
      `${this.base}/tables/${encodeURIComponent(name)}`,
      { params: new HttpParams().set('confirm', 'true') },
    );
  }
}
