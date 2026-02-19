import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { HealthCheck, TableInfo, DatabaseStatus } from '../models/database.model';
import { MOCK_HEALTH, MOCK_TABLES } from '../mocks/mock-data';

@Injectable({ providedIn: 'root' })
export class DatabaseService {
  getHealth(): Observable<HealthCheck> {
    return of(MOCK_HEALTH).pipe(delay(300));
  }

  getTables(): Observable<TableInfo[]> {
    return of(MOCK_TABLES).pipe(delay(400));
  }

  getStatus(): Observable<DatabaseStatus> {
    return of({
      health: MOCK_HEALTH,
      tables: MOCK_TABLES,
      total_size_pretty: '843 MB',
    }).pipe(delay(350));
  }
}
