/**
 * settings-field-parity.spec.ts — issue #966
 *
 * Field-level contract-parity specs for DatabaseService, SchedulerService,
 * and JobsService request/response models. Locks wire shapes to backend
 * interfaces so accidental casing drift fails CI.
 *
 * Convention split (encoded explicitly):
 *   • DatabaseService   → snake_case (plain BaseModel): HealthCheck, TableInfo, DatabaseStatus, TruncateResponse
 *   • SchedulerService  → camelCase  (CamelCaseModel):  SchedulerStatusResponse, SchedulerJobStatusDto
 *   • JobsService       → snake_case (plain BaseModel):  JobSummary, JobListResponse
 */

import { schemaOf } from './contract-parity';
import { assertFieldParity, requiredKeys } from './contract-field-parity';
import jobsSnapshot from './contract-snapshots/jobs.json';
import {
  makeDatabaseStatus,
  makeHealthCheck,
  makeJobListResponse,
  makeJobSummary,
  makeSchedulerJobStatusDto,
  makeSchedulerStatusResponse,
  makeTableInfo,
  makeTruncateResponse,
} from './domain-fixtures';

// ── DatabaseService ───────────────────────────────────────────────────────────

describe('DatabaseService — field-level parity (issue #966)', () => {

  describe('HealthCheck — snake_case health response', () => {
    const FIXTURE = makeHealthCheck();

    it('snake_case keys present: healthy, latency_ms, database_url', () => {
      expect('healthy' in FIXTURE).toBe(true);
      expect('latency_ms' in FIXTURE).toBe(true);
      expect('database_url' in FIXTURE).toBe(true);
    });

    it('does NOT use camelCase equivalents', () => {
      expect('latencyMs' in FIXTURE).toBe(false);
      expect('databaseUrl' in FIXTURE).toBe(false);
    });

    it('healthy is boolean', () => {
      expect(typeof FIXTURE.healthy).toBe('boolean');
    });

    it('latency_ms is number', () => {
      expect(typeof FIXTURE.latency_ms).toBe('number');
    });
  });

  describe('TableInfo — snake_case table row', () => {
    const FIXTURE = makeTableInfo();

    it('required snake_case keys present', () => {
      ['name', 'schema', 'row_count', 'size_bytes', 'size_pretty'].forEach((k) =>
        expect(k in FIXTURE).withContext(`expected '${k}' in TableInfo`).toBe(true),
      );
    });

    it('does NOT use camelCase (rowCount / sizeBytes / sizePretty)', () => {
      expect('rowCount' in FIXTURE).toBe(false);
      expect('sizeBytes' in FIXTURE).toBe(false);
      expect('sizePretty' in FIXTURE).toBe(false);
    });

    it('row_count and size_bytes are numbers', () => {
      expect(typeof FIXTURE.row_count).toBe('number');
      expect(typeof FIXTURE.size_bytes).toBe('number');
    });
  });

  describe('DatabaseStatus — snake_case aggregate status', () => {
    const FIXTURE = makeDatabaseStatus();

    it('top-level key is snake_case: total_size_pretty (not totalSizePretty)', () => {
      expect('total_size_pretty' in FIXTURE).toBe(true);
      expect('totalSizePretty' in FIXTURE).toBe(false);
    });

    it('embeds nested HealthCheck under health key', () => {
      expect(typeof FIXTURE.health).toBe('object');
      expect('healthy' in FIXTURE.health).toBe(true);
    });

    it('tables is an array', () => {
      expect(Array.isArray(FIXTURE.tables)).toBe(true);
    });
  });

  describe('TruncateResponse — snake_case DELETE response', () => {
    const FIXTURE = makeTruncateResponse();

    it('table and status are present strings', () => {
      expect(typeof FIXTURE.table).toBe('string');
      expect(typeof FIXTURE.status).toBe('string');
    });
  });

});

// ── SchedulerService ──────────────────────────────────────────────────────────

describe('SchedulerService — field-level parity (issue #966)', () => {

  describe('SchedulerStatusResponse — camelCase (CamelCaseModel)', () => {
    const FIXTURE = makeSchedulerStatusResponse();

    it('camelCase key: schedulerRunning (not scheduler_running)', () => {
      expect('schedulerRunning' in FIXTURE).toBe(true);
      expect('scheduler_running' in FIXTURE).toBe(false);
    });

    it('schedulerRunning is boolean', () => {
      expect(typeof FIXTURE.schedulerRunning).toBe('boolean');
    });

    it('jobs is an array', () => {
      expect(Array.isArray(FIXTURE.jobs)).toBe(true);
    });
  });

  describe('SchedulerJobStatusDto — camelCase job item', () => {
    const FIXTURE = makeSchedulerJobStatusDto();

    it('camelCase keys present: jobId, nextRunTime, lastRunTime, lastStatus', () => {
      expect('jobId' in FIXTURE).toBe(true);
      expect('nextRunTime' in FIXTURE).toBe(true);
      expect('lastRunTime' in FIXTURE).toBe(true);
      expect('lastStatus' in FIXTURE).toBe(true);
    });

    it('does NOT use snake_case (job_id / next_run_time / last_run_time)', () => {
      expect('job_id' in FIXTURE).toBe(false);
      expect('next_run_time' in FIXTURE).toBe(false);
      expect('last_run_time' in FIXTURE).toBe(false);
    });

    it('jobId and name are strings', () => {
      expect(typeof FIXTURE.jobId).toBe('string');
      expect(typeof FIXTURE.name).toBe('string');
    });

    it('trigger is a string', () => {
      expect(typeof FIXTURE.trigger).toBe('string');
    });
  });

});

// ── JobsService ───────────────────────────────────────────────────────────────

describe('JobsService — field-level parity (issue #966)', () => {

  describe('JobSummary — snake_case persisted shape', () => {
    const schema = schemaOf(jobsSnapshot, 'JobSummary');

    it('required fields: id, domain, status', () => {
      const req = requiredKeys(schema);
      expect(req.has('id')).toBe(true);
      expect(req.has('domain')).toBe(true);
      expect(req.has('status')).toBe(true);
    });

    it('optional snake_case fields in schema properties', () => {
      const props = schema.properties ?? {};
      expect('errors_count' in props).toBe(true);
      expect('started_at' in props).toBe(true);
      expect('finished_at' in props).toBe(true);
      expect('duration_seconds' in props).toBe(true);
    });

    it('assertFieldParity: fixture types match snake_case wire schema', () => {
      expect(() => assertFieldParity(schema, makeJobSummary(), jobsSnapshot as never)).not.toThrow();
    });
  });

  describe('JobListResponse — paginated wrapper', () => {
    const schema = schemaOf(jobsSnapshot, 'JobListResponse');

    it('required fields: jobs, total, limit, offset', () => {
      const req = requiredKeys(schema);
      ['jobs', 'total', 'limit', 'offset'].forEach((k) =>
        expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
      );
    });

    it('assertFieldParity: fixture types match snapshot schema', () => {
      expect(() => assertFieldParity(schema, makeJobListResponse(), jobsSnapshot as never)).not.toThrow();
    });
  });

});
