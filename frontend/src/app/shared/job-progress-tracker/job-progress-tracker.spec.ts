import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject, of, throwError } from 'rxjs';

import {
  JobProgressTrackerComponent,
  JOB_POLL_TICK,
} from './job-progress-tracker';
import { JobsService } from '../../services/jobs.service';
import { JobSummary } from '../../models/jobs.model';

function summary(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id: 'job-1',
    domain: 'yfinance',
    status: 'running',
    current: 0,
    total: 100,
    error: null,
    errors_count: 0,
    started_at: '2026-01-01T00:00:00Z',
    finished_at: null,
    duration_seconds: null,
    ...overrides,
  };
}

describe('JobProgressTrackerComponent', () => {
  let jobs: jasmine.SpyObj<JobsService>;
  let tick: Subject<unknown>;

  function setup(jobId: string) {
    const fixture = TestBed.createComponent(JobProgressTrackerComponent);
    fixture.componentRef.setInput('jobId', jobId);
    fixture.detectChanges();
    return fixture;
  }

  beforeEach(() => {
    tick = new Subject<unknown>();
    jobs = jasmine.createSpyObj<JobsService>('JobsService', ['getJob']);

    TestBed.configureTestingModule({
      imports: [JobProgressTrackerComponent],
      providers: [
        provideZonelessChangeDetection(),
        { provide: JobsService, useValue: jobs },
        { provide: JOB_POLL_TICK, useValue: tick.asObservable() },
      ],
    });
  });

  it('updates progress when a running job is fetched', () => {
    jobs.getJob.and.returnValue(of(summary({ status: 'running', current: 25, total: 100 })));
    const fixture = setup('job-1');

    tick.next(0);
    fixture.detectChanges();

    expect(jobs.getJob).toHaveBeenCalledWith('job-1');
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('25');
  });

  it('emits completed output and stops polling on terminal completed status', () => {
    const completedJob = summary({
      status: 'completed',
      current: 100,
      total: 100,
      finished_at: '2026-01-01T00:05:00Z',
      duration_seconds: 300,
    });
    jobs.getJob.and.returnValue(of(completedJob));
    const fixture = setup('job-1');

    let emitted: JobSummary | undefined;
    fixture.componentInstance.completed.subscribe((j) => (emitted = j));

    tick.next(0);
    fixture.detectChanges();
    tick.next(1); // second tick should not trigger another poll

    expect(emitted).toEqual(completedJob);
    expect(jobs.getJob).toHaveBeenCalledTimes(1);
  });

  it('emits failed output when job reaches failed status', () => {
    const failedJob = summary({ status: 'failed', error: 'worker crashed', errors_count: 1 });
    jobs.getJob.and.returnValue(of(failedJob));
    const fixture = setup('job-1');

    let emitted: JobSummary | undefined;
    fixture.componentInstance.failed.subscribe((j) => (emitted = j));

    tick.next(0);
    fixture.detectChanges();

    expect(emitted).toEqual(failedJob);
    const text = fixture.nativeElement.textContent as string;
    expect(text.toLowerCase()).toContain('worker crashed');
  });

  it('stops polling on destroy', () => {
    jobs.getJob.and.returnValue(of(summary({ status: 'running', current: 10, total: 100 })));
    const fixture = setup('job-1');

    tick.next(0);
    expect(jobs.getJob).toHaveBeenCalledTimes(1);

    fixture.destroy();
    tick.next(1);

    expect(jobs.getJob).toHaveBeenCalledTimes(1);
  });

  it('shows an error message when the API call fails persistently', () => {
    jobs.getJob.and.returnValue(throwError(() => new Error('network down')));
    const fixture = setup('job-1');

    tick.next(0);
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text.toLowerCase()).toContain('network down');
  });
});
