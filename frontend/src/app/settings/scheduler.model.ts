// Mirrors api/app/schemas/scheduler.py. Response uses CamelCaseModel so the
// TypeScript fields are camelCase.

export interface SchedulerJobStatusDto {
  jobId: string;
  name: string;
  nextRunTime: string | null;
  lastRunTime: string | null;
  lastStatus: string | null;
  trigger: string;
}

export interface SchedulerStatusResponse {
  schedulerRunning: boolean;
  jobs: SchedulerJobStatusDto[];
}
