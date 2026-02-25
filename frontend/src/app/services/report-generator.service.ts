import { Injectable, signal, inject } from '@angular/core';
import { NotificationService } from '../shared/notification/notification.service';
import type { ReportConfig, ReportJob } from '../models/report.model';

@Injectable({ providedIn: 'root' })
export class ReportGeneratorService {
  private readonly notifications = inject(NotificationService);

  readonly currentJob = signal<ReportJob | null>(null);

  generateReport(config: ReportConfig): void {
    const job: ReportJob = {
      id: crypto.randomUUID(),
      config,
      status: 'generating',
      progress: 0,
      createdAt: new Date(),
    };

    this.currentJob.set(job);

    const delayMs = 1500 + Math.random() * 1500;

    setTimeout(() => {
      const formatLabel = config.format.toUpperCase();
      const templateLabel = config.template.replace(/_/g, ' ');

      this.currentJob.set({
        ...job,
        status: 'complete',
        progress: 100,
        completedAt: new Date(),
        downloadUrl: `data:application/octet-stream;base64,mock-report-${job.id}`,
      });

      this.notifications.success(
        `${formatLabel} report "${templateLabel}" is ready for download.`,
        { autoDismissMs: 6000 },
      );
    }, delayMs);
  }
}
