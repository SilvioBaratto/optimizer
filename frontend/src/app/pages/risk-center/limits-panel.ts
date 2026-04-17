import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NotificationService } from '../../shared/notification/notification.service';
import { FormatService } from '../../services/format.service';
import type {
  RiskAlert,
  RiskLimit,
  RiskLimitCreatePayload,
  RiskLimitStatus,
  RiskLimitType,
} from '../../models/risk.model';

interface CreateForm {
  metric: string;
  limit_type: RiskLimitType;
  threshold: number;
}

const EMPTY_FORM: CreateForm = { metric: '', limit_type: 'upper', threshold: 0.1 };

@Component({
  selector: 'app-limits-panel',
  imports: [DecimalPipe, FormsModule],
  templateUrl: './limits-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LimitsPanelComponent {
  readonly limits = input<RiskLimit[]>([]);
  readonly alerts = input<RiskAlert[]>([]);

  readonly acknowledgeAlert = output<string>();
  readonly createLimit = output<RiskLimitCreatePayload>();
  readonly deleteLimit = output<string>();

  private readonly notifications = inject(NotificationService);
  private readonly fmt = inject(FormatService);

  readonly pendingDeleteId = signal<string | null>(null);
  readonly createForm = signal<CreateForm>(EMPTY_FORM);
  readonly showCreateForm = signal<boolean>(false);

  readonly sortedAlerts = computed(() =>
    [...this.alerts()].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    ),
  );

  readonly unackedCount = computed(
    () => this.alerts().filter((a) => !a.acknowledged).length,
  );

  readonly breachCount = computed(
    () => this.limits().filter((l) => l.status === 'breached').length,
  );

  readonly isFormValid = computed(() => {
    const f = this.createForm();
    return f.metric.trim().length >= 2 && f.threshold > 0;
  });

  getUtilization(limit: RiskLimit): number {
    if (limit.limit === 0) return 0;
    return Math.min((Math.abs(limit.current) / limit.limit) * 100, 100);
  }

  getBarColorClass(status: RiskLimitStatus): string {
    if (status === 'ok') return 'bg-gain';
    if (status === 'warning') return 'bg-chart-4';
    return 'bg-loss';
  }

  getBadgeClasses(status: RiskLimitStatus): string {
    if (status === 'ok') return 'bg-gain/10 text-gain';
    if (status === 'warning') return 'bg-chart-4/10 text-chart-4';
    return 'bg-loss/10 text-loss';
  }

  getSeverityDotClass(severity: string): string {
    if (severity === 'critical') return 'bg-loss';
    if (severity === 'warning') return 'bg-chart-4';
    if (severity === 'info') return 'bg-accent';
    return 'bg-text-tertiary';
  }

  formatLimitValue(limit: RiskLimit): string {
    return limit.limit >= 1
      ? this.fmt.formatRatio(limit.limit)
      : this.fmt.formatPercent(limit.limit);
  }

  formatCurrentValue(limit: RiskLimit): string {
    return limit.limit >= 1
      ? this.fmt.formatRatio(limit.current)
      : this.fmt.formatPercent(limit.current);
  }

  formatTimestamp(ts: string): string {
    return this.fmt.formatDate(ts, 'medium');
  }

  onAcknowledge(alertId: string): void {
    this.acknowledgeAlert.emit(alertId);
    this.notifications.success('Alert acknowledged');
  }

  toggleCreateForm(): void {
    this.showCreateForm.update((v) => !v);
    this.createForm.set(EMPTY_FORM);
  }

  updateFormField<K extends keyof CreateForm>(key: K, value: CreateForm[K]): void {
    this.createForm.update((f) => ({ ...f, [key]: value }));
  }

  submitCreate(): void {
    if (!this.isFormValid()) return;
    this.createLimit.emit({ ...this.createForm() });
    this.showCreateForm.set(false);
    this.createForm.set(EMPTY_FORM);
  }

  requestDelete(id: string): void {
    this.pendingDeleteId.set(id);
  }

  cancelDelete(): void {
    this.pendingDeleteId.set(null);
  }

  confirmDelete(): void {
    const id = this.pendingDeleteId();
    if (!id) return;
    this.deleteLimit.emit(id);
    this.pendingDeleteId.set(null);
  }
}
