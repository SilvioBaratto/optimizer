import { Component, input, output, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { NotificationService } from '../../shared/notification/notification.service';
import { FormatService } from '../../services/format.service';
import type { RiskLimit, RiskAlert, RiskLimitStatus } from '../../models/risk.model';

@Component({
  selector: 'app-limits-panel',
  imports: [DecimalPipe],
  templateUrl: './limits-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LimitsPanelComponent {
  limits = input<RiskLimit[]>([]);
  alerts = input<RiskAlert[]>([]);

  acknowledgeAlert = output<string>();
  addLimit = output<RiskLimit>();
  editLimit = output<RiskLimit>();

  private notifications = inject(NotificationService);
  private fmt = inject(FormatService);

  sortedAlerts = computed(() =>
    [...this.alerts()].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()),
  );

  unackedCount = computed(() => this.alerts().filter(a => !a.acknowledged).length);

  getUtilization(limit: RiskLimit): number {
    return Math.min((limit.current / limit.limit) * 100, 100);
  }

  getBarColorClass(status: RiskLimitStatus): string {
    switch (status) {
      case 'ok': return 'bg-gain';
      case 'warning': return 'bg-chart-4';
      case 'breached': return 'bg-loss';
    }
  }

  getBadgeClasses(status: RiskLimitStatus): string {
    switch (status) {
      case 'ok': return 'bg-gain/10 text-gain';
      case 'warning': return 'bg-chart-4/10 text-chart-4';
      case 'breached': return 'bg-loss/10 text-loss';
    }
  }

  getSeverityDotClass(severity: string): string {
    switch (severity) {
      case 'critical': return 'bg-loss';
      case 'warning': return 'bg-chart-4';
      case 'info': return 'bg-accent';
      default: return 'bg-text-tertiary';
    }
  }

  formatLimitValue(limit: RiskLimit): string {
    if (limit.limit >= 1) return this.fmt.formatRatio(limit.limit);
    return this.fmt.formatPercent(limit.limit);
  }

  formatCurrentValue(limit: RiskLimit): string {
    if (limit.limit >= 1) return this.fmt.formatRatio(limit.current);
    return this.fmt.formatPercent(limit.current);
  }

  formatTimestamp(ts: string): string {
    return this.fmt.formatDate(ts, 'medium');
  }

  onAcknowledge(alertId: string) {
    this.acknowledgeAlert.emit(alertId);
    this.notifications.success('Alert acknowledged');
  }
}
