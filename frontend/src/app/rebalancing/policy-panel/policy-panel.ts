import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
  signal,
} from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import type {
  PolicyType,
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
} from '../rebalancing.model';

interface PolicyTypeBadge {
  label: string;
  colorClass: string;
}

interface CreateForm {
  name: string;
  policy_type: PolicyType;
  frequency_days: number;
  threshold: number;
  kind: 'absolute' | 'relative';
}

const EMPTY_FORM: CreateForm = {
  name: '',
  policy_type: 'calendar',
  frequency_days: 63,
  threshold: 0.05,
  kind: 'absolute',
};

@Component({
  selector: 'app-policy-panel',
  imports: [FormsModule, KeyValuePipe],
  templateUrl: './policy-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PolicyPanelComponent {
  readonly policies = input<RebalancingPolicyDto[]>([]);

  readonly requestActivate = output<string>();
  readonly createPolicy = output<RebalancingPolicyCreatePayload>();

  readonly showCreate = signal<boolean>(false);
  readonly form = signal<CreateForm>(EMPTY_FORM);

  readonly typeBadgeMap: Record<PolicyType, PolicyTypeBadge> = {
    calendar: { label: 'Calendar', colorClass: 'bg-chart-1/15 text-chart-1' },
    threshold: { label: 'Threshold', colorClass: 'bg-chart-4/15 text-chart-4' },
    hybrid: { label: 'Hybrid', colorClass: 'bg-chart-3/15 text-chart-3' },
  };

  readonly isFormValid = computed(() => {
    const f = this.form();
    if (f.name.trim().length < 2) return false;
    if (f.policy_type === 'calendar' && f.frequency_days <= 0) return false;
    if (f.policy_type === 'threshold' && (f.threshold < 0 || f.threshold >= 1)) return false;
    return true;
  });

  getPolicyBadge(type: PolicyType): PolicyTypeBadge {
    return this.typeBadgeMap[type];
  }

  activate(id: string): void {
    this.requestActivate.emit(id);
  }

  toggleCreate(): void {
    this.showCreate.update((v) => !v);
    this.form.set(EMPTY_FORM);
  }

  updateField<K extends keyof CreateForm>(key: K, value: CreateForm[K]): void {
    this.form.update((f) => ({ ...f, [key]: value }));
  }

  submit(): void {
    if (!this.isFormValid()) return;
    this.createPolicy.emit(this.buildPayload());
    this.showCreate.set(false);
    this.form.set(EMPTY_FORM);
  }

  private buildPayload(): RebalancingPolicyCreatePayload {
    const f = this.form();
    return { name: f.name.trim(), policy_type: f.policy_type, config: this.buildConfig(f) };
  }

  private buildConfig(f: CreateForm): Record<string, unknown> {
    if (f.policy_type === 'calendar') return { frequency_days: f.frequency_days };
    if (f.policy_type === 'threshold') {
      return { threshold: f.threshold, kind: f.kind };
    }
    return {
      frequency_days: f.frequency_days,
      threshold: f.threshold,
      kind: f.kind,
    };
  }
}
