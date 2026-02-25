import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';
import type { RebalancingPolicy, RebalancingTrigger } from '../../models/rebalancing.model';

interface PolicyTypeBadge {
  label: string;
  colorClass: string;
}

@Component({
  selector: 'app-policy-panel',
  templateUrl: './policy-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PolicyPanelComponent {
  policies = input<RebalancingPolicy[]>([]);
  activatePolicy = output<string>();

  readonly typeBadgeMap: Record<RebalancingTrigger, PolicyTypeBadge> = {
    calendar: { label: 'Calendar', colorClass: 'bg-blue-500/15 text-blue-400' },
    threshold: { label: 'Threshold', colorClass: 'bg-amber-500/15 text-amber-400' },
    hybrid: { label: 'Hybrid', colorClass: 'bg-purple-500/15 text-purple-400' },
  };

  getPolicyBadge(trigger: RebalancingTrigger): PolicyTypeBadge {
    return this.typeBadgeMap[trigger];
  }
}
