import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import type { FreshnessLevel } from '../../models/jobs.model';

@Component({
  selector: 'app-freshness-indicator',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span class="inline-flex items-center gap-1.5">
      <span class="w-2 h-2 rounded-full" [class]="dotClass()"></span>
      <span class="text-xs font-medium" [class]="labelClass()">{{ displayLabel() }}</span>
    </span>
  `,
})
export class FreshnessIndicatorComponent {
  level = input.required<FreshnessLevel>();
  ageLabel = input('');

  dotClass = computed(() => {
    const map: Record<FreshnessLevel, string> = {
      fresh: 'bg-gain',
      stale: 'bg-warning',
      critical: 'bg-loss animate-pulse',
      unknown: 'bg-text-tertiary',
    };
    return map[this.level()];
  });

  labelClass = computed(() => {
    const map: Record<FreshnessLevel, string> = {
      fresh: 'text-gain',
      stale: 'text-warning',
      critical: 'text-loss',
      unknown: 'text-text-tertiary',
    };
    return map[this.level()];
  });

  displayLabel = computed(() => {
    if (this.level() === 'unknown') return 'No data';
    return this.ageLabel() || this.level();
  });
}
