import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import type {
  OptimizeStepResult,
  ReportStepResult,
} from '../../../models/pipeline-builder.model';

@Component({
  selector: 'app-analytics-pane',
  template: `
    <section
      data-region="analytics-pane"
      class="flex h-full w-full items-center justify-center p-6"
    >
      <div
        class="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border px-8 py-12 text-center"
      >
        <span class="text-data-sm font-medium text-text-secondary">
          Risk &amp; Exposure &mdash; Phase 2
        </span>
        <span class="text-data-xs text-text-tertiary">
          Charts arrive in Cycle 3.
        </span>
      </div>
    </section>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AnalyticsPaneComponent {
  readonly riskMetrics =
    input<ReportStepResult['metrics'] | undefined>(undefined);
  readonly exposures =
    input<OptimizeStepResult['sector_breakdown'] | undefined>(undefined);
}
