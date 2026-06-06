import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import type { PipelineStepId } from '../../core/models/pipeline-builder.model';
import {
  BUILDER_STAGES,
  type BuilderStage,
  STAGE_LABELS,
} from '../models/builder-stage';
import { stageSummary } from '../../pipeline-stepper/chip-summary';
import {
  type Tab,
  TabGroupComponent,
} from '../../shared/components/tab-group/tab-group';

interface StageChip {
  readonly stage: BuilderStage;
  readonly summary: string;
}

@Component({
  selector: 'app-stage-strip',
  imports: [TabGroupComponent],
  template: `
    <app-tab-group
      [tabs]="tabs()"
      [activeTab]="activeTab()"
      (tabChange)="onTabChange($event)"
    />
    @if (chips().length > 0) {
      <div data-region="stage-chips" class="flex flex-wrap gap-2 px-3 pb-2">
        @for (chip of chips(); track chip.stage) {
          <span
            class="inline-flex items-center px-2 py-0.5 rounded-full bg-surface-inset text-text-secondary text-data-xs border border-border"
            [attr.data-stage]="chip.stage"
          >{{ chip.summary }}</span>
        }
      </div>
    }
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StageStripComponent {
  readonly currentStage = input.required<BuilderStage | null>();
  readonly stepResults =
    input.required<Map<PipelineStepId, Record<string, unknown> | null>>();
  readonly stageSelect = output<BuilderStage>();

  readonly tabs = computed<Tab[]>(() =>
    BUILDER_STAGES.map((id) => ({ id, label: STAGE_LABELS[id] })),
  );

  readonly activeTab = computed<BuilderStage>(
    () => this.currentStage() ?? 'universe',
  );

  readonly chips = computed<StageChip[]>(() => {
    const results = this.stepResults();
    return BUILDER_STAGES.map((stage) => ({
      stage,
      summary: stageSummary(stage, results),
    })).filter((c) => c.summary.length > 0);
  });

  onTabChange(id: string): void {
    this.stageSelect.emit(id as BuilderStage);
  }
}
