import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import { StepStatus } from '../../../models/pipeline-builder.model';

// Collapsible shell for one pipeline step. Pure presentation: it owns no
// pipeline state and issues no HTTP. The host projects the step panel as
// content and only does so while the section is expanded (lazy mount).
@Component({
  selector: 'app-step-section',
  host: { class: 'block' },
  templateUrl: './step-section.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepSectionComponent {
  title = input.required<string>();
  description = input<string>('');
  status = input.required<StepStatus>();
  summary = input<string>('');
  expanded = input.required<boolean>();
  showContinue = input(false);
  showRetry = input(false);

  toggle = output<void>();
  continueStep = output<void>();
  retry = output<void>();

  protected readonly dotClass = computed(() => DOT[this.status()]);
  protected readonly badgeClass = computed(() => BADGE[this.status()]);
  protected readonly badgeLabel = computed(() => LABEL[this.status()]);
}

const DOT: Record<StepStatus, string> = {
  [StepStatus.Pending]: 'bg-text-tertiary',
  [StepStatus.Locked]: 'bg-text-tertiary opacity-40',
  [StepStatus.Ready]: 'bg-accent ring-2 ring-accent/30',
  [StepStatus.Running]: 'bg-accent animate-pulse',
  [StepStatus.Completed]: 'bg-gain',
  [StepStatus.Error]: 'bg-loss',
};

const BADGE: Record<StepStatus, string> = {
  [StepStatus.Pending]: 'bg-surface-inset text-text-tertiary border border-border',
  [StepStatus.Locked]: 'bg-surface-inset text-text-tertiary border border-border',
  [StepStatus.Ready]: 'bg-accent/10 text-accent border border-accent/40',
  [StepStatus.Running]: 'bg-accent/10 text-accent border border-accent/40',
  [StepStatus.Completed]: 'bg-gain/10 text-gain border border-gain/40',
  [StepStatus.Error]: 'bg-loss/10 text-loss border border-loss/40',
};

const LABEL: Record<StepStatus, string> = {
  [StepStatus.Pending]: 'Pending',
  [StepStatus.Locked]: 'Locked',
  [StepStatus.Ready]: 'Ready',
  [StepStatus.Running]: 'Running',
  [StepStatus.Completed]: 'Done',
  [StepStatus.Error]: 'Failed',
};
