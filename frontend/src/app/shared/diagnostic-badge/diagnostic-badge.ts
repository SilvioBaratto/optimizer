import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import type { FlagInstance, PositionFlag } from '../../core/models/drift.model';

interface BadgeMeta {
  readonly label: string;
  readonly iconPath: string;
  readonly className: string;
}

const BADGE_META: Readonly<Record<PositionFlag, BadgeMeta>> = {
  unmapped: {
    label: 'UNMAPPED',
    iconPath: 'M12 4a8 8 0 100 16 8 8 0 000-16zm0 5a2 2 0 011.8 3h-3.6A2 2 0 0112 9zm0 8a1 1 0 100-2 1 1 0 000 2z',
    className: 'bg-flag-unmapped-bg text-flag-unmapped',
  },
  fx_missing: {
    label: 'FX',
    iconPath: 'M4 6h16M4 12h10M4 18h16',
    className: 'bg-flag-fx-missing-bg text-flag-fx-missing',
  },
  stale_price: {
    label: 'STALE',
    iconPath: 'M12 6v6l4 2m6-2a10 10 0 11-20 0 10 10 0 0120 0z',
    className: 'bg-flag-stale-bg text-flag-stale',
  },
  target_not_on_broker: {
    label: 'NO BROKER',
    iconPath: 'M12 9v4m0 4h.01M5 21h14a2 2 0 001.7-3L13.7 5a2 2 0 00-3.4 0L3.3 18A2 2 0 005 21z',
    className: 'bg-flag-target-gap-bg text-flag-target-gap',
  },
  reconciliation_mismatch: {
    label: 'RECON',
    iconPath: 'M12 2L2 7l10 5 10-5-10-5zm0 13l-7-3.5V17l7 3.5L19 17v-5.5L12 15z',
    className: 'bg-flag-reconciliation-bg text-flag-reconciliation',
  },
};

@Component({
  selector: 'app-diagnostic-badge',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span
      [attr.data-flag-code]="flag().code"
      [class]="containerClass()"
      [title]="tooltipText()"
    >
      <svg
        class="h-3 w-3 shrink-0"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path [attr.d]="meta().iconPath" />
      </svg>
      <span data-region="badge-label">{{ meta().label }}</span>
    </span>
  `,
  styles: [
    `
      :host {
        display: inline-flex;
      }
    `,
  ],
})
export class DiagnosticBadgeComponent {
  readonly flag = input.required<FlagInstance>();

  protected readonly meta = computed<BadgeMeta>(() => BADGE_META[this.flag().code]);

  protected readonly containerClass = computed(
    () =>
      `inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-data-xs font-medium ${this.meta().className}`,
  );

  protected readonly tooltipText = computed(() => {
    const f = this.flag();
    return f.reference ? `${f.reason} (${f.reference})` : f.reason;
  });
}
