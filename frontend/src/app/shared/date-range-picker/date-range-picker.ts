import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
} from '@angular/core';

import {
  DateRangePreset,
  PortfolioContextService,
} from '../../services/portfolio-context.service';

const PRESETS: readonly DateRangePreset[] = [
  '1D', '1W', '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'Max',
] as const;

const BTN_ACTIVE =
  'px-2 py-0.5 rounded bg-accent text-surface-raised font-medium';
const BTN_INACTIVE =
  'px-2 py-0.5 rounded text-text-secondary hover:text-text hover:bg-surface-inset transition-colors';

@Component({
  selector: 'app-date-range-picker',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="relative flex items-center gap-1 text-data-sm">
      @for (preset of presets; track preset) {
        <button
          type="button"
          data-testid="preset-btn"
          [class]="activePreset() === preset ? activeClass : inactiveClass"
          (click)="selectPreset(preset)"
        >
          {{ preset }}
        </button>
      }
      <button
        type="button"
        data-testid="custom-toggle"
        [class]="isCustom() ? activeClass : inactiveClass"
        (click)="toggleCustom()"
      >
        Custom
      </button>

      @if (showCustomPanel()) {
        <div
          data-testid="custom-panel"
          class="absolute top-full right-0 mt-1 z-20 flex items-center gap-2 rounded-md border border-border bg-surface-raised p-2 shadow-md"
        >
          <input
            type="date"
            data-testid="custom-start"
            aria-label="Start date"
            class="rounded border border-border bg-surface px-2 py-1 text-data-sm text-text focus:outline-none focus:ring-1 focus:ring-accent"
            [value]="startStr()"
            (input)="onStartInput($event)"
          />
          <span aria-hidden="true" class="text-text-tertiary">—</span>
          <input
            type="date"
            data-testid="custom-end"
            aria-label="End date"
            class="rounded border border-border bg-surface px-2 py-1 text-data-sm text-text focus:outline-none focus:ring-1 focus:ring-accent"
            [value]="endStr()"
            (input)="onEndInput($event)"
          />
          <button
            type="button"
            data-testid="custom-apply"
            class="px-3 py-1 rounded bg-accent text-surface-raised font-medium disabled:opacity-40 disabled:cursor-not-allowed"
            [disabled]="!canApply()"
            (click)="applyCustomRange()"
          >
            Apply
          </button>
        </div>
      }
    </div>
  `,
})
export class DateRangePickerComponent {
  private readonly ctx = inject(PortfolioContextService);

  readonly presets = PRESETS;
  readonly activeClass = BTN_ACTIVE;
  readonly inactiveClass = BTN_INACTIVE;

  readonly showCustomPanel = signal(false);
  readonly startStr = signal('');
  readonly endStr = signal('');

  readonly activePreset = computed(() => this.ctx.dateRange().preset);
  readonly isCustom = computed(() => this.activePreset() === 'Custom');

  readonly canApply = computed(() => {
    const start = this.parseDate(this.startStr());
    const end = this.parseDate(this.endStr());
    return start !== null && end !== null && start < end;
  });

  selectPreset(preset: DateRangePreset): void {
    this.showCustomPanel.set(false);
    this.ctx.setPreset(preset);
  }

  toggleCustom(): void {
    this.showCustomPanel.update((v) => !v);
  }

  onStartInput(event: Event): void {
    this.startStr.set((event.target as HTMLInputElement).value);
  }

  onEndInput(event: Event): void {
    this.endStr.set((event.target as HTMLInputElement).value);
  }

  applyCustomRange(): void {
    const start = this.parseDate(this.startStr());
    const end = this.parseDate(this.endStr());
    if (start === null || end === null || start >= end) return;
    this.ctx.setCustomRange(start, end);
    this.showCustomPanel.set(false);
  }

  private parseDate(value: string): Date | null {
    if (!value) return null;
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
}
