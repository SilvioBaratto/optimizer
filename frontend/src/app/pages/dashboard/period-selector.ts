import {
  ChangeDetectionStrategy,
  Component,
  input,
  output,
} from '@angular/core';

export type DashboardPeriod = '1Y' | '3Y' | '5Y' | 'MAX';

const PERIOD_OPTIONS: readonly DashboardPeriod[] = ['1Y', '3Y', '5Y', 'MAX'];

@Component({
  selector: 'app-period-selector',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="inline-flex items-center gap-1 bg-surface-inset rounded-lg p-1"
      role="tablist" aria-label="Dashboard period selector">
      @for (opt of options; track opt) {
        <button type="button"
          [attr.aria-selected]="value() === opt"
          [attr.data-testid]="'period-' + opt"
          (click)="select(opt)"
          class="px-3 py-1 text-xs font-medium rounded-md transition-colors"
          [class]="value() === opt
            ? 'bg-surface-raised text-text shadow-sm'
            : 'text-text-secondary hover:text-text'">
          {{ opt }}
        </button>
      }
    </div>
  `,
})
export class PeriodSelectorComponent {
  readonly value = input<DashboardPeriod>('3Y');
  readonly valueChange = output<DashboardPeriod>();

  readonly options = PERIOD_OPTIONS;

  select(opt: DashboardPeriod): void {
    if (opt !== this.value()) this.valueChange.emit(opt);
  }
}
