import { Component, inject, ChangeDetectionStrategy } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import {
  PortfolioContextService,
  PortfolioMode,
  DateRangePreset,
} from '../../../services/portfolio-context.service';

@Component({
  selector: 'app-context-bar',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="border-b border-border bg-surface-raised px-4 py-2">
      <!-- Desktop: single row -->
      <div class="hidden md:flex items-center gap-3 text-data-sm">
        <!-- Mode segmented control -->
        <div class="flex rounded-md border border-border overflow-hidden">
          @for (mode of modes; track mode.value) {
            <button
              [class]="ctx.activeMode() === mode.value
                ? 'px-3 py-1 bg-accent text-surface-raised font-medium'
                : 'px-3 py-1 text-text-secondary hover:text-text hover:bg-surface-inset transition-colors'"
              (click)="ctx.setMode(mode.value)">
              {{ mode.label }}
            </button>
          }
        </div>

        <span class="text-border">|</span>

        <!-- Date range pills -->
        <div class="flex gap-1">
          @for (preset of datePresets; track preset) {
            <button
              [class]="ctx.dateRange().preset === preset
                ? 'px-2 py-0.5 rounded bg-accent text-surface-raised font-medium'
                : 'px-2 py-0.5 rounded text-text-secondary hover:text-text hover:bg-surface-inset transition-colors'"
              (click)="ctx.setPreset(preset)">
              {{ preset }}
            </button>
          }
        </div>

        <span class="text-border">|</span>

        <!-- Benchmark dropdown -->
        <div class="relative">
          <select
            class="appearance-none bg-surface-raised border border-border rounded px-3 py-1 pr-7 text-data-sm text-text cursor-pointer hover:border-accent transition-colors focus:outline-none focus:ring-1 focus:ring-accent"
            [value]="ctx.benchmark()"
            (change)="onBenchmarkChange($event)">
            @for (b of benchmarks; track b.value) {
              <option [value]="b.value">{{ b.label }}</option>
            }
          </select>
          <i-lucide name="chevron-down" [size]="12" class="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-text-tertiary" />
        </div>
      </div>

      <!-- Mobile: stacked -->
      <div class="md:hidden flex flex-col gap-2">
        <!-- Row 1: Mode -->
        <div class="flex rounded-md border border-border overflow-hidden text-data-sm">
          @for (mode of modes; track mode.value) {
            <button
              class="flex-1"
              [class]="ctx.activeMode() === mode.value
                ? 'px-3 py-1.5 bg-accent text-surface-raised font-medium flex-1'
                : 'px-3 py-1.5 text-text-secondary flex-1 hover:bg-surface-inset transition-colors'"
              (click)="ctx.setMode(mode.value)">
              {{ mode.label }}
            </button>
          }
        </div>

        <!-- Row 2: Date pills + benchmark -->
        <div class="flex items-center gap-2">
          <div class="flex gap-1 overflow-x-auto scrollbar-hide flex-1 text-data-sm">
            @for (preset of datePresets; track preset) {
              <button
                class="shrink-0"
                [class]="ctx.dateRange().preset === preset
                  ? 'px-2 py-0.5 rounded bg-accent text-surface-raised font-medium shrink-0'
                  : 'px-2 py-0.5 rounded text-text-secondary shrink-0 hover:bg-surface-inset transition-colors'"
                (click)="ctx.setPreset(preset)">
                {{ preset }}
              </button>
            }
          </div>

          <div class="relative shrink-0">
            <select
              class="appearance-none bg-surface-raised border border-border rounded px-2 py-0.5 pr-6 text-data-sm text-text cursor-pointer focus:outline-none focus:ring-1 focus:ring-accent"
              [value]="ctx.benchmark()"
              (change)="onBenchmarkChange($event)">
              @for (b of benchmarks; track b.value) {
                <option [value]="b.value">{{ b.label }}</option>
              }
            </select>
            <i-lucide name="chevron-down" [size]="12" class="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-text-tertiary" />
          </div>
        </div>
      </div>
    </div>
  `,
})
export class ContextBarComponent {
  protected readonly ctx = inject(PortfolioContextService);

  readonly modes: { label: string; value: PortfolioMode }[] = [
    { label: 'Live', value: 'live' },
    { label: 'Backtest', value: 'backtest' },
    { label: 'Paper', value: 'paper' },
  ];

  readonly datePresets: DateRangePreset[] = [
    '1D', '1W', '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'Max',
  ];

  readonly benchmarks = [
    { label: 'SPY', value: 'SPY' },
    { label: 'MSCI World (URTH)', value: 'URTH' },
    { label: '60/40 Balanced (VBINX)', value: 'VBINX' },
  ];

  onBenchmarkChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.ctx.setBenchmark(value);
  }
}
