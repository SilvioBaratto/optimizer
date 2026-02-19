import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-progress-bar',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div>
      <div class="flex items-center justify-between mb-1">
        <span class="text-xs font-medium text-text-secondary">{{ label() }}</span>
        <span class="text-xs text-text-tertiary">{{ pct() }}%</span>
      </div>
      <div class="w-full h-1.5 bg-surface-inset rounded-full overflow-hidden">
        <div class="h-full bg-accent rounded-full transition-all duration-300"
             [style.width.%]="pct()"></div>
      </div>
      @if (detail()) {
        <p class="mt-1 text-xs text-text-tertiary">{{ detail() }}</p>
      }
    </div>
  `,
})
export class ProgressBarComponent {
  label = input<string>('');
  current = input(0);
  total = input(100);
  detail = input<string>('');

  pct = computed(() => {
    const t = this.total();
    return t > 0 ? Math.round((this.current() / t) * 100) : 0;
  });
}
