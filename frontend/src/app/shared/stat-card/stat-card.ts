import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-stat-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="bg-surface-raised border border-border rounded-lg p-4">
      <p class="text-xs font-medium text-text-secondary uppercase tracking-wide">{{ label() }}</p>
      <p class="mt-1 text-2xl font-semibold text-text font-mono tabular-nums">{{ value() }}</p>
      @if (subtitle()) {
        <p class="mt-0.5 text-xs text-text-tertiary">{{ subtitle() }}</p>
      }
      @if (sparklineData().length > 1) {
        <svg class="mt-2 w-full h-12" viewBox="0 0 100 48" preserveAspectRatio="none">
          <path [attr.d]="sparklinePath()" fill="none" [class]="sparklineStrokeClass()" stroke-width="1.5" />
        </svg>
      }
      @if (delta() !== null) {
        <div class="mt-2 flex items-center gap-1.5 text-xs">
          <span [class]="deltaColorClass()">
            @if (trend() === 'up') { &#9650; }
            @else if (trend() === 'down') { &#9660; }
            @else { &#9644; }
            {{ deltaFormatted() }}
          </span>
          <span class="text-text-tertiary">{{ deltaLabel() }}</span>
        </div>
      }
    </div>
  `,
})
export class StatCardComponent {
  label = input.required<string>();
  value = input.required<string | number>();
  subtitle = input<string>('');
  sparklineData = input<number[]>([]);
  delta = input<number | null>(null);
  trend = input<'up' | 'down' | 'flat'>('flat');
  deltaLabel = input<string>('vs prior period');

  sparklinePath = computed(() => {
    const data = this.sparklineData();
    if (data.length < 2) return '';

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    return data
      .map((v, i) => {
        const x = (i / (data.length - 1)) * 100;
        const y = 48 - ((v - min) / range) * 44 - 2;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  });

  sparklineStrokeClass = computed(() => {
    const map: Record<string, string> = {
      up: 'stroke-gain',
      down: 'stroke-loss',
      flat: 'stroke-text-tertiary',
    };
    return map[this.trend()];
  });

  deltaFormatted = computed(() => {
    const d = this.delta();
    if (d == null) return '';
    const pct = (d * 100).toFixed(2);
    return d > 0 ? `+${pct}%` : `${pct}%`;
  });

  deltaColorClass = computed(() => {
    const map: Record<string, string> = {
      up: 'text-gain',
      down: 'text-loss',
      flat: 'text-flat',
    };
    return map[this.trend()];
  });
}
