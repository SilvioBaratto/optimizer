import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';

export interface BarData {
  label: string;
  value: number;
}

@Component({
  selector: 'app-bar-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg [attr.width]="'100%'" [attr.height]="height()" [attr.viewBox]="viewBox()">
      <!-- Zero line -->
      <line [attr.x1]="0" [attr.y1]="zeroY()" [attr.x2]="totalWidth()" [attr.y2]="zeroY()"
            stroke="#e4e4e7" stroke-width="0.5" />
      <!-- Bars -->
      @for (bar of bars(); track bar.label; let i = $index) {
        <rect [attr.x]="i * barStep() + barPadding()"
              [attr.y]="bar.value >= 0 ? barY(bar) : zeroY()"
              [attr.width]="barWidth()"
              [attr.height]="barHeight(bar)"
              [attr.fill]="bar.value >= 0 ? '#16a34a' : '#dc2626'"
              rx="1" />
      }
    </svg>
  `,
})
export class BarChartComponent {
  data = input<BarData[]>([]);
  height = input(120);

  private maxAbs = computed(() => {
    const vals = this.data().map(d => Math.abs(d.value));
    return vals.length > 0 ? Math.max(...vals) : 1;
  });

  totalWidth = computed(() => this.data().length * 20);
  barStep = computed(() => this.totalWidth() / Math.max(this.data().length, 1));
  barPadding = computed(() => this.barStep() * 0.15);
  barWidth = computed(() => this.barStep() * 0.7);

  zeroY = computed(() => this.height() / 2);

  viewBox = computed(() => `0 0 ${this.totalWidth()} ${this.height()}`);

  bars = computed(() => this.data());

  barY(bar: BarData): number {
    if (bar.value >= 0) {
      const scale = (this.height() / 2 - 4) / this.maxAbs();
      return this.zeroY() - bar.value * scale;
    }
    return this.zeroY();
  }

  barHeight(bar: BarData): number {
    const scale = (this.height() / 2 - 4) / this.maxAbs();
    return Math.max(Math.abs(bar.value) * scale, 1);
  }
}
