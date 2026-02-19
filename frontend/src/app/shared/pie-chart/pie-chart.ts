import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';

export interface PieSegment {
  label: string;
  value: number;
  color: string;
}

const COLORS = [
  '#18181b', '#3f3f46', '#52525b', '#71717a', '#a1a1aa',
  '#d4d4d8', '#27272a', '#404040', '#525252', '#737373',
];

interface ArcPath {
  d: string;
  color: string;
  label: string;
  pct: number;
  labelX: number;
  labelY: number;
}

@Component({
  selector: 'app-pie-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="flex items-start gap-6">
      <svg [attr.width]="size()" [attr.height]="size()" [attr.viewBox]="'0 0 ' + size() + ' ' + size()">
        @for (arc of arcs(); track arc.label) {
          <path [attr.d]="arc.d" [attr.fill]="arc.color" stroke="white" stroke-width="1.5" />
        }
      </svg>
      <div class="flex flex-col gap-1.5 min-w-0 pt-1">
        @for (arc of arcs(); track arc.label) {
          <div class="flex items-center gap-2 text-xs">
            <span class="w-2.5 h-2.5 rounded-sm shrink-0" [style.background]="arc.color"></span>
            <span class="text-text-secondary truncate">{{ arc.label }}</span>
            <span class="text-text font-medium ml-auto">{{ arc.pct }}%</span>
          </div>
        }
      </div>
    </div>
  `,
})
export class PieChartComponent {
  segments = input<PieSegment[]>([]);
  size = input(160);

  arcs = computed<ArcPath[]>(() => {
    const segs = this.segments();
    const total = segs.reduce((s, seg) => s + seg.value, 0);
    if (total === 0) return [];

    const r = this.size() / 2;
    const cx = r;
    const cy = r;
    const radius = r - 2;
    let startAngle = -Math.PI / 2;
    const paths: ArcPath[] = [];

    for (let i = 0; i < segs.length; i++) {
      const seg = segs[i];
      const pct = Math.round((seg.value / total) * 1000) / 10;
      const sweep = (seg.value / total) * Math.PI * 2;
      const endAngle = startAngle + sweep;
      const largeArc = sweep > Math.PI ? 1 : 0;

      const x1 = cx + radius * Math.cos(startAngle);
      const y1 = cy + radius * Math.sin(startAngle);
      const x2 = cx + radius * Math.cos(endAngle);
      const y2 = cy + radius * Math.sin(endAngle);

      const midAngle = startAngle + sweep / 2;
      const labelR = radius * 0.65;

      paths.push({
        d: `M${cx},${cy} L${x1},${y1} A${radius},${radius} 0 ${largeArc} 1 ${x2},${y2} Z`,
        color: seg.color || COLORS[i % COLORS.length],
        label: seg.label,
        pct,
        labelX: cx + labelR * Math.cos(midAngle),
        labelY: cy + labelR * Math.sin(midAngle),
      });
      startAngle = endAngle;
    }
    return paths;
  });
}
