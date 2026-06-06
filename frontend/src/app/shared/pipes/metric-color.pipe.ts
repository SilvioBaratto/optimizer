import { Pipe, PipeTransform, inject } from '@angular/core';
import { FormatService } from '../../core/services/format.service';

export type MetricKind = 'return' | 'percent' | 'ratio' | 'bps';

export interface MetricFormatted {
  text: string;
  colorClass: string;
}

export function safeNum(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function signClass(value: number): string {
  if (value > 0) return 'text-gain';
  if (value < 0) return 'text-loss';
  return 'text-flat';
}

@Pipe({ name: 'metricColor', pure: true })
export class MetricColorPipe implements PipeTransform {
  private readonly fmt = inject(FormatService);

  transform(value: unknown, kind: MetricKind = 'return'): MetricFormatted {
    const num = safeNum(value);
    if (num === null) {
      return { text: '--', colorClass: '' };
    }
    return { text: this.formatText(num, kind), colorClass: signClass(num) };
  }

  private formatText(value: number, kind: MetricKind): string {
    switch (kind) {
      case 'return':
        return this.fmt.formatReturn(value).text;
      case 'percent':
        return this.fmt.formatPercent(value);
      case 'ratio':
        return this.fmt.formatRatio(value);
      case 'bps':
        return this.fmt.formatBps(value);
    }
  }
}
