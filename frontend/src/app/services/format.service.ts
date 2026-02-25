import { Injectable } from '@angular/core';

export interface FormattedReturn {
  text: string;
  colorClass: string;
}

@Injectable({ providedIn: 'root' })
export class FormatService {
  formatReturn(value: number | null | undefined): FormattedReturn {
    if (value == null || isNaN(value)) {
      return { text: '--', colorClass: 'text-flat' };
    }
    const pct = (value * 100).toFixed(2);
    if (value > 0) {
      return { text: `+${pct}%`, colorClass: 'text-gain' };
    }
    if (value < 0) {
      return { text: `${pct}%`, colorClass: 'text-loss' };
    }
    return { text: `${pct}%`, colorClass: 'text-flat' };
  }

  formatCurrency(value: number | null | undefined, currency = 'USD'): string {
    if (value == null || isNaN(value)) return '--';
    return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(value);
  }

  formatBps(value: number | null | undefined): string {
    if (value == null || isNaN(value)) return '--';
    return `${Math.round(value * 10000)} bps`;
  }

  formatRatio(value: number | null | undefined, decimals = 2): string {
    if (value == null || isNaN(value)) return '--';
    return value.toFixed(decimals);
  }

  formatPercent(value: number | null | undefined, decimals = 2): string {
    if (value == null || isNaN(value)) return '--';
    return `${(value * 100).toFixed(decimals)}%`;
  }

  formatInteger(value: number | null | undefined): string {
    if (value == null || isNaN(value)) return '--';
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);
  }

  formatDate(date: Date | string | null | undefined, format: 'medium' | 'short' | 'iso' = 'medium'): string {
    if (date == null) return '--';
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return '--';

    switch (format) {
      case 'iso':
        return d.toISOString().slice(0, 10);
      case 'short':
        return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' }).format(d);
      case 'medium':
        return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }).format(d);
    }
  }

  formatDuration(ms: number | null | undefined): string {
    if (ms == null || isNaN(ms)) return '--';
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }
}
