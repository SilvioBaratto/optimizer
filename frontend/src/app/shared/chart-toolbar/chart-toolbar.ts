import {
  Component,
  input,
  output,
  signal,
  contentChild,
  ElementRef,
  viewChild,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';

@Component({
  selector: 'app-chart-toolbar',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div #wrapper class="relative">
      <div class="flex items-center justify-end gap-1 mb-1">
        @if (showExportPng()) {
          <button
            type="button"
            class="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-raised)] transition-colors"
            title="Export PNG"
            (click)="onExportPng()"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          </button>
        }
        @if (showExportSvg()) {
          <button
            type="button"
            class="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-raised)] transition-colors"
            title="Export SVG"
            (click)="onExportSvg()"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          </button>
        }
        @if (showFullscreen()) {
          <button
            type="button"
            class="p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-raised)] transition-colors"
            title="Toggle fullscreen"
            (click)="onFullscreenToggle()"
          >
            @if (isFullscreen()) {
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
            } @else {
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
            }
          </button>
        }
        @if (showDataTable()) {
          <button
            type="button"
            class="p-1.5 rounded transition-colors"
            [class]="isDataTableVisible() ? 'text-[var(--color-chart-1)] bg-[var(--color-surface-raised)]' : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-raised)]'"
            title="Toggle data table"
            (click)="onDataTableToggle()"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
          </button>
        }
      </div>
      <ng-content />
    </div>
  `,
})
export class ChartToolbarComponent implements OnDestroy {
  showExportPng = input(true);
  showExportSvg = input(true);
  showFullscreen = input(true);
  showDataTable = input(true);

  exportPng = output<void>();
  exportSvg = output<void>();
  fullscreenToggle = output<boolean>();
  showDataTableToggle = output<boolean>();

  isFullscreen = signal(false);
  isDataTableVisible = signal(false);

  private readonly wrapper = viewChild.required<ElementRef<HTMLElement>>('wrapper');
  private readonly childChart = contentChild(CHART_EXPORTABLE);

  private readonly onFullscreenChange = () => {
    this.isFullscreen.set(!!document.fullscreenElement);
  };

  constructor() {
    document.addEventListener('fullscreenchange', this.onFullscreenChange);
  }

  onExportPng() {
    const chart = this.childChart()?.getChartInstance();
    if (!chart) {
      this.exportPng.emit();
      return;
    }
    const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
    this.triggerDownload(url, 'chart.png');
    this.exportPng.emit();
  }

  onExportSvg() {
    const chart = this.childChart()?.getChartInstance();
    if (!chart) {
      this.exportSvg.emit();
      return;
    }
    // True SVG export requires ECharts SVGRenderer — using PNG fallback until then.
    const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
    this.triggerDownload(url, 'chart.png');
    this.exportSvg.emit();
  }

  onFullscreenToggle() {
    const el = this.wrapper().nativeElement;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      el.requestFullscreen();
    }
    this.fullscreenToggle.emit(!this.isFullscreen());
  }

  onDataTableToggle() {
    this.isDataTableVisible.update(v => !v);
    this.showDataTableToggle.emit(this.isDataTableVisible());
  }

  private triggerDownload(url: string, filename: string) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  ngOnDestroy() {
    document.removeEventListener('fullscreenchange', this.onFullscreenChange);
  }
}
