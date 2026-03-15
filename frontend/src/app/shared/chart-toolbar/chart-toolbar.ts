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
import { LucideAngularModule } from 'lucide-angular';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';

@Component({
  selector: 'app-chart-toolbar',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div #wrapper class="relative">
      <div class="flex items-center justify-end gap-1.5 mb-2">
        @if (showExportPng()) {
          <button
            type="button"
            class="p-2 rounded-md text-(--color-text-tertiary) hover:text-(--color-text-secondary) hover:bg-surface-inset transition-colors"
            title="Export PNG"
            aria-label="Export chart as PNG"
            (click)="onExportPng()"
          >
            <i-lucide name="download" />
          </button>
        }
        @if (showExportSvg()) {
          <button
            type="button"
            class="p-2 rounded-md text-(--color-text-tertiary) hover:text-(--color-text-secondary) hover:bg-surface-inset transition-colors"
            title="Export SVG"
            aria-label="Export chart as SVG"
            (click)="onExportSvg()"
          >
            <i-lucide name="image" />
          </button>
        }
        @if (showFullscreen()) {
          <button
            type="button"
            class="p-2 rounded-md text-(--color-text-tertiary) hover:text-(--color-text-secondary) hover:bg-surface-inset transition-colors"
            title="Toggle fullscreen"
            aria-label="Toggle fullscreen"
            (click)="onFullscreenToggle()"
          >
            @if (isFullscreen()) {
              <i-lucide name="minimize" [size]="14" />
            } @else {
              <i-lucide name="maximize" [size]="14" />
            }
          </button>
        }
        @if (showDataTable()) {
          <button
            type="button"
            class="p-2 rounded-md transition-colors"
            [class]="
              isDataTableVisible()
                ? 'text-chart-1 bg-surface-inset'
                : 'text-(--color-text-tertiary) hover:text-(--color-text-secondary) hover:bg-surface-inset'
            "
            title="Toggle data table"
            aria-label="Toggle data table"
            (click)="onDataTableToggle()"
          >
            <i-lucide name="table" />
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
    this.isDataTableVisible.update((v) => !v);
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
