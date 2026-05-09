import {
  Component,
  input,
  output,
  signal,
  computed,
  effect,
  contentChild,
  inject,
  ElementRef,
  viewChild,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';

interface DataTableSnapshot {
  columns: string[];
  rows: (string | number)[][];
}

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
      @if (isDataTableVisible() && dataTable(); as t) {
        @if (t.rows.length === 0) {
          <p class="mt-3 text-xs text-text-secondary italic">No tabular data available for this chart.</p>
        } @else {
          <div class="mt-3 max-h-64 overflow-auto border border-border rounded-md">
            <table class="w-full text-data-xs">
              <thead class="bg-surface-inset sticky top-0">
                <tr>
                  @for (col of t.columns; track col) {
                    <th class="px-2 py-1 text-left font-medium text-text-secondary whitespace-nowrap">{{ col }}</th>
                  }
                </tr>
              </thead>
              <tbody>
                @for (row of t.rows; track $index) {
                  <tr class="border-t border-border-muted">
                    @for (cell of row; track $index) {
                      <td class="px-2 py-1 font-mono tabular-nums text-text whitespace-nowrap">{{ cell }}</td>
                    }
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      }
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

  // Backing signal updated by an effect that resolves the chart instance
  // (sync via injected ChartExportable, fallback async via ECharts'
  // getInstanceByDom for projected raw <div> charts).
  private readonly dataTableSig = signal<DataTableSnapshot | null>(null);
  readonly dataTable = computed<DataTableSnapshot | null>(() => this.dataTableSig());

  private getInstanceByDom:
    | ((el: HTMLElement) => { getOption: () => unknown } | undefined | null)
    | null = null;
  private getInstanceByDomLoading = false;

  private async ensureGetInstanceByDom(): Promise<typeof this.getInstanceByDom> {
    if (this.getInstanceByDom || this.getInstanceByDomLoading) return this.getInstanceByDom;
    this.getInstanceByDomLoading = true;
    try {
      const mod = (await import('echarts/core')) as unknown as {
        getInstanceByDom: (el: HTMLElement) => unknown;
      };
      this.getInstanceByDom = (el: HTMLElement) => {
        const inst = mod.getInstanceByDom(el);
        if (inst && typeof (inst as { getOption?: unknown }).getOption === 'function') {
          return inst as { getOption: () => unknown };
        }
        return null;
      };
    } catch {
      this.getInstanceByDom = null;
    }
    return this.getInstanceByDom;
  }

  private async refreshDataTable(): Promise<void> {
    if (!this.isDataTableVisible()) {
      this.dataTableSig.set(null);
      return;
    }
    const wrapper = this.wrapper().nativeElement;
    let chart = this.resolveChart()?.getChartInstance() as { getOption: () => unknown } | undefined;
    if (!chart) {
      const get = await this.ensureGetInstanceByDom();
      const host = wrapper.querySelector('[_echarts_instance_]') as HTMLElement | null;
      if (get && host) chart = get(host) ?? undefined;
    }
    if (!chart) {
      this.dataTableSig.set({ columns: [], rows: [] });
      return;
    }
    this.dataTableSig.set(extractChartTable(chart));
  }

  private readonly wrapper = viewChild.required<ElementRef<HTMLElement>>('wrapper');
  private readonly childChart = contentChild(CHART_EXPORTABLE);
  // Fallback: when the projected content is a raw <div> (no Angular component
  // providing CHART_EXPORTABLE), walk the injector hierarchy to find a parent
  // that provides it (e.g. the page component).
  private readonly ancestorChart = inject(CHART_EXPORTABLE, {
    optional: true,
    skipSelf: true,
  });

  private resolveChart(): ChartExportable | null {
    return this.childChart() ?? this.ancestorChart ?? null;
  }

  private readonly onFullscreenChange = () => {
    this.isFullscreen.set(!!document.fullscreenElement);
  };

  constructor() {
    document.addEventListener('fullscreenchange', this.onFullscreenChange);
    effect(() => {
      const _ = this.isDataTableVisible();
      void this.refreshDataTable();
    });
  }

  onExportPng() {
    const chart = this.resolveChart()?.getChartInstance();
    if (!chart) {
      this.exportPng.emit();
      return;
    }
    const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
    this.triggerDownload(url, 'chart.png');
    this.exportPng.emit();
  }

  onExportSvg() {
    const chart = this.resolveChart()?.getChartInstance();
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

function extractChartTable(chart: { getOption: () => unknown } | undefined | null): DataTableSnapshot {
  const empty: DataTableSnapshot = { columns: [], rows: [] };
  if (!chart) return empty;
  let opts: unknown;
  try {
    opts = chart.getOption();
  } catch {
    return empty;
  }
  if (!opts || typeof opts !== 'object') return empty;
  const o = opts as { xAxis?: unknown; series?: unknown };
  const xAxisRaw = Array.isArray(o.xAxis) ? o.xAxis[0] : o.xAxis;
  const labels = (xAxisRaw && typeof xAxisRaw === 'object' && Array.isArray((xAxisRaw as { data?: unknown }).data)
    ? ((xAxisRaw as { data: unknown[] }).data as (string | number)[])
    : []);
  const seriesArr = Array.isArray(o.series) ? (o.series as { name?: string; data?: unknown }[]) : [];
  if (seriesArr.length === 0) return empty;

  const columns = ['#', ...seriesArr.map((s, i) => s.name ?? `Series ${i + 1}`)];
  const maxLen = Math.max(labels.length, ...seriesArr.map((s) => Array.isArray(s.data) ? s.data.length : 0));
  const rows: (string | number)[][] = [];
  for (let i = 0; i < maxLen; i++) {
    const row: (string | number)[] = [labels[i] ?? i];
    for (const s of seriesArr) {
      const val = Array.isArray(s.data) ? s.data[i] : undefined;
      row.push(formatTableCell(val));
    }
    rows.push(row);
  }
  return { columns, rows };
}

function formatTableCell(value: unknown): string | number {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    return Math.abs(value) >= 1000 || Math.abs(value) < 0.01
      ? value.toExponential(2)
      : value.toFixed(4);
  }
  if (Array.isArray(value)) return value.map(formatTableCell).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
