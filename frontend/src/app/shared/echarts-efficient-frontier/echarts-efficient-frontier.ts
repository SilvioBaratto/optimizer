import {
  Component,
  input,
  output,
  computed,
  ElementRef,
  viewChild,
  afterNextRender,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

import { readCssVar } from '../charts/echarts-theme';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';
import type { EfficientFrontierPoint } from '../../models/optimization.model';

interface FrontierClickPayload {
  seriesName?: string;
  dataIndex?: number;
}

const FRONTIER_SERIES_NAME = 'Efficient Frontier';
const CAL_SERIES_NAME = 'Capital Allocation Line';

@Component({
  selector: 'app-echarts-efficient-frontier',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (hasData()) {
      <div
        #container
        data-testid="efficient-frontier-chart"
        class="w-full"
        [style.height.px]="height()"
      ></div>
    } @else {
      <div
        data-testid="efficient-frontier-empty"
        class="flex h-full w-full items-center justify-center py-12 text-sm text-text-secondary"
        [style.height.px]="height()"
      >
        No efficient frontier data yet.
      </div>
    }
  `,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsEfficientFrontierComponent }],
})
export class EchartsEfficientFrontierComponent implements OnDestroy, ChartExportable {
  readonly frontier = input.required<EfficientFrontierPoint[]>();
  readonly optimal = input<EfficientFrontierPoint | null>(null);
  readonly assetMarkers = input<EfficientFrontierPoint[]>([]);
  readonly riskFreeRate = input<number | null>(null);
  readonly height = input(320);

  readonly pointClick = output<EfficientFrontierPoint>();

  readonly hasData = computed(() => this.frontier().length > 0);

  private readonly container = viewChild<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => this.renderIfReady());
  }

  getChartInstance(): EChartsType | undefined {
    return this.chart;
  }

  ngOnDestroy(): void {
    this.disposeChart();
  }

  buildOptionForTest(): EChartsCoreOption {
    return this.buildOption();
  }

  handleFrontierClickForTest(payload: FrontierClickPayload): void {
    this.emitFrontierClick(payload);
  }

  private renderIfReady(): void {
    // Track dependencies to re-run on changes.
    void this.frontier();
    void this.optimal();
    void this.assetMarkers();
    void this.riskFreeRate();

    if (!this.hasData()) {
      this.disposeChart();
      return;
    }
    if (!this.chart) {
      return;
    }
    this.chart.setOption(this.buildOption(), true);
  }

  private async initChart(): Promise<void> {
    if (!this.hasData()) {
      return;
    }
    const el = this.container()?.nativeElement;
    if (!el) {
      return;
    }

    const { init, use } = await import('echarts/core');
    const { ScatterChart, LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption());
    this.chart.on('click', (params: unknown) => this.emitFrontierClick(params as FrontierClickPayload));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(): EChartsCoreOption {
    const palette = readFrontierPalette();
    const series: unknown[] = [buildFrontierSeries(this.frontier(), palette.frontier)];

    const optimal = this.optimal();
    if (optimal) {
      series.push(buildScatterSeries(optimal, 'Optimal', palette.optimal));
    }

    for (const marker of this.assetMarkers()) {
      series.push(buildScatterSeries(marker, marker.label ?? 'Marker', palette.marker));
    }

    const cal = this.maybeBuildCalSeries(palette.cal);
    if (cal) {
      series.push(cal);
    }

    return {
      legend: { show: true, top: 0, right: 0 },
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => formatTooltip(params),
      },
      grid: { left: 56, right: 16, top: 32, bottom: 40 },
      xAxis: buildAxis('Risk (\u03c3 %)', 'bottom'),
      yAxis: buildAxis('Return (%)', 'left'),
      series,
    };
  }

  private maybeBuildCalSeries(color: string): unknown | null {
    const rfr = this.riskFreeRate();
    if (rfr === null || rfr === undefined) {
      return null;
    }
    const anchor = this.optimal() ?? pickTangencyPoint(this.frontier());
    if (!anchor) {
      return null;
    }
    return buildCalSeries(rfr, anchor, color);
  }

  private emitFrontierClick(payload: FrontierClickPayload): void {
    if (payload.seriesName !== FRONTIER_SERIES_NAME) {
      return;
    }
    const idx = payload.dataIndex ?? -1;
    const point = this.frontier()[idx];
    if (!point) {
      return;
    }
    this.pointClick.emit(point);
  }

  private disposeChart(): void {
    this.ro?.disconnect();
    this.ro = undefined;
    this.chart?.dispose();
    this.chart = undefined;
  }
}

interface FrontierPalette {
  frontier: string;
  optimal: string;
  marker: string;
  cal: string;
}

function readFrontierPalette(): FrontierPalette {
  return {
    frontier: readCssVar('--color-chart-7') || '#6366f1',
    optimal: readCssVar('--color-chart-1') || '#f97316',
    marker: readCssVar('--color-chart-3') || '#10b981',
    cal: readCssVar('--color-chart-5') || '#94a3b8',
  };
}

function toPercentPair(p: EfficientFrontierPoint): [number, number] {
  return [+(p.risk * 100).toFixed(3), +(p.return * 100).toFixed(3)];
}

function buildFrontierSeries(points: EfficientFrontierPoint[], color: string): unknown {
  return {
    name: FRONTIER_SERIES_NAME,
    type: 'line',
    data: points.map(toPercentPair),
    smooth: true,
    symbol: 'circle',
    symbolSize: 6,
    lineStyle: { color, width: 2 },
    itemStyle: { color },
    z: 1,
  };
}

function buildScatterSeries(
  point: EfficientFrontierPoint,
  name: string,
  color: string,
): unknown {
  return {
    name,
    type: 'scatter',
    data: [toPercentPair(point)],
    symbolSize: 14,
    itemStyle: { color, borderColor: '#ffffff', borderWidth: 2 },
    label: {
      show: Boolean(point.label),
      formatter: point.label ?? '',
      position: 'top',
      fontSize: 11,
      fontWeight: 'bold',
    },
    z: 10,
  };
}

function buildCalSeries(
  riskFreeRate: number,
  anchor: EfficientFrontierPoint,
  color: string,
): unknown {
  const rfrPct = +(riskFreeRate * 100).toFixed(3);
  const [anchorRisk, anchorReturn] = toPercentPair(anchor);
  return {
    name: CAL_SERIES_NAME,
    type: 'line',
    data: [
      [0, rfrPct],
      [anchorRisk, anchorReturn],
    ],
    symbol: 'none',
    lineStyle: { color, width: 1.5, type: 'dashed' },
    itemStyle: { color },
    z: 2,
  };
}

function pickTangencyPoint(points: EfficientFrontierPoint[]): EfficientFrontierPoint | null {
  if (points.length === 0) {
    return null;
  }
  return points.reduce((best, p) => (p.sharpe > best.sharpe ? p : best), points[0]);
}

function buildAxis(name: string, side: 'bottom' | 'left'): unknown {
  return {
    type: 'value',
    name,
    nameLocation: 'middle',
    nameGap: side === 'bottom' ? 28 : 44,
    axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
  };
}

function formatTooltip(params: unknown): string {
  const p = params as { value?: [number, number]; seriesName?: string };
  const val = p.value ?? [0, 0];
  const name = p.seriesName ?? '';
  return `${name}<br/>Risk: ${val[0].toFixed(2)}%<br/>Return: ${val[1].toFixed(2)}%`;
}
