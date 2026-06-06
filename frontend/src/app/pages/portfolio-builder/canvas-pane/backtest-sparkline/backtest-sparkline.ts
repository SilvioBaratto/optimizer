import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  Signal,
  afterNextRender,
  computed,
  effect,
  inject,
  viewChild,
} from '@angular/core';
import type { EChartsCoreOption, EChartsType } from 'echarts/core';

import { readCssVar } from '../../../../shared/charts/echarts-theme';
import {
  MetricColorPipe,
  safeNum,
} from '../../../../shared/pipes/metric-color.pipe';
import { BuilderStore } from '../../builder.store';

export interface DrawdownRegion {
  readonly startIdx: number;
  readonly endIdx: number;
}

export interface BacktestStats {
  readonly cumReturn: number | null;
  readonly maxDd: number | null;
  readonly sharpe: number | null;
}

export function computeDrawdownRegion(
  values: readonly number[],
): DrawdownRegion | null {
  if (values.length === 0) return null;
  let peak = values[0];
  let peakIdx = 0;
  let worst = 0;
  let startIdx = 0;
  let endIdx = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v > peak) {
      peak = v;
      peakIdx = i;
    }
    if (peak === 0) continue;
    const dd = (v - peak) / peak;
    if (dd < worst) {
      worst = dd;
      startIdx = peakIdx;
      endIdx = i;
    }
  }
  return worst < 0 ? { startIdx, endIdx } : null;
}

@Component({
  selector: 'app-backtest-sparkline',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MetricColorPipe],
  template: `
    @let s = stats();
    @let cum = s.cumReturn | metricColor: 'percent';
    @let dd = s.maxDd | metricColor: 'percent';
    @let sh = s.sharpe | metricColor: 'ratio';
    <div class="flex flex-col gap-2">
      <div
        data-region="backtest-stats"
        class="grid grid-cols-3 gap-2 rounded-md border border-border px-3 py-2"
      >
        <span data-stat="cum-return" class="flex flex-col text-data-xs">
          <span class="text-text-tertiary">Cum Return</span>
          <span
            class="tabular-nums text-data-sm"
            [class]="cum.colorClass"
          >
            {{ cum.text }}
          </span>
        </span>
        <span data-stat="max-dd" class="flex flex-col text-data-xs">
          <span class="text-text-tertiary">Max DD</span>
          <span
            class="tabular-nums text-data-sm"
            [class]="dd.colorClass"
          >
            {{ dd.text }}
          </span>
        </span>
        <span data-stat="sharpe" class="flex flex-col text-data-xs">
          <span class="text-text-tertiary">Sharpe</span>
          <span
            class="tabular-nums text-data-sm"
            [class]="sh.colorClass"
          >
            {{ sh.text }}
          </span>
        </span>
      </div>
      @if (hasData()) {
        <div
          #container
          data-region="backtest-sparkline-chart"
          class="w-full"
          [style.height.px]="240"
        ></div>
      } @else {
        <div
          data-region="backtest-sparkline-empty"
          class="flex h-[240px] w-full items-center justify-center rounded-lg border border-dashed border-border text-data-xs text-text-tertiary"
        >
          No backtest yet
        </div>
      }
    </div>
  `,
})
export class BacktestSparklineComponent implements OnDestroy {
  private readonly store = inject(BuilderStore);
  private readonly container = viewChild<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  readonly hasData: Signal<boolean> = computed(
    () => Object.keys(this.store.backtest()?.equityCurve ?? {}).length > 0,
  );

  readonly labels: Signal<string[]> = computed(() =>
    Object.keys(this.store.backtest()?.equityCurve ?? {}),
  );

  readonly values: Signal<number[]> = computed(() =>
    Object.values(this.store.backtest()?.equityCurve ?? {}).map(toPercent),
  );

  readonly stats: Signal<BacktestStats> = computed(() => {
    const s = this.store.backtest()?.summaryStats;
    return {
      cumReturn: safeNum(s?.['totalReturn']),
      maxDd: safeNum(s?.['maxDrawdown']),
      sharpe: safeNum(s?.['sharpe']),
    };
  });

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => this.renderIfReady());
  }

  buildOptionForTest(): EChartsCoreOption {
    return this.buildOption();
  }

  ngOnDestroy(): void {
    this.disposeChart();
  }

  private renderIfReady(): void {
    void this.labels();
    void this.values();
    if (!this.hasData()) {
      this.disposeChart();
      return;
    }
    if (!this.chart) return;
    this.chart.setOption(this.buildOption(), true);
  }

  private async initChart(): Promise<void> {
    if (!this.hasData()) return;
    const el = this.container()?.nativeElement;
    if (!el) return;

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, MarkAreaComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([
      LineChart,
      GridComponent,
      TooltipComponent,
      MarkAreaComponent,
      CanvasRenderer,
    ]);

    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption());

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(): EChartsCoreOption {
    const labels = this.labels();
    const values = this.values();
    const region = computeDrawdownRegion(values);
    const lineColor = readCssVar('--color-chart-1') || '#6366f1';
    const lossColor = readCssVar('--color-loss') || '#ef4444';

    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 48, right: 16, top: 16, bottom: 32 },
      xAxis: { type: 'category', data: labels },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      series: [buildLineSeries(values, lineColor, region, labels, lossColor)],
    };
  }

  private disposeChart(): void {
    this.ro?.disconnect();
    this.ro = undefined;
    this.chart?.dispose();
    this.chart = undefined;
  }
}

function toPercent(value: number | null): number {
  return value == null ? 0 : value * 100;
}

function buildLineSeries(
  values: number[],
  lineColor: string,
  region: DrawdownRegion | null,
  labels: string[],
  lossColor: string,
): unknown {
  return {
    type: 'line',
    data: values,
    symbol: 'circle',
    symbolSize: 4,
    lineStyle: { color: lineColor, width: 2 },
    itemStyle: { color: lineColor },
    markArea: region ? buildMarkArea(region, labels, lossColor) : undefined,
  };
}

function buildMarkArea(
  region: DrawdownRegion,
  labels: string[],
  lossColor: string,
): unknown {
  const startKey = labels[region.startIdx] ?? region.startIdx;
  const endKey = labels[region.endIdx] ?? region.endIdx;
  return {
    silent: true,
    itemStyle: { color: `${lossColor}33` },
    data: [[{ xAxis: startKey }, { xAxis: endKey }]],
  };
}
