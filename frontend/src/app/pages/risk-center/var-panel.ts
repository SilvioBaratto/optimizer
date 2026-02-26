import {
  Component,
  input,
  output,
  computed,
  ElementRef,
  viewChild,
  effect,
  OnDestroy,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsBarComponent, BarData } from '../../shared/echarts-bar/echarts-bar';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import { MOCK_CONCENTRATION } from '../../mocks/risk-mocks';
import { MOCK_VAR_RESULTS } from '../../mocks/risk-mocks';
import { generateDailyTimeSeries } from '../../mocks/generators';
import type { VaRMethod, VaRResult } from '../../models/risk.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

@Component({
  selector: 'app-var-panel',
  imports: [StatCardComponent, EchartsBarComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './var-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class VarPanelComponent implements OnDestroy {
  varResults = input<VaRResult[]>([]);
  selectedMethod = input<VaRMethod>('historical');
  selectedConfidence = input(0.95);

  methodChange = output<VaRMethod>();
  confidenceChange = output<number>();

  private fmt = inject(FormatService);
  private readonly chartContainer = viewChild<ElementRef<HTMLElement>>('varTimeChart');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  readonly methods: { value: VaRMethod; label: string }[] = [
    { value: 'historical', label: 'Historical' },
    { value: 'parametric', label: 'Parametric' },
    { value: 'monte_carlo', label: 'Monte Carlo' },
  ];

  readonly confidenceLevels = [0.95, 0.99];

  activeResult = computed(() => {
    const results = this.varResults();
    const conf = this.selectedConfidence();
    return results.find(r => r.confidence === conf) ?? results[0];
  });

  varDollarStr = computed(() => {
    const r = this.activeResult();
    return r ? this.fmt.formatCurrencyCompact(r.varDollar) : '--';
  });

  cvarDollarStr = computed(() => {
    const r = this.activeResult();
    return r ? this.fmt.formatCurrencyCompact(r.cvarDollar) : '--';
  });

  varAumRatio = computed(() => {
    const r = this.activeResult();
    return r ? this.fmt.formatPercent(Math.abs(r.var)) : '--';
  });

  breakdownData = computed<BarData[]>(() =>
    MOCK_CONCENTRATION.slice(0, 10).map(c => ({
      label: c.ticker,
      value: c.componentVar,
    })),
  );

  comparisonColumns: TableColumn[] = [
    { key: 'method', label: 'Method', sortable: true },
    { key: 'confidence', label: 'Confidence', type: 'percentage', sortable: true },
    { key: 'var', label: 'VaR', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'cvar', label: 'CVaR', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'varDollar', label: 'VaR ($)', type: 'currency', sortable: true },
    { key: 'cvarDollar', label: 'CVaR ($)', type: 'currency', sortable: true },
  ];

  comparisonRows = computed(() =>
    MOCK_VAR_RESULTS.map(r => ({
      method: r.method === 'monte_carlo' ? 'Monte Carlo' : r.method.charAt(0).toUpperCase() + r.method.slice(1),
      confidence: r.confidence,
      var: r.var,
      cvar: r.cvar,
      varDollar: r.varDollar,
      cvarDollar: r.cvarDollar,
    })),
  );

  private readonly varTimeSeries = generateDailyTimeSeries('2025-06-01', 180, -0.02, 0.16, -0.018, 99);

  constructor() {
    effect((onCleanup) => {
      const container = this.chartContainer();
      this.selectedMethod();
      this.selectedConfidence();
      if (container && !this.chart) {
        void this.initChart();
      } else if (this.chart) {
        this.chart.setOption(this.buildTimeSeriesOption());
      }
      onCleanup(() => {
        this.ro?.disconnect();
        this.chart?.dispose();
        this.chart = undefined;
        this.ro = undefined;
      });
    });
  }

  private async initChart() {
    const container = this.chartContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, MarkLineComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildTimeSeriesOption());

    this.ro = new ResizeObserver(() => {
      this.chart?.resize();
      this.chart?.setOption(this.buildTimeSeriesOption());
    });
    this.ro.observe(el);
  }

  private buildTimeSeriesOption(): EChartsCoreOption {
    const style = getComputedStyle(document.documentElement);
    const chartColor = style.getPropertyValue('--color-chart-1').trim();
    const lossColor = style.getPropertyValue('--color-loss').trim();

    const labels = this.varTimeSeries.map(p => p.date);
    const values = this.varTimeSeries.map(p => +(p.value * 100).toFixed(4));
    const threshold = -(this.selectedConfidence() === 0.99 ? 3.2 : 1.8);

    const containerWidth = this.chartContainer()?.nativeElement.clientWidth ?? 800;
    const isNarrow = containerWidth < 500;

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: ${p.value.toFixed(4)}%`;
        },
      },
      grid: { left: isNarrow ? 40 : 50, right: 16, top: 16, bottom: isNarrow ? 48 : 32 },
      xAxis: { type: 'category', data: labels },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(2)}%` },
      },
      series: [
        {
          type: 'line',
          data: values,
          symbol: 'none',
          lineStyle: { width: 1.5, color: chartColor },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: `${chartColor}1f` },
                { offset: 1, color: `${chartColor}00` },
              ],
            },
          },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              {
                yAxis: threshold,
                lineStyle: { color: lossColor, type: 'dashed', width: 2 },
                label: {
                  formatter: `VaR Limit ${threshold}%`,
                  position: 'insideEndTop',
                  fontSize: 10,
                  color: lossColor,
                },
              },
            ],
          },
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
