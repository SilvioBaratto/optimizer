import {
  Component,
  input,
  output,
  ElementRef,
  viewChild,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { EchartsHeatmapComponent } from '../../shared/echarts-heatmap/echarts-heatmap';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { generateDailyTimeSeries } from '../../mocks/generators';
import type { CorrelationData } from '../../models/risk.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

@Component({
  selector: 'app-correlation-panel',
  imports: [EchartsHeatmapComponent, ChartToolbarComponent],
  templateUrl: './correlation-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CorrelationPanelComponent implements OnDestroy {
  correlationData = input<CorrelationData>({ assets: [], matrix: [] });
  method = input<'pearson' | 'spearman' | 'kendall'>('pearson');
  methodChange = output<'pearson' | 'spearman' | 'kendall'>();

  private readonly rollingContainer = viewChild<ElementRef<HTMLElement>>('rollingChart');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  readonly methods = [
    { value: 'pearson' as const, label: 'Pearson' },
    { value: 'spearman' as const, label: 'Spearman' },
    { value: 'kendall' as const, label: 'Kendall' },
  ];

  private readonly rollingCorr1 = generateDailyTimeSeries('2025-06-01', 180, 0, 0.05, 0.45, 201);
  private readonly rollingCorr2 = generateDailyTimeSeries('2025-06-01', 180, 0, 0.04, 0.30, 202);

  constructor() {
    effect(() => {
      const container = this.rollingContainer();
      if (container && !this.chart) {
        void this.initChart();
      }
    });
  }

  private async initChart() {
    const container = this.rollingContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption());

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(): EChartsCoreOption {
    const style = getComputedStyle(document.documentElement);
    const color1 = style.getPropertyValue('--color-chart-1').trim();
    const color2 = style.getPropertyValue('--color-chart-3').trim();

    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['Avg Pairwise Corr', 'AAPL-MSFT Corr'], bottom: 0 },
      grid: { left: 50, right: 16, top: 16, bottom: 40 },
      xAxis: { type: 'category', data: this.rollingCorr1.map(p => p.date) },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        axisLabel: { formatter: (v: number) => v.toFixed(2) },
      },
      series: [
        {
          name: 'Avg Pairwise Corr',
          type: 'line',
          data: this.rollingCorr1.map(p => +p.value.toFixed(3)),
          symbol: 'none',
          lineStyle: { width: 1.5, color: color1 },
        },
        {
          name: 'AAPL-MSFT Corr',
          type: 'line',
          data: this.rollingCorr2.map(p => +p.value.toFixed(3)),
          symbol: 'none',
          lineStyle: { width: 1.5, color: color2 },
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
