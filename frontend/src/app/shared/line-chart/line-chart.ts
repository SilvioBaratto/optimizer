import {
  Component,
  input,
  ElementRef,
  viewChild,
  afterNextRender,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { createChart, LineSeries, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts';

export interface LineChartData {
  time: string;
  value: number;
}

@Component({
  selector: 'app-line-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #chartContainer class="w-full"></div>`,
})
export class LineChartComponent implements OnDestroy {
  data = input<LineChartData[]>([]);
  height = input(300);
  color = input('#18181b');

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('chartContainer');
  private chart?: IChartApi;
  private series?: ISeriesApi<'Line'>;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const d = this.data();
      if (this.series !== undefined && d.length > 0) {
        this.series.setData(d as LineData<Time>[]);
        this.chart?.timeScale().fitContent();
      }
    });
  }

  private initChart() {
    const el = this.container().nativeElement;
    this.chart = createChart(el, {
      height: this.height(),
      layout: {
        background: { color: 'transparent' },
        textColor: '#71717a',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#f4f4f5' },
        horzLines: { color: '#f4f4f5' },
      },
      rightPriceScale: { borderColor: '#e4e4e7' },
      timeScale: { borderColor: '#e4e4e7' },
      crosshair: {
        vertLine: { color: '#a1a1aa', width: 1, style: 2 },
        horzLine: { color: '#a1a1aa', width: 1, style: 2 },
      },
    });

    this.series = this.chart.addSeries(LineSeries, {
      color: this.color(),
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    const d = this.data();
    if (d.length > 0) {
      this.series.setData(d as LineData<Time>[]);
      this.chart.timeScale().fitContent();
    }

    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        this.chart?.applyOptions({ width: entry.contentRect.width });
      }
    });
    ro.observe(el);
  }

  ngOnDestroy() {
    this.chart?.remove();
  }
}
