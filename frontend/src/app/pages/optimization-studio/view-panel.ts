import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HelpTooltipComponent } from '../../shared/help-tooltip/help-tooltip';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsBarComponent } from '../../shared/echarts-bar/echarts-bar';
import { BarData } from '../../shared/bar-chart/bar-chart';
import { ViewFormation } from '../../models/optimization.model';
import { MOCK_VIEW_FORMATIONS, MOCK_MOMENT_OUTPUT } from '../../mocks/optimization-mocks';

type ViewFramework = 'none' | 'black_litterman' | 'entropy_pooling';

const VIEW_FRAMEWORK_LABELS: Record<ViewFramework, string> = {
  none: 'None',
  black_litterman: 'Black-Litterman',
  entropy_pooling: 'Entropy Pooling',
};

@Component({
  selector: 'app-view-panel',
  imports: [FormsModule, HelpTooltipComponent, DataTableComponent, EchartsBarComponent],
  templateUrl: './view-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ViewPanelComponent {
  readonly viewFrameworkOptions: ViewFramework[] = ['none', 'black_litterman', 'entropy_pooling'];
  readonly viewFrameworkLabels = VIEW_FRAMEWORK_LABELS;

  selectedViewFramework = signal<ViewFramework>('black_litterman');
  views = signal<ViewFormation[]>(MOCK_VIEW_FORMATIONS);

  readonly viewTableColumns: TableColumn[] = [
    { key: 'type', label: 'Type', type: 'badge', badgeMap: {
      absolute: { value: 'Absolute', colorClass: 'inline-flex px-1.5 py-0.5 rounded text-xs font-medium bg-accent/10 text-accent' },
      relative: { value: 'Relative', colorClass: 'inline-flex px-1.5 py-0.5 rounded text-xs font-medium bg-text-tertiary/10 text-text-secondary' },
    }},
    { key: 'assetsDisplay', label: 'Assets' },
    { key: 'valueDisplay', label: 'Value (%)', align: 'right' },
    { key: 'confidenceDisplay', label: 'Confidence', align: 'right' },
    { key: 'source', label: 'Source' },
  ];

  readonly viewTableRows = computed(() =>
    this.views().map(v => ({
      ...v,
      assetsDisplay: v.assets.join(' vs '),
      valueDisplay: `${(v.value * 100).toFixed(1)}%`,
      confidenceDisplay: `${(v.confidence * 100).toFixed(0)}%`,
    })),
  );

  readonly priorBarData = computed<BarData[]>(() =>
    MOCK_MOMENT_OUTPUT.expectedReturns.map(e => ({ label: e.ticker, value: e.value })),
  );

  readonly posteriorBarData = computed<BarData[]>(() => {
    const views = this.views();
    return MOCK_MOMENT_OUTPUT.expectedReturns.map(e => {
      const view = views.find(v => v.assets.includes(e.ticker));
      const adjustment = view ? (view.value - e.value) * view.confidence * 0.3 : 0;
      return { label: e.ticker, value: +(e.value + adjustment).toFixed(4) };
    });
  });

  setViewFramework(value: ViewFramework): void {
    this.selectedViewFramework.set(value);
  }

  addView(): void {
    const newView: ViewFormation = {
      id: `v${Date.now()}`,
      type: 'absolute',
      assets: ['AAPL'],
      value: 0.10,
      confidence: 0.5,
      source: 'Manual',
    };
    this.views.update(views => [...views, newView]);
  }
}
