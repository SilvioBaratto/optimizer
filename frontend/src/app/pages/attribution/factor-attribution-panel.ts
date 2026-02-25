import {
  Component,
  input,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { EchartsStackedAreaComponent, AreaSeries } from '../../shared/echarts-stacked-area/echarts-stacked-area';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import { seededRandom } from '../../mocks/generators';
import type { FactorAttribution } from '../../models/attribution.model';

const ROLLING_LABELS = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'];

@Component({
  selector: 'app-factor-attribution-panel',
  imports: [
    EchartsStackedAreaComponent,
    DataTableComponent,
    StatCardComponent,
    ChartToolbarComponent,
  ],
  templateUrl: './factor-attribution-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FactorAttributionPanelComponent {
  factors = input.required<FactorAttribution[]>();

  private fmt = inject(FormatService);

  readonly rollingLabels = ROLLING_LABELS;

  kpiRSquared = computed(() => {
    const all = this.factors();
    if (all.length === 0) return '—';
    const total = all.reduce((sum, f) => sum + Math.abs(f.contribution), 0);
    const residual = all.find(f => f.factor === 'Residual');
    const specific = residual ? Math.abs(residual.contribution) : 0;
    const factorExplained = total - specific;
    const rSquared = total > 0 ? factorExplained / total : 0;
    return this.fmt.formatRatio(rSquared);
  });

  kpiSpecificReturn = computed(() => {
    const residual = this.factors().find(f => f.factor === 'Residual');
    return residual ? this.fmt.formatPercent(residual.contribution) : '—';
  });

  kpiTotalFactorContribution = computed(() => {
    const total = this.factors().reduce((sum, f) => sum + f.contribution, 0);
    return this.fmt.formatPercent(total);
  });

  readonly factorTableColumns: TableColumn[] = [
    { key: 'factor', label: 'Factor', sortable: true },
    { key: 'exposure', label: 'Exposure', type: 'ratio', sortable: true },
    { key: 'factorReturn', label: 'Factor Return', type: 'percentage', sortable: true },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'cumulative', label: 'Cumulative', type: 'percentage', sortable: true },
  ];

  factorTableRows = computed(() =>
    this.factors().map(f => ({
      factor: f.factor,
      exposure: f.exposure,
      factorReturn: f.factorReturn,
      contribution: f.contribution,
      cumulative: f.cumulative,
    })),
  );

  rollingAreaSeries = computed<AreaSeries[]>(() => {
    const factors = this.factors();
    const rng = seededRandom(700);
    const months = ROLLING_LABELS.length;

    return factors.map(f => {
      const values: number[] = [];
      const basePerMonth = f.contribution / months;
      for (let i = 0; i < months; i++) {
        const noise = (rng() - 0.5) * Math.abs(basePerMonth) * 0.8;
        values.push(Math.round((basePerMonth + noise) * 1e6) / 1e6);
      }
      return { name: f.factor, values };
    });
  });
}
