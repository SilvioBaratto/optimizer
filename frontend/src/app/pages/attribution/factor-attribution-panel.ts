import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
} from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../services/format.service';
import type { FactorAttributionApiResponse } from '../../models/attribution.model';

@Component({
  selector: 'app-factor-attribution-panel',
  imports: [DataTableComponent, StatCardComponent],
  templateUrl: './factor-attribution-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FactorAttributionPanelComponent {
  readonly response = input<FactorAttributionApiResponse | null>(null);

  private readonly fmt = inject(FormatService);

  readonly hasData = computed(() => (this.response()?.factors.length ?? 0) > 0);

  readonly factorTableColumns: TableColumn[] = [
    { key: 'factorName', label: 'Factor', sortable: true },
    { key: 'exposure', label: 'Exposure', type: 'ratio', sortable: true, align: 'right' },
    { key: 'factorReturn', label: 'Factor Return', type: 'percentage', sortable: true, align: 'right' },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
  ];

  readonly factorTableRows = computed(() =>
    (this.response()?.factors ?? []).map((f) => ({
      factorName: f.factorName,
      exposure: f.exposure,
      factorReturn: f.factorReturn,
      contribution: f.contribution,
    }) as Record<string, unknown>),
  );

  readonly kpiPortfolioReturn = computed(() =>
    this.response() ? this.fmt.formatPercent(this.response()!.portfolioReturn) : '—',
  );

  readonly kpiExplained = computed(() =>
    this.response() ? this.fmt.formatPercent(this.response()!.explainedReturn) : '—',
  );

  readonly kpiResidual = computed(() =>
    this.response() ? this.fmt.formatPercent(this.response()!.residual) : '—',
  );

  formatPercent(v: number): string {
    return this.fmt.formatPercent(v);
  }
}
