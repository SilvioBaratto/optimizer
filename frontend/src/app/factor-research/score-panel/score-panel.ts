import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';

import { FormatService } from '../../core/services/format.service';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import {
  computeQuintiles,
  type CompositeMethod,
  type FactorScoreApiResponse,
  type FactorScoreRequest,
  type ScoreRow,
} from '../factor.model';

const COMPOSITE_METHODS: readonly CompositeMethod[] = [
  'equal_weight',
  'ic_weighted',
  'icir_weighted',
  'ridge_weighted',
  'gbt_weighted',
];

const ML_METHODS: ReadonlySet<CompositeMethod> = new Set<CompositeMethod>([
  'ridge_weighted',
  'gbt_weighted',
]);

const TRAINING_WINDOW_DAYS = 365;

@Component({
  selector: 'app-score-panel',
  imports: [FormsModule, DataTableComponent],
  templateUrl: './score-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ScorePanelComponent {
  private readonly fmt = inject(FormatService);

  readonly tickers = input<string[]>([]);
  readonly scoreResult = input<FactorScoreApiResponse | null>(null);
  readonly loading = input<boolean>(false);
  readonly errorMessage = input<string | null>(null);

  readonly runScore = output<FactorScoreRequest>();

  readonly methods = COMPOSITE_METHODS;
  readonly method = signal<CompositeMethod>('equal_weight');

  readonly columns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'score', label: 'Score', align: 'right', sortable: true, type: 'ratio' },
    { key: 'quintile', label: 'Quintile', align: 'right', sortable: true, type: 'number' },
  ];

  readonly scoreRows = computed<ScoreRow[]>(() => {
    const result = this.scoreResult();
    if (!result) return [];
    return computeQuintiles(result.scores);
  });

  readonly tableRows = computed<Record<string, unknown>[]>(() =>
    this.scoreRows().map((row) => ({ ...row })),
  );

  run(): void {
    this.runScore.emit(this.buildPayload());
  }

  private buildPayload(): FactorScoreRequest {
    const method = this.method();
    const scoreDate = this.fmt.todayIso();
    const payload: FactorScoreRequest = {
      tickers: this.tickers(),
      score_date: scoreDate,
      composite_method: method,
    };
    if (ML_METHODS.has(method)) {
      payload.training_start_date = daysBeforeIso(scoreDate, TRAINING_WINDOW_DAYS);
      payload.training_end_date = scoreDate;
    }
    return payload;
  }
}

function daysBeforeIso(iso: string, days: number): string {
  const d = new Date(iso);
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}
