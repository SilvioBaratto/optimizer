import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
} from '@angular/core';

import type { DriftRow, PositionFlag, TradeAction } from '../../../../models/drift.model';
import {
  DataTableComponent,
  type TableColumn,
} from '../../../../shared/data-table/data-table';
import { BuilderStore } from '../../builder.store';

export type TradeListRow = {
  readonly ticker: string;
  readonly action: TradeAction;
  readonly delta_weight: number;
  readonly delta_eur: number;
  readonly est_shares: number;
  readonly est_cost_eur: number;
  readonly flags: string;
} & Record<string, unknown>;

export interface TradeListFooter {
  readonly turnoverPct: number;
  readonly estCostEur: number;
  readonly residualCashEur: number;
}

const ZERO_FOOTER: TradeListFooter = {
  turnoverPct: 0,
  estCostEur: 0,
  residualCashEur: 0,
};

const COLUMNS: TableColumn[] = [
  { key: 'ticker', label: 'Ticker', sortable: true },
  {
    key: 'action',
    label: 'Action',
    type: 'badge',
    sortable: true,
    badgeMap: {
      buy: { value: 'BUY', colorClass: 'bg-gain/15 text-gain' },
      sell: { value: 'SELL', colorClass: 'bg-loss/15 text-loss' },
      hold: { value: 'HOLD', colorClass: 'bg-surface-inset text-text-secondary' },
    },
  },
  {
    key: 'delta_weight',
    label: 'Δ %',
    type: 'percentage',
    sortable: true,
    align: 'right',
    colorBySign: true,
  },
  {
    key: 'delta_eur',
    label: 'Δ EUR',
    type: 'currency',
    sortable: true,
    align: 'right',
    currency: 'EUR',
    colorBySign: true,
  },
  { key: 'est_shares', label: 'Est. Shares', type: 'number', align: 'right' },
  {
    key: 'est_cost_eur',
    label: 'Est. Cost',
    type: 'currency',
    align: 'right',
    currency: 'EUR',
  },
  { key: 'flags', label: 'Flags' },
];

function flagsByTicker(
  rows: readonly DriftRow[] | undefined,
): Map<string, readonly PositionFlag[]> {
  const map = new Map<string, readonly PositionFlag[]>();
  rows?.forEach((r) => map.set(r.ticker, r.flags));
  return map;
}

@Component({
  selector: 'app-trade-list',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DataTableComponent],
  template: `
    @if (rows().length > 0) {
      <section
        data-region="trade-list"
        class="flex h-full w-full flex-col gap-2"
      >
        <app-data-table
          [columns]="columns"
          [rows]="rows()"
          [total]="rows().length"
          [showExport]="true"
          [exportFilename]="exportFilename()"
          [dense]="true"
        />
        <div
          data-region="trade-footer"
          class="grid grid-cols-3 gap-3 rounded-lg border border-border bg-surface-raised px-3 py-2 text-data-xs"
        >
          <div class="flex flex-col">
            <span class="text-text-tertiary uppercase tracking-wide">
              Turnover
            </span>
            <span
              data-cell="footer-turnover"
              class="tabular-nums font-medium text-text"
            >
              {{ formatPct(footer().turnoverPct) }}
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-text-tertiary uppercase tracking-wide">
              Est. cost
            </span>
            <span
              data-cell="footer-cost"
              class="tabular-nums font-medium text-text"
            >
              {{ formatEur(footer().estCostEur) }}
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-text-tertiary uppercase tracking-wide">
              Residual cash
            </span>
            <span
              data-cell="footer-residual"
              class="tabular-nums font-medium text-text"
            >
              {{ formatEur(footer().residualCashEur) }}
            </span>
          </div>
        </div>
      </section>
    } @else {
      <div
        data-region="trade-empty"
        class="flex h-[120px] w-full items-center justify-center rounded-lg border border-dashed border-border text-data-xs text-text-tertiary"
      >
        No trades suggested
      </div>
    }
  `,
})
export class TradeListComponent {
  protected readonly store = inject(BuilderStore);
  protected readonly columns = COLUMNS;

  readonly rows: Signal<TradeListRow[]> = computed(() => this.toRows());
  readonly footer: Signal<TradeListFooter> = computed(() => this.toFooter());
  readonly exportFilename = computed(
    () => `drift-trades-${this.store.portfolioName() ?? 'portfolio'}`,
  );

  private toRows(): TradeListRow[] {
    const trades = this.store.trades()?.trades ?? [];
    const flagMap = flagsByTicker(this.store.drift()?.drift);
    return [...trades]
      .sort((a, b) => Math.abs(b.delta_eur) - Math.abs(a.delta_eur))
      .map((t) => ({
        ticker: t.ticker,
        action: t.action,
        delta_weight: t.delta_weight,
        delta_eur: t.delta_eur,
        est_shares: t.est_shares,
        est_cost_eur: t.est_cost_eur,
        flags: (flagMap.get(t.ticker) ?? []).join(', '),
      }));
  }

  private toFooter(): TradeListFooter {
    const totals = this.store.drift()?.totals;
    if (!totals) return ZERO_FOOTER;
    return {
      turnoverPct: totals.total_drift_abs,
      estCostEur: totals.buy_eur + totals.sell_eur,
      residualCashEur: totals.deployable_eur - totals.buy_eur + totals.sell_eur,
    };
  }

  protected formatPct(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }

  protected formatEur(value: number): string {
    return `€${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  }
}
