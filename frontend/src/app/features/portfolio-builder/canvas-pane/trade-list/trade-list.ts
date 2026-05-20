import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
} from '@angular/core';

import {
  isGreyedByFlags,
  type FlagInstance,
  type TradeAction,
} from '../../../../models/drift.model';
import { DiagnosticBadgeComponent } from '../../../../shared/diagnostic-badge/diagnostic-badge.component';
import { BuilderStore } from '../../builder.store';

export interface TradeListRow {
  readonly ticker: string;
  readonly action: TradeAction;
  readonly delta_weight: number;
  readonly delta_eur: number;
  readonly est_shares: number;
  readonly est_cost_eur: number;
  readonly flags: readonly FlagInstance[];
  readonly greyed: boolean;
}

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

const ACTION_BADGE: Readonly<
  Record<TradeAction, { readonly label: string; readonly className: string }>
> = {
  buy: { label: 'BUY', className: 'bg-gain/15 text-gain' },
  sell: { label: 'SELL', className: 'bg-loss/15 text-loss' },
  hold: { label: 'HOLD', className: 'bg-surface-inset text-text-secondary' },
};

@Component({
  selector: 'app-trade-list',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DiagnosticBadgeComponent],
  template: `
    @if (rows().length > 0) {
      <section
        data-region="trade-list"
        class="flex h-full w-full flex-col gap-2"
      >
        <table
          data-region="trade-table"
          class="w-full border-collapse text-data-xs"
        >
          <thead class="bg-surface-raised text-text-tertiary">
            <tr>
              <th class="px-2 py-1 text-left font-medium">Ticker</th>
              <th class="px-2 py-1 text-left font-medium">Action</th>
              <th class="px-2 py-1 text-right font-medium">Δ %</th>
              <th class="px-2 py-1 text-right font-medium">Δ EUR</th>
              <th class="px-2 py-1 text-right font-medium">Est. Shares</th>
              <th class="px-2 py-1 text-right font-medium">Est. Cost</th>
              <th class="px-2 py-1 text-left font-medium">Flags</th>
            </tr>
          </thead>
          <tbody>
            @for (row of rows(); track row.ticker) {
              <tr
                data-region="trade-row"
                [attr.data-row-ticker]="row.ticker"
                [attr.data-row-greyed]="row.greyed"
                [class]="rowClass(row)"
              >
                <td class="px-2 py-1 font-medium text-text">{{ row.ticker }}</td>
                <td class="px-2 py-1">
                  <span
                    data-cell="action-badge"
                    [class]="actionBadgeClass(row.action)"
                  >
                    {{ actionLabel(row.action) }}
                  </span>
                </td>
                <td
                  class="px-2 py-1 text-right tabular-nums"
                  [class]="signColor(row.delta_weight)"
                >
                  {{ formatPct(row.delta_weight) }}
                </td>
                <td
                  class="px-2 py-1 text-right tabular-nums"
                  [class]="signColor(row.delta_eur)"
                >
                  {{ formatEur(row.delta_eur) }}
                </td>
                <td class="px-2 py-1 text-right tabular-nums text-text">
                  {{ formatShares(row.est_shares) }}
                </td>
                <td class="px-2 py-1 text-right tabular-nums text-text">
                  {{ formatEur(row.est_cost_eur) }}
                </td>
                <td class="px-2 py-1">
                  <div
                    data-cell="flag-badges"
                    class="flex flex-wrap items-center gap-1"
                  >
                    @for (flag of row.flags; track flag.code) {
                      <app-diagnostic-badge [flag]="flag" />
                    }
                  </div>
                </td>
              </tr>
            }
          </tbody>
        </table>
        <div
          data-region="trade-footer"
          class="grid grid-cols-3 gap-3 rounded-lg border border-border bg-surface-raised px-3 py-2 text-data-xs"
        >
          <div class="flex flex-col">
            <span class="text-text-tertiary uppercase tracking-wide">Turnover</span>
            <span
              data-cell="footer-turnover"
              class="tabular-nums font-medium text-text"
            >
              {{ formatPct(footer().turnoverPct) }}
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-text-tertiary uppercase tracking-wide">Est. cost</span>
            <span
              data-cell="footer-cost"
              class="tabular-nums font-medium text-text"
            >
              {{ formatEur(footer().estCostEur) }}
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-text-tertiary uppercase tracking-wide">Residual cash</span>
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

  readonly rows: Signal<TradeListRow[]> = computed(() => this.toRows());
  readonly footer: Signal<TradeListFooter> = computed(() => this.toFooter());

  private toRows(): TradeListRow[] {
    const trades = this.store.trades()?.trades ?? [];
    return [...trades]
      .sort((a, b) => Math.abs(b.delta_eur) - Math.abs(a.delta_eur))
      .map((t) => ({
        ticker: t.ticker,
        action: t.action,
        delta_weight: t.delta_weight,
        delta_eur: t.delta_eur,
        est_shares: t.est_shares,
        est_cost_eur: t.est_cost_eur,
        flags: t.flags,
        greyed: isGreyedByFlags(t.flags),
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

  protected rowClass(row: TradeListRow): string {
    const base = 'border-t border-border-muted';
    return row.greyed ? `${base} opacity-50` : base;
  }

  protected actionBadgeClass(action: TradeAction): string {
    return `inline-flex items-center rounded-md px-2 py-0.5 text-data-xs font-medium ${ACTION_BADGE[action].className}`;
  }

  protected actionLabel(action: TradeAction): string {
    return ACTION_BADGE[action].label;
  }

  protected signColor(value: number): string {
    if (value > 0) return 'text-gain';
    if (value < 0) return 'text-loss';
    return 'text-flat';
  }

  protected formatPct(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }

  protected formatEur(value: number): string {
    return `€${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  }

  protected formatShares(value: number): string {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
}
