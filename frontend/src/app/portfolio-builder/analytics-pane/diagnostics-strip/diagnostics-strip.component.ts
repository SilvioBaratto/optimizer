import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import type {
  DriftDiagnostics,
  FlagInstance,
  PositionFlag,
} from '../../../models/drift.model';
import { DiagnosticBadgeComponent } from '../../../shared/diagnostic-badge/diagnostic-badge.component';

interface CategoryCount {
  readonly code: PositionFlag;
  readonly count: number;
  readonly flag: FlagInstance;
}

const CATEGORY_ORDER: ReadonlyArray<{
  code: PositionFlag;
  field: keyof Pick<
    DriftDiagnostics,
    | 'unmapped_count'
    | 'fx_missing_count'
    | 'stale_price_count'
    | 'target_not_on_broker_count'
  >;
}> = [
  { code: 'unmapped', field: 'unmapped_count' },
  { code: 'fx_missing', field: 'fx_missing_count' },
  { code: 'stale_price', field: 'stale_price_count' },
  { code: 'target_not_on_broker', field: 'target_not_on_broker_count' },
];

@Component({
  selector: 'app-diagnostics-strip',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DiagnosticBadgeComponent],
  template: `
    <section
      data-region="diagnostics-strip"
      class="flex flex-col gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-data-xs"
    >
      @if (nonZeroCategories().length > 0) {
        <div data-region="diagnostics-counts" class="flex flex-wrap items-center gap-3">
          @for (cat of nonZeroCategories(); track cat.code) {
            <div
              data-region="diagnostics-count-item"
              [attr.data-cat-code]="cat.code"
              class="inline-flex items-center gap-1.5"
            >
              <app-diagnostic-badge [flag]="cat.flag" />
              <span
                data-cell="count-value"
                class="tabular-nums font-medium text-text"
              >
                {{ cat.count }}
              </span>
            </div>
          }
        </div>
      }
      @if (showReconciliation()) {
        <div
          data-region="diagnostics-reconciliation"
          class="inline-flex items-center gap-2 text-flag-reconciliation"
        >
          <app-diagnostic-badge [flag]="reconciliationFlag()" />
          <span data-cell="reconciliation-label">{{ reconciliationLabel() }}</span>
        </div>
      }
    </section>
  `,
})
export class DiagnosticsStripComponent {
  readonly diagnostics = input.required<DriftDiagnostics>();

  readonly nonZeroCategories = computed<CategoryCount[]>(() => {
    const d = this.diagnostics();
    return CATEGORY_ORDER.flatMap((c) => this.buildCategory(d, c.code, c.field));
  });

  readonly showReconciliation = computed(
    () => this.diagnostics().reconciliation_ok === false,
  );

  readonly reconciliationFlag = computed<FlagInstance>(() => ({
    code: 'reconciliation_mismatch',
    reason: this.reconciliationLabel(),
    reference: null,
  }));

  readonly reconciliationLabel = computed(() => this.buildReconciliationLabel());

  private buildCategory(
    d: DriftDiagnostics,
    code: PositionFlag,
    field: keyof DriftDiagnostics,
  ): CategoryCount[] {
    const count = d[field] as number;
    if (count <= 0) return [];
    return [
      {
        code,
        count,
        flag: { code, reason: `${count} affected`, reference: null },
      },
    ];
  }

  private buildReconciliationLabel(): string {
    const d = this.diagnostics();
    const pctStr = this.formatPctSigned(d.reconciliation_delta_pct);
    const tolStr = this.formatPctUnsigned(d.tolerance_pct);
    return (
      `Reconciliation mismatch: sum ${this.formatEur(d.sum_eur)} vs invested ` +
      `${this.formatEur(d.invested_eur)} (Δ ${this.formatEur(d.delta_eur)} = ${pctStr}, tolerance ±${tolStr})`
    );
  }

  private formatEur(value: number): string {
    return `€${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  }

  private formatPctSigned(value: number | null): string {
    if (value === null) return 'n/a';
    return `${(value * 100).toFixed(2)}%`;
  }

  private formatPctUnsigned(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }
}
