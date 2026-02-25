import { Component, input, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { EchartsBarComponent, BarData } from '../../shared/echarts-bar/echarts-bar';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../services/format.service';
import type { TradePreview, TradeSummary, DriftEntry } from '../../models/rebalancing.model';

const THRESHOLD_LEVELS_BPS = [100, 200, 300, 400, 500] as const;

@Component({
  selector: 'app-whatif-panel',
  imports: [EchartsBarComponent, StatCardComponent],
  templateUrl: './whatif-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WhatifPanelComponent {
  trades      = input<TradePreview[]>([]);
  summary     = input.required<TradeSummary>();
  driftEntries = input<DriftEntry[]>([]);

  private fmt = inject(FormatService);

  readonly partialRebalPct   = signal(100);
  readonly thresholdOverride = signal(250);
  readonly costMultiplier    = signal(100);

  readonly adjustedTurnover = computed(() => {
    const raw = this.summary().totalTurnover * (this.partialRebalPct() / 100);
    return this.fmt.formatPercent(raw);
  });

  readonly adjustedCost = computed(() => {
    const raw = this.summary().totalCost
      * (this.partialRebalPct() / 100)
      * (this.costMultiplier() / 100);
    return this.fmt.formatCurrency(raw);
  });

  readonly remainingDrift = computed(() => {
    const entries = this.driftEntries();
    const maxDrift = entries.length > 0
      ? Math.max(...entries.map(e => e.driftAbsolute))
      : 0;
    const remaining = maxDrift * (1 - this.partialRebalPct() / 100);
    return this.fmt.formatBps(remaining);
  });

  readonly tradesAffected = computed(() =>
    String(Math.round(this.summary().totalTrades * this.partialRebalPct() / 100)),
  );

  readonly thresholdSensitivityData = computed((): BarData[] => {
    const entries = this.driftEntries();
    const total = entries.length || 1;

    return THRESHOLD_LEVELS_BPS.map(bps => {
      const threshold = bps / 10000;
      const count = entries.filter(e => e.driftAbsolute > threshold).length;
      return {
        label: `${bps} bps`,
        value: count / total,
      };
    });
  });

  onPartialRebalChange(event: Event) {
    this.partialRebalPct.set(+(event.target as HTMLInputElement).value);
  }

  onThresholdOverrideChange(event: Event) {
    this.thresholdOverride.set(+(event.target as HTMLInputElement).value);
  }

  onCostMultiplierChange(event: Event) {
    this.costMultiplier.set(+(event.target as HTMLInputElement).value);
  }
}
