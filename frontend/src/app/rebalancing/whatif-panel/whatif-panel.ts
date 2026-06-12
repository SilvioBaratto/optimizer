import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { PageErrorBannerComponent } from '../../shared/components/page-error-banner/page-error-banner';
import type {
  PolicyType,
  RebalanceDecideApiResponse,
  RebalanceDecideRequest,
} from '../rebalancing.model';

interface WeightRow extends Record<string, unknown> {
  ticker: string;
  delta: string;
}

function parseWeights(raw: string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const pair of raw.split(',')) {
    const [key, value] = pair.split(':').map((s) => s.trim());
    const num = Number(value);
    if (key && Number.isFinite(num)) result[key.toUpperCase()] = num;
  }
  return result;
}

function weightsSumTolerant(weights: Record<string, number>): boolean {
  if (Object.keys(weights).length === 0) return false;
  const sum = Object.values(weights).reduce((a, b) => a + b, 0);
  return Math.abs(sum - 1) < 0.02;
}

@Component({
  selector: 'app-whatif-panel',
  imports: [FormsModule, StatCardComponent, DataTableComponent, PageErrorBannerComponent],
  templateUrl: './whatif-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WhatifPanelComponent {
  readonly decideResponse = input<RebalanceDecideApiResponse | null>(null);
  readonly error = input<string | null>(null);

  readonly runDecide = output<RebalanceDecideRequest>();

  readonly currentRaw = signal<string>('AAPL:0.5, MSFT:0.5');
  readonly targetRaw = signal<string>('AAPL:0.6, MSFT:0.4');
  readonly policyType = signal<PolicyType>('threshold');
  readonly thresholdValue = signal<number>(0.05);
  readonly frequencyDays = signal<number>(63);

  readonly currentWeights = computed(() => parseWeights(this.currentRaw()));
  readonly targetWeights = computed(() => parseWeights(this.targetRaw()));

  readonly isFormValid = computed(() =>
    weightsSumTolerant(this.currentWeights()) && weightsSumTolerant(this.targetWeights()),
  );

  readonly kpiShouldRebalance = computed(() => {
    const r = this.decideResponse();
    if (!r) return '—';
    return r.shouldRebalance ? 'YES' : 'NO';
  });

  readonly kpiTurnover = computed(() => {
    const r = this.decideResponse();
    return r ? `${(r.turnover * 100).toFixed(2)}%` : '—';
  });

  readonly kpiEstCost = computed(() => {
    const r = this.decideResponse();
    return r ? `${(r.estimatedCost * 100).toFixed(3)}%` : '—';
  });

  readonly tradeColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker' },
    { key: 'delta', label: 'Δ weight', align: 'right' },
  ];

  readonly tradeRows = computed<WeightRow[]>(() => {
    const r = this.decideResponse();
    if (!r) return [];
    return Object.entries(r.tradeWeights)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .map(([ticker, delta]) => ({
        ticker,
        delta: `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(2)}%`,
      }));
  });

  submit(): void {
    if (!this.isFormValid()) return;
    const body: RebalanceDecideRequest = {
      current_weights: this.currentWeights(),
      target_weights: this.targetWeights(),
      policy_type: this.policyType(),
      policy_config: this.buildPolicyConfig(),
    };
    if (this.policyType() === 'hybrid') {
      const today = new Date().toISOString().slice(0, 10);
      body.current_date = today;
      body.last_review_date = today;
    }
    this.runDecide.emit(body);
  }

  private buildPolicyConfig(): Record<string, unknown> {
    const type = this.policyType();
    if (type === 'calendar') return { frequency_days: this.frequencyDays() };
    if (type === 'threshold') return { threshold: this.thresholdValue(), kind: 'absolute' };
    return {
      frequency_days: this.frequencyDays(),
      threshold: this.thresholdValue(),
      kind: 'absolute',
    };
  }
}
