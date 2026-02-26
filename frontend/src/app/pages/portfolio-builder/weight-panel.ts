import {
  Component,
  input,
  output,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ModalService } from '../../shared/modal/modal.service';
import { Constraint, UniverseTicker } from '../../models/portfolio-builder.model';
import { ConstraintModalComponent } from './constraint-modal';

interface SectorBar {
  sector: string;
  weight: number;
  color: string;
}

const SECTOR_COLORS: Record<string, string> = {
  Technology: 'bg-chart-1',
  'Financial Services': 'bg-chart-2',
  Healthcare: 'bg-chart-3',
  'Consumer Defensive': 'bg-chart-4',
  'Consumer Cyclical': 'bg-chart-5',
  Industrials: 'bg-chart-6',
  Energy: 'bg-chart-8',
  Utilities: 'bg-chart-7',
  Communication: 'bg-accent',
  'Real Estate': 'bg-info',
};

@Component({
  selector: 'app-weight-panel',
  imports: [FormsModule],
  templateUrl: './weight-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WeightPanelComponent {
  tickers = input.required<UniverseTicker[]>();
  constraints = input.required<Constraint[]>();
  constraintsChange = output<Constraint[]>();

  private readonly modalService = inject(ModalService);

  weights = signal<Record<string, number>>({});

  selectedTickers = computed(() => this.tickers().filter((t) => t.selected));

  effectiveWeights = computed(() => {
    const overrides = this.weights();
    return this.selectedTickers().map((t) => ({
      ticker: t.ticker,
      name: t.name,
      sector: t.sector,
      weight: Object.hasOwn(overrides, t.ticker) ? (overrides[t.ticker] ?? 0) : t.weight,
    }));
  });

  totalWeight = computed(() =>
    this.effectiveWeights().reduce((sum, t) => sum + t.weight, 0),
  );

  totalWeightValid = computed(() => {
    const total = this.totalWeight();
    return total >= 0.99 && total <= 1.01;
  });

  sectorBars = computed((): SectorBar[] => {
    const sectorMap = new Map<string, number>();
    for (const t of this.effectiveWeights()) {
      sectorMap.set(t.sector, (sectorMap.get(t.sector) ?? 0) + t.weight);
    }
    return Array.from(sectorMap.entries())
      .map(([sector, weight]) => ({
        sector,
        weight,
        color: SECTOR_COLORS[sector] ?? 'bg-chart-7',
      }))
      .sort((a, b) => b.weight - a.weight);
  });

  getWeight(ticker: string): number {
    const overrides = this.weights();
    const ticker_data = this.tickers().find((t) => t.ticker === ticker);
    return Object.hasOwn(overrides, ticker)
      ? (overrides[ticker] ?? 0)
      : (ticker_data?.weight ?? 0);
  }

  parseWeight(raw: string): number {
    const cleaned = raw.replace(',', '.').replace(/[^0-9.]/g, '');
    const parsed = parseFloat(cleaned);
    return isNaN(parsed) ? 0 : parsed / 100;
  }

  setWeight(ticker: string, value: number) {
    this.weights.update((w) => ({ ...w, [ticker]: value }));
  }

  distributeEqual() {
    const selected = this.selectedTickers();
    if (selected.length === 0) return;
    const equalWeight = Math.round((1 / selected.length) * 10000) / 10000;
    const newWeights: Record<string, number> = {};
    for (const t of selected) {
      newWeights[t.ticker] = equalWeight;
    }
    this.weights.set(newWeights);
  }

  openConstraintModal() {
    this.modalService.open({
      title: 'Add Constraint',
      size: 'md',
      component: ConstraintModalComponent,
      inputs: {
        onConstraintAdded: (constraint: Constraint) => {
          this.constraintsChange.emit([...this.constraints(), constraint]);
        },
      },
    });
  }

  toggleConstraint(id: string) {
    const updated = this.constraints().map((c) =>
      c.id === id ? { ...c, enabled: !c.enabled } : c,
    );
    this.constraintsChange.emit(updated);
  }

  removeConstraint(id: string) {
    const updated = this.constraints().filter((c) => c.id !== id);
    this.constraintsChange.emit(updated);
  }

  constraintTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      weight_bounds: 'Weight Bounds',
      sector_bounds: 'Sector Bounds',
      cardinality: 'Cardinality',
      turnover: 'Turnover',
      tracking_error: 'Tracking Error',
    };
    return labels[type] ?? type;
  }

  constraintSummary(c: Constraint): string {
    const parts: string[] = [];
    if (c.min != null) parts.push(`min ${(c.min * 100).toFixed(1)}%`);
    if (c.max != null) parts.push(`max ${(c.max * 100).toFixed(1)}%`);
    if (c.value != null) parts.push(`${c.value}`);
    if (c.target) parts.push(`target: ${c.target}`);
    return parts.join(', ');
  }

  formatPct(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }
}
