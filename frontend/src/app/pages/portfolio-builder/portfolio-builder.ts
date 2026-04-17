import {
  Component,
  signal,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { UniversePanelComponent } from './universe-panel';
import { WeightPanelComponent } from './weight-panel';
import { IpsPanelComponent } from './ips-panel';
import { UniverseTicker, Constraint, IPS } from '../../models/portfolio-builder.model';
import { LucideAngularModule } from 'lucide-angular';
import {
  MOCK_UNIVERSE_TICKERS,
  MOCK_CONSTRAINTS_MODERATE,
  MOCK_IPS_PRESETS,
} from '../../mocks/portfolio-builder-mocks';

@Component({
  selector: 'app-portfolio-builder',
  imports: [PageHeaderComponent, TabGroupComponent, UniversePanelComponent, WeightPanelComponent, IpsPanelComponent, LucideAngularModule],
  templateUrl: './portfolio-builder.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PortfolioBuilderComponent {
  readonly isLoading = signal(true);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  activeTab = signal<string>('universe');

  selectedTickers = signal<UniverseTicker[]>(MOCK_UNIVERSE_TICKERS);
  constraints = signal<Constraint[]>(MOCK_CONSTRAINTS_MODERATE);
  ips = signal<IPS>(MOCK_IPS_PRESETS[0]);

  readonly tabs: Tab[] = [
    { id: 'universe', label: 'Universe' },
    { id: 'weights', label: 'Weights & Constraints' },
    { id: 'ips', label: 'IPS & Summary' },
  ];

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.hasError.set(false);
    this.isLoading.set(false);
  }

  retry(): void {
    this.loadData();
  }

  selectedCount = computed(
    () => this.selectedTickers().filter((t) => t.selected).length,
  );

  totalWeight = computed(() =>
    this.selectedTickers()
      .filter((t) => t.selected)
      .reduce((sum, t) => sum + t.weight, 0),
  );

  onTickersChange(updated: UniverseTicker[]) {
    this.selectedTickers.set(updated);
  }

  onConstraintsChange(updated: Constraint[]) {
    this.constraints.set(updated);
  }

  onIpsChange(updated: IPS) {
    this.ips.set(updated);
  }
}
