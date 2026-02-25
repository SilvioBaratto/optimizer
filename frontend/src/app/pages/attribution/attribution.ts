import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../services/format.service';
import { MockFetchService } from '../../services/mock-fetch.service';
import { BrinsonPanelComponent } from './brinson-panel';
import { SaaTaaPanelComponent } from './saa-taa-panel';
import { FactorAttributionPanelComponent } from './factor-attribution-panel';
import { HoldingsAttributionPanelComponent } from './holdings-attribution-panel';
import {
  MOCK_BRINSON_ATTRIBUTION,
  MOCK_MULTI_LEVEL_ATTRIBUTION,
  MOCK_FACTOR_ATTRIBUTION,
  MOCK_HOLDINGS_ATTRIBUTION,
} from '../../mocks/attribution-mocks';

@Component({
  selector: 'app-attribution',
  imports: [
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    BrinsonPanelComponent,
    SaaTaaPanelComponent,
    FactorAttributionPanelComponent,
    HoldingsAttributionPanelComponent,
  ],
  templateUrl: './attribution.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AttributionComponent {
  private fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly mockFetch = inject(MockFetchService);

  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  activeTab = signal('brinson');

  readonly brinson = MOCK_BRINSON_ATTRIBUTION;
  readonly multiLevel = MOCK_MULTI_LEVEL_ATTRIBUTION;
  readonly factorAttribution = MOCK_FACTOR_ATTRIBUTION;
  readonly holdingsAttribution = MOCK_HOLDINGS_ATTRIBUTION;

  kpiTotalActive = computed(() => this.fmt.formatPercent(this.brinson.totalActive));
  kpiAllocation = computed(() => this.fmt.formatPercent(this.brinson.totalAllocation));
  kpiSelection = computed(() => this.fmt.formatPercent(this.brinson.totalSelection));
  kpiInteraction = computed(() => this.fmt.formatPercent(this.brinson.totalInteraction));

  readonly tabs: Tab[] = [
    { id: 'brinson', label: 'Brinson-Fachler' },
    { id: 'saa-taa', label: 'SAA / TAA' },
    { id: 'factor', label: 'Factor Attribution' },
    { id: 'holdings', label: 'Holdings' },
  ];

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({
        brinson: MOCK_BRINSON_ATTRIBUTION,
        multiLevel: MOCK_MULTI_LEVEL_ATTRIBUTION,
        factorAttribution: MOCK_FACTOR_ATTRIBUTION,
        holdingsAttribution: MOCK_HOLDINGS_ATTRIBUTION,
      })
      .then(() => {
        this.isLoading.set(false);
      })
      .catch((err: Error) => {
        this.hasError.set(true);
        this.errorMessage.set(err.message);
        this.isLoading.set(false);
      });
  }

  retry(): void {
    this.loadData();
  }

  openReportModal(): void {
    this.modalService.open({ component: ExportReportModalComponent, title: 'Export Report', size: 'lg' });
  }
}
