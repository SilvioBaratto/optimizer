import {
  Component,
  signal,
  ChangeDetectionStrategy,
} from '@angular/core';
import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../shared/components/tab-group/tab-group';
import { DataManagementPanelComponent } from './data-management-panel/data-management-panel';
import { PortfoliosPanelComponent } from './portfolios-panel/portfolios-panel';
import { SchedulerStatusPanelComponent } from './scheduler-status-panel/scheduler-status-panel';
import { JobsPanelComponent } from './jobs-panel/jobs-panel';

type SettingsTab = 'data' | 'portfolios' | 'scheduler' | 'jobs';

@Component({
  selector: 'app-settings',
  imports: [
    PageHeaderComponent,
    TabGroupComponent,
    DataManagementPanelComponent,
    PortfoliosPanelComponent,
    SchedulerStatusPanelComponent,
    JobsPanelComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './settings.html',
})
export class SettingsComponent {
  readonly activeTab = signal<SettingsTab>('data');

  readonly tabs: Tab[] = [
    { id: 'data', label: 'Data Management' },
    { id: 'portfolios', label: 'Portfolios' },
    { id: 'scheduler', label: 'Scheduler' },
    { id: 'jobs', label: 'Jobs' },
  ];

  onTabChange(tab: string): void {
    this.activeTab.set(tab as SettingsTab);
  }
}
