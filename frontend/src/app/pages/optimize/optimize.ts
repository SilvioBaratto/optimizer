import { Component, signal, inject, DestroyRef, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { PortfolioService } from '../../services/portfolio.service';
import { MOCK_PORTFOLIO_RESULT, MOCK_STRATEGIES } from '../../mocks/mock-data';
import { StrategySelectorComponent } from './strategy-selector';
import { ConfigPanelComponent } from './config-panel';
import { ResultsViewComponent } from './results-view';
import { ProgressBarComponent } from '../../shared/progress-bar/progress-bar';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state';
import { StrategyInfo, StrategyType, PortfolioOptimizeRequest, PortfolioResult, PortfolioJobProgress } from '../../models/portfolio.model';

@Component({
  selector: 'app-optimize',
  imports: [StrategySelectorComponent, ConfigPanelComponent, ResultsViewComponent, ProgressBarComponent, EmptyStateComponent],
  templateUrl: './optimize.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OptimizeComponent implements OnInit {
  private readonly portfolioService = inject(PortfolioService);
  private readonly destroyRef = inject(DestroyRef);

  strategies = signal<StrategyInfo[]>(MOCK_STRATEGIES);
  selectedStrategy = signal<StrategyType | null>('max_sharpe');
  jobActive = signal(false);
  jobProgress = signal<PortfolioJobProgress>({ phase: '', pct: 0, detail: '' });
  result = signal<PortfolioResult | null>(MOCK_PORTFOLIO_RESULT);

  ngOnInit() {
    this.portfolioService.getStrategies()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(s => this.strategies.set(s));
  }

  onStrategyChange(type: StrategyType) {
    this.selectedStrategy.set(type);
    this.result.set(null);
  }

  onRunOptimization(req: PortfolioOptimizeRequest) {
    this.result.set(null);
    this.jobActive.set(true);

    const phases: PortfolioJobProgress[] = [
      { phase: 'Pre-selection', pct: 10, detail: 'Filtering universe...' },
      { phase: 'Data prep', pct: 25, detail: 'Computing returns...' },
      { phase: 'Moment estimation', pct: 45, detail: 'Estimating covariance...' },
      { phase: 'Optimization', pct: 65, detail: 'Running ' + req.strategy + '...' },
      { phase: 'Backtest', pct: 85, detail: 'Walk-forward validation...' },
      { phase: 'Complete', pct: 100, detail: 'Finalizing results...' },
    ];

    let i = 0;
    const interval = setInterval(() => {
      if (i < phases.length) {
        this.jobProgress.set(phases[i]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 250);

    this.portfolioService.runOptimization(req)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(res => {
        clearInterval(interval);
        this.jobProgress.set({ phase: 'Complete', pct: 100, detail: 'Done' });
        this.jobActive.set(false);
        this.result.set(res);
      });
  }
}
