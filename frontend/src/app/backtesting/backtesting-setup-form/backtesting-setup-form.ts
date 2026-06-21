import {
  ChangeDetectionStrategy,
  Component,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import {
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';

import { PortfolioApiService } from '../../core/services/portfolio-api.service';
import type { PortfolioDto } from '../../core/models/portfolio-api.model';

export interface BacktestRunConfig {
  portfolio: string;
  period: string;
  benchmark: string;
}

export const PERIOD_PRESETS = [
  '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'Max',
] as const;

export const BENCHMARK_OPTIONS = [
  { label: 'SPY (S&P 500)', value: 'SPY' },
  { label: 'MSCI World (URTH)', value: 'URTH' },
  { label: '60/40 (VBINX)', value: 'VBINX' },
  { label: 'QQQ (Nasdaq 100)', value: 'QQQ' },
  { label: 'IWM (Russell 2000)', value: 'IWM' },
] as const;

@Component({
  selector: 'app-backtest-setup-form',
  imports: [ReactiveFormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './backtesting-setup-form.html',
})
export class BacktestingSetupFormComponent {
  private readonly api = inject(PortfolioApiService);

  readonly isRunning = input<boolean>(false);

  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly periodPresets = PERIOD_PRESETS;
  readonly benchmarkOptions = BENCHMARK_OPTIONS;

  readonly setupForm = new FormGroup({
    portfolio: new FormControl<string>('', {
      validators: [Validators.required],
      nonNullable: true,
    }),
    period: new FormControl<string>('1Y', { nonNullable: true }),
    benchmark: new FormControl<string>('SPY', { nonNullable: true }),
  });

  readonly runConfig = output<BacktestRunConfig>();

  constructor() {
    this.loadPortfolios();
  }

  private loadPortfolios(): void {
    this.api.list().subscribe({
      next: (res) => this.portfolios.set(res.items),
      error: () => {},
    });
  }

  onRun(): void {
    if (!this.setupForm.valid) return;
    const { portfolio, period, benchmark } = this.setupForm.getRawValue();
    this.runConfig.emit({ portfolio, period, benchmark });
  }
}
