import { Component, input, output, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { PercentPipe } from '@angular/common';
import { ReactiveFormsModule, FormGroup, FormControl } from '@angular/forms';
import { StrategyType, PortfolioOptimizeRequest } from '../../models/portfolio.model';

@Component({
  selector: 'app-config-panel',
  imports: [ReactiveFormsModule, PercentPipe],
  templateUrl: './config-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConfigPanelComponent implements OnInit {
  strategy = input<StrategyType | null>(null);
  disabled = input(false);

  runOptimization = output<PortfolioOptimizeRequest>();

  form = new FormGroup({
    backtest: new FormControl(true, { nonNullable: true }),
    pre_selection: new FormControl(true, { nonNullable: true }),
    sector_tolerance: new FormControl(0.20, { nonNullable: true }),
    risk_aversion: new FormControl(1.0, { nonNullable: true }),
    cvar_beta: new FormControl(0.95, { nonNullable: true }),
  });

  ngOnInit() {}

  showRiskAversion(): boolean {
    return this.strategy() === 'max_utility';
  }

  showCvarBeta(): boolean {
    const s = this.strategy();
    return s === 'min_cvar' || s === 'cvar_parity';
  }

  onSubmit() {
    const s = this.strategy();
    if (!s) return;

    const val = this.form.getRawValue();
    const req: PortfolioOptimizeRequest = {
      strategy: s,
      backtest: val.backtest,
      pre_selection: val.pre_selection,
      sector_tolerance: val.sector_tolerance,
    };
    if (this.showRiskAversion()) req.risk_aversion = val.risk_aversion;
    if (this.showCvarBeta()) req.cvar_beta = val.cvar_beta;

    this.runOptimization.emit(req);
  }
}
