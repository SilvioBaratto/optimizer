import {
  Component,
  input,
  output,
  signal,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { IPS, RiskProfile } from '../../models/portfolio-builder.model';

// IPS preset catalogue — static defaults for the summary panel.
const IPS_PRESETS: readonly IPS[] = [
  {
    name: 'Conservative Income',
    riskProfile: 'conservative',
    targetReturn: 0.06,
    maxVolatility: 0.1,
    maxDrawdown: -0.08,
    rebalanceFrequency: 'quarterly',
    constraints: [],
  },
  {
    name: 'Balanced Growth',
    riskProfile: 'moderate',
    targetReturn: 0.1,
    maxVolatility: 0.16,
    maxDrawdown: -0.15,
    rebalanceFrequency: 'quarterly',
    constraints: [],
  },
  {
    name: 'Aggressive Growth',
    riskProfile: 'aggressive',
    targetReturn: 0.15,
    maxVolatility: 0.22,
    maxDrawdown: -0.25,
    rebalanceFrequency: 'monthly',
    constraints: [],
  },
];

type ObjectiveType =
  | 'max_sharpe'
  | 'min_variance'
  | 'risk_parity'
  | 'max_diversification'
  | 'equal_weight'
  | 'custom';

type TimeHorizon = '1Y' | '3Y' | '5Y' | '10Y';
type BenchmarkType = 'sp500' | 'msci_world' | 'russell2000' | 'none';

interface ObjectiveOption {
  value: ObjectiveType;
  label: string;
}

interface BenchmarkOption {
  value: BenchmarkType;
  label: string;
}

@Component({
  selector: 'app-ips-panel',
  imports: [FormsModule, StatCardComponent],
  templateUrl: './ips-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class IpsPanelComponent {
  ips = input.required<IPS>();
  tickerCount = input.required<number>();
  totalWeight = input.required<number>();
  ipsChange = output<IPS>();

  readonly presets = IPS_PRESETS;

  objective = signal<ObjectiveType>('max_sharpe');
  timeHorizon = signal<TimeHorizon>('5Y');
  benchmark = signal<BenchmarkType>('sp500');
  riskSlider = signal<number>(50);

  readonly objectiveOptions: ObjectiveOption[] = [
    { value: 'max_sharpe', label: 'Max Sharpe Ratio' },
    { value: 'min_variance', label: 'Min Variance' },
    { value: 'risk_parity', label: 'Risk Parity' },
    { value: 'max_diversification', label: 'Max Diversification' },
    { value: 'equal_weight', label: 'Equal Weight' },
    { value: 'custom', label: 'Custom' },
  ];

  readonly timeHorizons: TimeHorizon[] = ['1Y', '3Y', '5Y', '10Y'];

  readonly benchmarkOptions: BenchmarkOption[] = [
    { value: 'sp500', label: 'S&P 500' },
    { value: 'msci_world', label: 'MSCI World' },
    { value: 'russell2000', label: 'Russell 2000' },
    { value: 'none', label: 'None' },
  ];

  riskProfileLabel = computed<RiskProfile>(() => {
    const v = this.riskSlider();
    if (v <= 30) return 'conservative';
    if (v <= 65) return 'moderate';
    return 'aggressive';
  });

  riskProfileColorClass = computed(() => {
    const map: Record<RiskProfile, string> = {
      conservative: 'text-info bg-severity-info-bg border-info/30',
      moderate: 'text-warning bg-severity-medium-bg border-warning/30',
      aggressive: 'text-danger bg-severity-high-bg border-danger/30',
      custom: 'text-text-secondary bg-surface-inset border-border',
    };
    return map[this.riskProfileLabel()];
  });

  summaryAssetCount = computed(() => String(this.tickerCount()));

  summaryTotalWeight = computed(() =>
    `${(this.totalWeight() * 100).toFixed(2)}%`,
  );

  summaryTargetReturn = computed(() =>
    `${(this.ips().targetReturn * 100).toFixed(1)}%`,
  );

  summaryMaxVol = computed(() =>
    `${(this.ips().maxVolatility * 100).toFixed(1)}%`,
  );

  loadPreset(preset: IPS) {
    this.ipsChange.emit(preset);
    const profileToSlider: Record<RiskProfile, number> = {
      conservative: 15,
      moderate: 50,
      aggressive: 80,
      custom: 50,
    };
    this.riskSlider.set(profileToSlider[preset.riskProfile]);
  }

  onRebalanceChange(value: string) {
    const freq = value as IPS['rebalanceFrequency'];
    this.ipsChange.emit({ ...this.ips(), rebalanceFrequency: freq });
  }

  onTargetReturnChange(value: number) {
    this.ipsChange.emit({ ...this.ips(), targetReturn: value / 100 });
  }

  onMaxVolChange(value: number) {
    this.ipsChange.emit({ ...this.ips(), maxVolatility: value / 100 });
  }

  onMaxDrawdownChange(value: number) {
    this.ipsChange.emit({ ...this.ips(), maxDrawdown: -(value / 100) });
  }

  readonly Math = Math;

  onRiskSliderChange(value: number) {
    this.riskSlider.set(value);
    const profile = this.riskProfileLabel();
    this.ipsChange.emit({ ...this.ips(), riskProfile: profile });
  }
}
