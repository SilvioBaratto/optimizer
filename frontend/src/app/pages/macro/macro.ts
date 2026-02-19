import { Component, signal, computed, inject, DestroyRef, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { forkJoin } from 'rxjs';
import { MacroService } from '../../services/macro.service';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsLineComponent } from '../../shared/echarts-line/echarts-line';
import { LoadingSkeletonComponent } from '../../shared/loading-skeleton/loading-skeleton';
import { EconomicIndicator, BondYield, CountryMacroSummary } from '../../models/macro.model';

@Component({
  selector: 'app-macro',
  imports: [StatCardComponent, DataTableComponent, EchartsLineComponent, LoadingSkeletonComponent],
  templateUrl: './macro.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MacroComponent implements OnInit {
  private readonly macroService = inject(MacroService);
  private readonly destroyRef = inject(DestroyRef);

  indicators = signal<EconomicIndicator[]>([]);
  bondYields = signal<BondYield[]>([]);
  summaries = signal<CountryMacroSummary[]>([]);
  selectedCountry = signal('US');
  loading = signal(true);

  countries = computed(() => {
    const set = new Set(this.indicators().map(i => i.country));
    return [...set].sort();
  });

  filteredIndicators = computed(() => {
    const country = this.selectedCountry();
    return this.indicators().filter(i => i.country === country);
  });

  selectedSummary = computed(() => this.summaries().find(s => s.country === this.selectedCountry()));

  private readonly maturityOrder = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'];

  yieldCurveLabels = computed<string[]>(() =>
    this.bondYields()
      .filter(y => y.country === 'US')
      .sort((a, b) => this.maturityOrder.indexOf(a.maturity) - this.maturityOrder.indexOf(b.maturity))
      .map(y => y.maturity)
  );

  yieldCurveValues = computed<number[]>(() =>
    this.bondYields()
      .filter(y => y.country === 'US')
      .sort((a, b) => this.maturityOrder.indexOf(a.maturity) - this.maturityOrder.indexOf(b.maturity))
      .map(y => y.yield_pct)
  );

  indicatorColumns: TableColumn[] = [
    { key: 'name', label: 'Indicator', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'value', label: 'Value', align: 'right', sortable: true },
    { key: 'previous', label: 'Previous', align: 'right' },
    { key: 'unit', label: 'Unit' },
    { key: 'frequency', label: 'Freq' },
  ];

  yieldColumns: TableColumn[] = [
    { key: 'maturity', label: 'Maturity', sortable: true },
    { key: 'yield_pct', label: 'Yield %', align: 'right', sortable: true, format: (v) => Number(v).toFixed(2) + '%' },
    { key: 'change_bps', label: 'Change (bps)', align: 'right', format: (v) => {
      const n = Number(v);
      return (n >= 0 ? '+' : '') + n;
    }},
  ];

  ngOnInit() {
    forkJoin({
      indicators: this.macroService.getEconomicIndicators(),
      yields: this.macroService.getBondYields(),
      summaries: this.macroService.getCountrySummaries(),
    }).pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(result => {
        this.indicators.set(result.indicators);
        this.bondYields.set(result.yields);
        this.summaries.set(result.summaries);
        this.loading.set(false);
      });
  }

  onCountryChange(event: Event) {
    this.selectedCountry.set((event.target as HTMLSelectElement).value);
  }
}
