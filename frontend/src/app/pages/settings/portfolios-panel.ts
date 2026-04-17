import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { PortfolioApiService } from '../../services/portfolio-api.service';
import type {
  CreatePortfolioDto,
  PortfolioDto,
} from '../../models/portfolio-api.model';

interface CreateForm {
  name: string;
  description: string;
  currency: string;
  benchmark_ticker: string;
}

const EMPTY_FORM: CreateForm = {
  name: '',
  description: '',
  currency: 'USD',
  benchmark_ticker: 'SPY',
};

const CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'] as const;

@Component({
  selector: 'app-portfolios-panel',
  imports: [FormsModule, DataTableComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './portfolios-panel.html',
})
export class PortfoliosPanelComponent {
  private readonly api = inject(PortfolioApiService);
  private readonly destroyRef = inject(DestroyRef);

  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly loading = signal<boolean>(false);
  readonly loadError = signal<string | null>(null);

  readonly showCreate = signal<boolean>(false);
  readonly form = signal<CreateForm>(EMPTY_FORM);
  readonly creating = signal<boolean>(false);
  readonly createError = signal<string | null>(null);

  readonly currencies = CURRENCIES;

  readonly tableColumns: TableColumn[] = [
    { key: 'name', label: 'Name', sortable: true },
    { key: 'currency', label: 'Currency', sortable: true },
    { key: 'benchmark_ticker', label: 'Benchmark', sortable: true },
    { key: 'is_active', label: 'Active', sortable: true },
    { key: 'created_at', label: 'Created', sortable: true, align: 'right' },
  ];

  readonly tableRows = computed(() =>
    this.portfolios().map((p) => ({
      name: p.name,
      currency: p.currency,
      benchmark_ticker: p.benchmark_ticker,
      is_active: p.is_active ? 'yes' : 'no',
      created_at: p.created_at,
    }) as Record<string, unknown>),
  );

  readonly isFormValid = computed(() => {
    const f = this.form();
    return f.name.trim().length >= 2 && f.currency.length === 3;
  });

  constructor() {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.loadError.set(null);
    this.api
      .list()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => {
          this.portfolios.set(list.items);
          this.loading.set(false);
        },
        error: (err: Error) => {
          this.loadError.set(err.message ?? 'Failed to load portfolios');
          this.loading.set(false);
        },
      });
  }

  toggleCreate(): void {
    this.showCreate.update((v) => !v);
    this.form.set(EMPTY_FORM);
    this.createError.set(null);
  }

  updateField<K extends keyof CreateForm>(key: K, value: CreateForm[K]): void {
    this.form.update((f) => ({ ...f, [key]: value }));
  }

  submit(): void {
    if (!this.isFormValid() || this.creating()) return;
    this.creating.set(true);
    this.createError.set(null);
    this.api
      .create(this.buildPayload())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.onCreateSuccess(),
        error: (err: Error) => this.onCreateError(err.message ?? 'Create failed'),
      });
  }

  private buildPayload(): CreatePortfolioDto {
    const f = this.form();
    return {
      name: f.name.trim(),
      description: f.description.trim() || null,
      currency: f.currency,
      benchmark_ticker: f.benchmark_ticker.trim() || undefined,
    };
  }

  private onCreateSuccess(): void {
    this.creating.set(false);
    this.showCreate.set(false);
    this.form.set(EMPTY_FORM);
    this.refresh();
  }

  private onCreateError(message: string): void {
    this.creating.set(false);
    this.createError.set(message);
  }
}
