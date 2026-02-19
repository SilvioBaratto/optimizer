import { Component, input, output, signal, inject, DestroyRef, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { UniverseService } from '../../services/universe.service';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { SearchInputComponent } from '../../shared/search-input/search-input';
import { Instrument, Exchange } from '../../models/universe.model';

@Component({
  selector: 'app-universe-list',
  imports: [DataTableComponent, SearchInputComponent],
  templateUrl: './universe-list.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class UniverseListComponent implements OnInit {
  exchanges = input<Exchange[]>([]);
  instrumentSelect = output<Instrument>();

  private readonly universeService = inject(UniverseService);
  private readonly destroyRef = inject(DestroyRef);

  instruments = signal<Instrument[]>([]);
  total = signal(0);
  loading = signal(true);
  searchTerm = signal('');
  exchangeFilter = signal('');
  currentPage = signal(1);

  columns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'exchange', label: 'Exchange', sortable: true },
    { key: 'type', label: 'Type' },
    { key: 'currency', label: 'Currency' },
  ];

  ngOnInit() {
    this.loadInstruments();
  }

  onSearch(term: string) {
    this.searchTerm.set(term);
    this.currentPage.set(1);
    this.loadInstruments();
  }

  onExchangeFilter(event: Event) {
    this.exchangeFilter.set((event.target as HTMLSelectElement).value);
    this.currentPage.set(1);
    this.loadInstruments();
  }

  onPageChange(page: number) {
    this.currentPage.set(page);
    this.loadInstruments();
  }

  onRowClick(row: Record<string, unknown>) {
    this.instrumentSelect.emit(row as unknown as Instrument);
  }

  private loadInstruments() {
    this.loading.set(true);
    this.universeService.getInstruments({
      page: this.currentPage(),
      page_size: 15,
      search: this.searchTerm(),
      exchange: this.exchangeFilter(),
    }).pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(result => {
        this.instruments.set(result.items);
        this.total.set(result.total);
        this.loading.set(false);
      });
  }
}
