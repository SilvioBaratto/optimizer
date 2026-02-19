import { Component, signal, inject, DestroyRef, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { DatePipe } from '@angular/common';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { UniverseService } from '../../services/universe.service';
import { UniverseListComponent } from './universe-list';
import { InstrumentDetailComponent } from './instrument-detail';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { Exchange, Instrument, UniverseStats } from '../../models/universe.model';

@Component({
  selector: 'app-universe',
  imports: [DatePipe, UniverseListComponent, InstrumentDetailComponent, StatCardComponent],
  templateUrl: './universe.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class UniverseComponent implements OnInit {
  private readonly universeService = inject(UniverseService);
  private readonly destroyRef = inject(DestroyRef);

  stats = signal<UniverseStats | null>(null);
  exchanges = signal<Exchange[]>([]);
  selectedInstrumentId = signal<string | null>(null);

  ngOnInit() {
    this.universeService.getStats()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(s => this.stats.set(s));

    this.universeService.getExchanges()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(e => this.exchanges.set(e));
  }

  onInstrumentSelect(instrument: Instrument) {
    this.selectedInstrumentId.set(instrument.id);
  }
}
