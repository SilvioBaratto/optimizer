import { ChangeDetectionStrategy, Component, inject, output } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { filter, map, merge, of } from 'rxjs';

import { UniverseService } from '../../core/services/universe.service';
import { Instrument } from '../../core/models/universe.model';

@Component({
  selector: 'app-asset-search',
  imports: [ReactiveFormsModule],
  template: `
    <div class="flex flex-col gap-2">
      <input
        [formControl]="queryCtrl"
        type="search"
        placeholder="Search instruments…"
        class="w-full rounded border p-2"
        aria-label="Search instruments"
      />
      @if (results().length > 0) {
        <ul class="max-h-48 overflow-y-auto rounded border bg-white shadow-sm" role="listbox">
          @for (inst of results(); track inst.id) {
            <li role="option">
              <button
                type="button"
                class="w-full p-2 text-left hover:bg-gray-50 flex gap-2"
                (click)="add.emit(inst)"
              >
                <span class="font-mono font-bold text-sm">{{ inst.ticker }}</span>
                <span class="text-gray-600 text-sm">{{ inst.short_name }}</span>
              </button>
            </li>
          }
        </ul>
      }
    </div>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AssetSearchComponent {
  readonly add = output<Instrument>();
  readonly queryCtrl = new FormControl('', { nonNullable: true });

  private readonly universe = inject(UniverseService);

  readonly results = toSignal(
    merge(
      this.queryCtrl.valueChanges.pipe(
        filter((q) => q.length === 0),
        map(() => [] as Instrument[]),
      ),
      this.universe
        .searchTickers(this.queryCtrl.valueChanges.pipe(filter((q) => q.length > 0)))
        .pipe(map((list) => list.items)),
    ),
    { initialValue: [] as Instrument[] },
  );
}
