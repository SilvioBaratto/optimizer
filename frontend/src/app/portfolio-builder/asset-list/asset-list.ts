import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { Instrument } from '../../core/models/universe.model';

@Component({
  selector: 'app-asset-list',
  template: `
    <ul class="flex flex-col gap-1">
      @for (asset of assets(); track asset.id) {
        <li class="flex items-center justify-between rounded border p-2">
          <span>
            <span class="font-mono font-bold">{{ asset.ticker }}</span>
            @if (asset.short_name) {
              <span class="ml-2 text-gray-600 text-sm">{{ asset.short_name }}</span>
            }
          </span>
          <button
            type="button"
            class="text-sm text-red-500 hover:text-red-700"
            (click)="remove.emit(asset.id)"
          >
            Remove
          </button>
        </li>
      } @empty {
        <li class="text-gray-400 p-2 text-sm">No assets selected.</li>
      }
    </ul>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AssetListComponent {
  readonly assets = input<Instrument[]>([]);
  readonly remove = output<string>();
}
