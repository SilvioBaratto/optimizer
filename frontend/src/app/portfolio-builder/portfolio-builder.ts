import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
} from '@angular/core';

import { Instrument } from '../core/models/universe.model';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AssetListComponent } from './asset-list/asset-list';
import { AssetSearchComponent } from './asset-search/asset-search';
import { DiversificationPreviewComponent } from './diversification-preview/diversification-preview';

@Component({
  selector: 'app-portfolio-builder',
  imports: [AssetSearchComponent, AssetListComponent, DiversificationPreviewComponent],
  template: `
    <div class="flex flex-col h-full overflow-hidden gap-6 p-6">
      <header>
        <h1 class="text-2xl font-semibold">Portfolio Builder</h1>
        <p class="text-gray-500 text-sm mt-1">
          Search for instruments and build your asset list.
        </p>
      </header>

      <section data-region="search">
        <app-asset-search (add)="addAsset($event)" />
      </section>

      <section data-region="list" class="overflow-y-auto">
        <h2 class="text-sm font-medium text-gray-700 mb-2">
          Selected assets ({{ selectedAssets().length }})
        </h2>
        <app-asset-list
          [assets]="selectedAssets()"
          (remove)="removeAsset($event)"
        />
      </section>

      <section data-region="preview" class="flex-1 overflow-y-auto">
        <h2 class="text-sm font-medium text-gray-700 mb-2">Diversification preview</h2>
        <app-diversification-preview [tickers]="selectedTickers()" />
      </section>

      <section data-region="save" class="flex flex-col gap-2 pt-4 border-t border-gray-200">
        <h2 class="text-sm font-medium text-gray-700">Save portfolio</h2>
        <div class="flex gap-2 items-start">
          <div class="flex-1">
            <input
              type="text"
              [value]="portfolioName()"
              (input)="portfolioName.set($any($event.target).value)"
              placeholder="Portfolio name"
              class="w-full border border-gray-300 rounded px-3 py-2 text-sm"
              aria-label="Portfolio name"
            />
            @if (nameError()) {
              <p class="text-red-500 text-xs mt-1" role="alert">{{ nameError() }}</p>
            }
          </div>
          <button
            type="button"
            (click)="savePortfolio(portfolioName())"
            class="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
          >
            Save
          </button>
        </div>
        @if (saveSuccess()) {
          <p class="text-green-600 text-sm">Portfolio saved.</p>
        }
        @if (saveError()) {
          <p class="text-red-500 text-sm" role="alert">{{ saveError() }}</p>
        }
      </section>
    </div>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PortfolioBuilderComponent {
  private readonly api = inject(PortfolioApiService);

  readonly selectedAssets = signal<Instrument[]>([]);
  readonly selectedTickers = computed(() => this.selectedAssets().map(a => a.id));
  readonly portfolioName = signal('');
  readonly saveSuccess = signal(false);
  readonly saveError = signal<string | null>(null);
  readonly nameError = signal<string | null>(null);
  readonly searchQuery = signal('');
  readonly searchResults = computed(() => [] as Instrument[]);
  readonly diversificationPreview = computed(() =>
    this.selectedAssets().length > 0 ? { count: this.selectedAssets().length } : null,
  );

  addAsset(instrument: Instrument): void {
    if (this.selectedAssets().some((a) => a.id === instrument.id)) return;
    this.selectedAssets.update((list) => [...list, instrument]);
  }

  addInstrument(instrument: Instrument): void {
    this.addAsset(instrument);
  }

  removeAsset(id: string): void {
    this.selectedAssets.update((list) => list.filter((a) => a.id !== id));
  }

  savePortfolio(name: string): void {
    if (!name.trim()) {
      this.nameError.set('Portfolio name is required');
      return;
    }
    this.nameError.set(null);
    this.saveSuccess.set(false);
    this.saveError.set(null);
    this.api.create({ name: name.trim() }).subscribe({
      next: () => this.saveSuccess.set(true),
      error: (e: unknown) => this.saveError.set(this.toSaveErrMsg(e)),
    });
  }

  private toSaveErrMsg(e: unknown): string {
    if ((e as { status?: number }).status === 409) return 'duplicate — this name already exists';
    return e instanceof Error ? e.message : 'Failed to save portfolio';
  }
}
