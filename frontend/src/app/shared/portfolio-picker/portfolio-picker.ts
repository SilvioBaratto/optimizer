import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
} from '@angular/core';

import { PortfolioApiService } from '../../services/portfolio-api.service';
import { PortfolioDto } from '../../models/portfolio-api.model';
import { PortfolioContextService } from '../../services/portfolio-context.service';

type State =
  | { kind: 'loading' }
  | { kind: 'error' }
  | { kind: 'ready'; portfolios: readonly PortfolioDto[] };

@Component({
  selector: 'app-portfolio-picker',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="px-3 py-2 border-b border-border bg-surface-raised">
      <label
        class="block text-[11px] font-medium uppercase tracking-wider text-text-tertiary mb-1"
      >
        Portfolio
      </label>
      @switch (state().kind) {
        @case ('loading') {
          <div
            class="h-8 w-full rounded-md bg-surface-inset animate-pulse"
            data-testid="portfolio-picker-skeleton"
            aria-label="Loading portfolios"
          ></div>
        }
        @case ('error') {
          <div class="flex items-center justify-between gap-2">
            <span class="text-xs text-text-secondary">
              Failed to load portfolios
            </span>
            <button
              type="button"
              (click)="load()"
              class="text-xs font-medium text-accent hover:underline"
            >
              Retry
            </button>
          </div>
        }
        @case ('ready') {
          @if (portfolios().length === 0) {
            <p class="text-xs text-text-secondary">No portfolios — create one</p>
          } @else {
            <select
              aria-label="Portfolio"
              [value]="currentId() ?? ''"
              (change)="onChange($event)"
              class="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-text focus:outline-none focus:ring-1 focus:ring-accent"
            >
              @for (p of portfolios(); track p.id) {
                <option [value]="p.id">{{ p.name }}</option>
              }
            </select>
          }
        }
      }
    </div>
  `,
})
export class PortfolioPickerComponent {
  private readonly api = inject(PortfolioApiService);
  private readonly ctx = inject(PortfolioContextService);

  readonly state = signal<State>({ kind: 'loading' });
  readonly currentId = this.ctx.currentPortfolioId;
  readonly portfolios = computed(() =>
    this.state().kind === 'ready'
      ? (this.state() as { portfolios: readonly PortfolioDto[] }).portfolios
      : [],
  );

  constructor() {
    this.load();
  }

  load(): void {
    this.state.set({ kind: 'loading' });
    this.api.list().subscribe({
      next: (res) => this.state.set({ kind: 'ready', portfolios: res.items }),
      error: () => this.state.set({ kind: 'error' }),
    });
  }

  onChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    this.ctx.setPortfolio(value || null);
  }
}
