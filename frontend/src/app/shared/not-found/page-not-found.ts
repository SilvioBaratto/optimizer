import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-page-not-found',
  imports: [RouterLink],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section
      class="flex flex-col items-center justify-center py-24 text-center"
    >
      <p class="text-6xl font-semibold text-text-tertiary tracking-tight">
        404
      </p>
      <h1 class="mt-4 text-lg font-medium text-text">Page not found</h1>
      <p class="mt-2 max-w-sm text-sm text-text-secondary">
        The page you are looking for does not exist or has been moved.
      </p>
      <a
        routerLink="/"
        class="mt-6 inline-flex items-center rounded-md border border-border bg-surface px-4 py-2 text-sm font-medium text-text hover:bg-surface-hover transition-colors"
      >
        Back to Dashboard
      </a>
    </section>
  `,
})
export class PageNotFoundComponent {}
