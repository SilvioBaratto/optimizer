import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-page-header',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <header class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between mb-6">
      <div>
        <h1 class="font-heading text-heading-lg text-text">{{ title() }}</h1>
        @if (subtitle()) {
          <p class="text-data-sm text-text-secondary mt-0.5">{{ subtitle() }}</p>
        }
        @if (lastUpdated()) {
          <p class="text-data-xs text-text-tertiary mt-0.5">Updated {{ relativeTime() }}</p>
        }
      </div>
      <div class="flex items-center gap-2 mt-2 sm:mt-0">
        <ng-content select="[actions]" />
      </div>
    </header>
  `,
})
export class PageHeaderComponent {
  title = input.required<string>();
  subtitle = input<string>();
  lastUpdated = input<Date | null>();

  relativeTime = computed(() => {
    const date = this.lastUpdated();
    if (!date) return '';
    const now = Date.now();
    const diffMs = now - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);

    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    return `${diffDay}d ago`;
  });
}
