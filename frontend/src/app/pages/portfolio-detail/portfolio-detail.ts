import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-portfolio-detail',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="p-6">
      <h1 class="text-lg font-semibold text-text">Portfolio detail</h1>
      <p class="mt-2 text-sm text-text-secondary">
        Portfolio: <span class="font-mono text-text">{{ name() }}</span>
      </p>
    </section>
  `,
})
export class PortfolioDetailComponent {
  readonly name = input.required<string>();
}
