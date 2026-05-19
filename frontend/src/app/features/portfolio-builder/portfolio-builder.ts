import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { BuilderStore } from './builder.store';

@Component({
  selector: 'app-portfolio-builder',
  template: `
    <div class="grid grid-rows-[auto_1fr_auto] h-full overflow-hidden">
      <div data-region="stage-strip"></div>
      <div
        class="grid lg:grid-cols-[320px_1fr_300px] max-lg:grid-cols-1 overflow-hidden"
      >
        <div data-region="left" class="overflow-y-auto"></div>
        <div data-region="center" class="overflow-y-auto"></div>
        <div data-region="right" class="overflow-y-auto"></div>
      </div>
      <div data-region="action-bar"></div>
    </div>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [BuilderStore],
})
export class PortfolioBuilderComponent {
  private readonly store = inject(BuilderStore);
}
