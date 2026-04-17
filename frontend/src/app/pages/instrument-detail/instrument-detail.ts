import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-instrument-detail',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="p-6">
      <h1 class="text-lg font-semibold text-text">Instrument detail</h1>
      <p class="mt-2 text-sm text-text-secondary">
        Instrument id: <span class="font-mono text-text">{{ id() }}</span>
      </p>
    </section>
  `,
})
export class InstrumentDetailComponent {
  readonly id = input.required<string>();
}
