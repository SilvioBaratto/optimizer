import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ActionBarComponent } from './action-bar/action-bar';
import { AnalyticsPaneComponent } from './analytics-pane/analytics-pane';
import { BuilderResultService } from './builder-result.service';
import { BuilderStore } from './builder.store';
import { CanvasPaneComponent } from './canvas-pane/canvas-pane';
import { InputsPaneComponent } from './inputs-pane/inputs-pane';
import { StageStripComponent } from './stage-strip/stage-strip';

@Component({
  selector: 'app-portfolio-builder',
  imports: [
    StageStripComponent,
    InputsPaneComponent,
    CanvasPaneComponent,
    AnalyticsPaneComponent,
    ActionBarComponent,
  ],
  template: `
    <div class="grid grid-rows-[auto_1fr_auto] h-full overflow-hidden">
      <div data-region="stage-strip">
        <app-stage-strip
          [currentStage]="store.currentStage()"
          [stepResults]="store.stepResults()"
          (stageSelect)="store.setStage($event)"
        />
      </div>
      <div
        class="grid lg:grid-cols-[320px_1fr_300px] max-lg:grid-cols-1 overflow-hidden"
      >
        <div data-region="left" class="overflow-y-auto">
          <app-inputs-pane />
        </div>
        <div data-region="center" class="overflow-y-auto">
          <app-canvas-pane />
        </div>
        <div data-region="right" class="overflow-y-auto">
          <app-analytics-pane />
        </div>
      </div>
      <div data-region="action-bar">
        <app-action-bar />
      </div>
    </div>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [BuilderStore, BuilderResultService],
})
export class PortfolioBuilderComponent {
  readonly store = inject(BuilderStore);
  private readonly resultService = inject(BuilderResultService);

  constructor() {
    this.resultService.init();
  }
}
