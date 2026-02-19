import { Component, input, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-loading-skeleton',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="animate-pulse bg-surface-inset rounded"
         [style.height]="height()"
         [style.width]="width()">
    </div>
  `,
})
export class LoadingSkeletonComponent {
  height = input('1rem');
  width = input('100%');
}
