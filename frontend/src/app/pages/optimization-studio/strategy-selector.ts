import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';
import { StrategyInfo, StrategyType } from '../../models/portfolio.model';

@Component({
  selector: 'app-strategy-selector',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="space-y-4">
      @for (category of ['convex', 'hierarchical', 'naive']; track category) {
        <div>
          <h4 class="text-xs font-medium text-text-tertiary uppercase tracking-wide mb-2">{{ category }}</h4>
          <div class="space-y-1">
            @for (s of strategiesByCategory(category); track s.type) {
              <button (click)="strategyChange.emit(s.type)"
                      class="w-full text-left px-3 py-2 rounded-md text-sm transition-colors border"
                      [class.border-accent]="s.type === selected()"
                      [class.bg-surface-inset]="s.type === selected()"
                      [class.text-text]="s.type === selected()"
                      [class.border-transparent]="s.type !== selected()"
                      [class.text-text-secondary]="s.type !== selected()"
                      [class.hover:bg-surface-inset]="s.type !== selected()">
                <span class="font-medium">{{ s.name }}</span>
                <p class="text-xs text-text-tertiary mt-0.5">{{ s.description }}</p>
              </button>
            }
          </div>
        </div>
      }
    </div>
  `,
})
export class StrategySelectorComponent {
  strategies = input<StrategyInfo[]>([]);
  selected = input<StrategyType | null>(null);
  strategyChange = output<StrategyType>();

  strategiesByCategory(category: string): StrategyInfo[] {
    return this.strategies().filter(s => s.category === category);
  }
}
