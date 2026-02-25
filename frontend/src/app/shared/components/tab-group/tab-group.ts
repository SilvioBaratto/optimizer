import { Component, input, output, viewChildren, ElementRef, ChangeDetectionStrategy } from '@angular/core';

export interface Tab {
  id: string;
  label: string;
  badge?: number;
}

@Component({
  selector: 'app-tab-group',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="overflow-x-auto scrollbar-hide border-b border-border"
         role="tablist"
         (keydown)="onKeydown($event)">
      <div class="flex gap-0 min-w-max">
        @for (tab of tabs(); track tab.id) {
          <button
            #tabBtn
            role="tab"
            [attr.aria-selected]="tab.id === activeTab()"
            [attr.tabindex]="tab.id === activeTab() ? 0 : -1"
            class="px-3 py-2 text-data-sm whitespace-nowrap transition-colors border-b-2 -mb-px"
            [class]="tab.id === activeTab()
              ? 'border-accent text-text font-medium'
              : 'border-transparent text-text-secondary hover:text-text hover:border-border'"
            (click)="tabChange.emit(tab.id)">
            {{ tab.label }}
            @if (tab.badge != null) {
              <span class="ml-1.5 inline-flex items-center justify-center px-1.5 py-0.5 rounded-full text-label bg-surface-inset">
                {{ tab.badge }}
              </span>
            }
          </button>
        }
      </div>
    </div>
  `,
})
export class TabGroupComponent {
  tabs = input.required<Tab[]>();
  activeTab = input.required<string>();
  tabChange = output<string>();

  private readonly tabButtons = viewChildren<ElementRef<HTMLButtonElement>>('tabBtn');

  onKeydown(event: KeyboardEvent): void {
    const allTabs = this.tabs();
    const currentIndex = allTabs.findIndex(t => t.id === this.activeTab());
    let next = currentIndex;

    if (event.key === 'ArrowRight') {
      next = (currentIndex + 1) % allTabs.length;
    } else if (event.key === 'ArrowLeft') {
      next = (currentIndex - 1 + allTabs.length) % allTabs.length;
    } else if (event.key === 'Home') {
      next = 0;
    } else if (event.key === 'End') {
      next = allTabs.length - 1;
    } else {
      return;
    }

    event.preventDefault();
    const nextTab = allTabs[next];
    if (nextTab) {
      this.tabChange.emit(nextTab.id);
      const btns = this.tabButtons();
      btns[next]?.nativeElement.focus();
    }
  }
}
