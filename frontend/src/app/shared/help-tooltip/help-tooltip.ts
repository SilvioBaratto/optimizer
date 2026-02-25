import {
  Component,
  input,
  signal,
  computed,
  ElementRef,
  ChangeDetectionStrategy,
  inject,
} from '@angular/core';

@Component({
  selector: 'app-help-tooltip',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    class: 'relative inline-flex',
    '(mouseenter)': 'show()',
    '(mouseleave)': 'hide()',
    '(focusin)': 'show()',
    '(focusout)': 'hide()',
  },
  template: `
    <button type="button"
            class="inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-medium text-text-tertiary border border-border-muted hover:text-text-secondary hover:border-border transition-colors"
            aria-label="Help">
      ?
    </button>
    @if (isVisible()) {
      <div class="fixed z-[310] w-64 px-3 py-2 text-xs leading-relaxed text-text bg-surface-raised border border-border rounded-lg shadow-lg"
           [style.top.px]="tooltipPosition().top"
           [style.left.px]="tooltipPosition().left"
           role="tooltip">
        <p>{{ text() }}</p>
        @if (link()) {
          <a [href]="link()" target="_blank" rel="noopener noreferrer"
             class="mt-1 inline-block text-accent hover:underline">
            Learn more &rarr;
          </a>
        }
      </div>
    }
  `,
})
export class HelpTooltipComponent {
  text = input.required<string>();
  link = input<string>('');

  isVisible = signal(false);

  private el = inject(ElementRef);
  private rect = signal<DOMRect | null>(null);

  tooltipPosition = computed(() => {
    const r = this.rect();
    if (!r) return { top: 0, left: 0 };

    const tooltipWidth = 256;
    const tooltipHeight = 80;
    const gap = 8;

    let top = r.bottom + gap;
    let left = r.left + r.width / 2 - tooltipWidth / 2;

    if (typeof window !== 'undefined') {
      if (top + tooltipHeight > window.innerHeight) {
        top = r.top - tooltipHeight - gap;
      }
      if (left < 8) left = 8;
      if (left + tooltipWidth > window.innerWidth - 8) {
        left = window.innerWidth - tooltipWidth - 8;
      }
    }

    return { top, left };
  });

  show() {
    const el = this.el.nativeElement as HTMLElement;
    this.rect.set(el.getBoundingClientRect());
    this.isVisible.set(true);
  }

  hide() {
    this.isVisible.set(false);
  }
}
