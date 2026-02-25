import {
  Component,
  inject,
  effect,
  viewChild,
  ViewContainerRef,
  ComponentRef,
  ElementRef,
  ChangeDetectionStrategy,
  AfterViewChecked,
} from '@angular/core';
import { ModalService } from './modal.service';

const FOCUSABLE_SELECTOR = 'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

@Component({
  selector: 'app-modal-container',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    '(document:keydown.escape)': 'modalService.close()',
    '(document:keydown)': 'onKeydown($event)',
  },
  template: `
    @if (modalService.config(); as cfg) {
      <!-- Backdrop -->
      <div class="fixed inset-0 z-[300] bg-black/40" (click)="modalService.close()"></div>
      <!-- Dialog -->
      <div class="fixed inset-0 z-[310] flex items-center justify-center p-4 pointer-events-none">
        <div #dialog
             role="dialog"
             aria-modal="true"
             [attr.aria-labelledby]="cfg.title ? 'modal-title' : null"
             class="pointer-events-auto w-full bg-surface-raised border border-border rounded-xl shadow-[var(--shadow-modal,0_20px_60px_rgba(0,0,0,0.3))]"
             [class]="sizeClass(cfg.size)"
             (click)="$event.stopPropagation()">
          @if (cfg.title) {
            <div class="flex items-center justify-between px-6 py-4 border-b border-border">
              <h2 id="modal-title" class="text-base font-semibold text-text">{{ cfg.title }}</h2>
              <button (click)="modalService.close()"
                      class="p-1 text-text-tertiary hover:text-text rounded transition-colors"
                      aria-label="Close">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          }
          <div class="p-6">
            <ng-container #outlet />
          </div>
        </div>
      </div>
    }
  `,
})
export class ModalContainerComponent implements AfterViewChecked {
  modalService = inject(ModalService);

  private outletRef = viewChild('outlet', { read: ViewContainerRef });
  private dialogRef = viewChild<ElementRef<HTMLElement>>('dialog');
  private componentRef: ComponentRef<unknown> | null = null;
  private previouslyFocused: Element | null = null;
  private needsFocus = false;

  constructor() {
    effect(() => {
      const cfg = this.modalService.config();
      const outlet = this.outletRef();

      this.destroyComponent();

      if (cfg && outlet) {
        this.previouslyFocused = document.activeElement;
        this.componentRef = outlet.createComponent(cfg.component);
        if (cfg.inputs) {
          for (const [key, value] of Object.entries(cfg.inputs)) {
            this.componentRef.setInput(key, value);
          }
        }
        this.needsFocus = true;
      } else {
        this.restoreFocus();
      }
    });
  }

  ngAfterViewChecked() {
    if (this.needsFocus) {
      this.needsFocus = false;
      const dialog = this.dialogRef()?.nativeElement;
      if (dialog) {
        const first = dialog.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
        (first ?? dialog).focus();
      }
    }
  }

  onKeydown(event: KeyboardEvent) {
    if (event.key !== 'Tab' || !this.modalService.config()) return;

    const dialog = this.dialogRef()?.nativeElement;
    if (!dialog) return;

    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  sizeClass(size: 'sm' | 'md' | 'lg'): string {
    const map: Record<string, string> = {
      sm: 'max-w-sm',
      md: 'max-w-lg',
      lg: 'max-w-2xl',
    };
    return map[size];
  }

  private restoreFocus() {
    if (this.previouslyFocused && this.previouslyFocused instanceof HTMLElement) {
      this.previouslyFocused.focus();
    }
    this.previouslyFocused = null;
  }

  private destroyComponent() {
    if (this.componentRef) {
      this.componentRef.destroy();
      this.componentRef = null;
    }
  }
}
