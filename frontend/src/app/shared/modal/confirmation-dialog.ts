import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

/**
 * Reusable confirmation dialog primitive.
 *
 *   <app-confirmation-dialog
 *     [open]="pending()"
 *     title="Truncate table?"
 *     [message]="'All rows in ' + tableName + ' will be removed.'"
 *     confirmLabel="Truncate"
 *     destructive
 *     (confirmed)="onConfirm()"
 *     (cancelled)="onCancel()" />
 *
 * The parent owns the pending state; this component is a pure presentation
 * primitive with confirm / cancel outputs.
 */
@Component({
  selector: 'app-confirmation-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (open()) {
      <div class="fixed inset-0 z-[260] bg-black/40 flex items-center justify-center p-4"
        role="dialog" aria-modal="true" [attr.aria-label]="title()">
        <div class="bg-surface border border-border rounded-lg shadow-xl w-full max-w-sm p-5 space-y-4">
          <h3 class="text-sm font-semibold text-text">{{ title() }}</h3>
          <p class="text-xs text-text-secondary">{{ message() }}</p>
          <div class="flex justify-end gap-2">
            <button type="button" (click)="cancelled.emit()"
              data-testid="confirm-dialog-cancel"
              [disabled]="inflight()"
              class="px-3 py-1.5 text-xs font-medium rounded-md border border-border text-text-secondary hover:bg-surface-inset disabled:opacity-50 transition-colors">
              {{ cancelLabel() }}
            </button>
            <button type="button" (click)="confirmed.emit()"
              data-testid="confirm-dialog-confirm"
              [disabled]="inflight()"
              class="px-3 py-1.5 text-xs font-medium rounded-md text-white disabled:opacity-50 transition-colors"
              [class]="confirmClass()">
              {{ confirmLabel() }}
            </button>
          </div>
        </div>
      </div>
    }
  `,
})
export class ConfirmationDialogComponent {
  readonly open = input<boolean>(false);
  readonly title = input<string>('Confirm');
  readonly message = input<string>('');
  readonly confirmLabel = input<string>('Confirm');
  readonly cancelLabel = input<string>('Cancel');
  readonly destructive = input<boolean>(false);
  readonly inflight = input<boolean>(false);

  readonly confirmed = output<void>();
  readonly cancelled = output<void>();

  readonly confirmClass = computed(() =>
    this.destructive() ? 'bg-loss hover:bg-loss/90' : 'bg-accent hover:bg-accent/90',
  );
}
