import { Component, input, inject, ChangeDetectionStrategy } from '@angular/core';
import { ReactiveFormsModule, FormGroup, FormControl, Validators } from '@angular/forms';
import { ModalService } from '../../shared/modal/modal.service';
import { Constraint, ConstraintType } from '../../models/portfolio-builder.model';

interface ConstraintTypeOption {
  value: ConstraintType;
  label: string;
}

@Component({
  selector: 'app-constraint-modal',
  imports: [ReactiveFormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <form [formGroup]="form" (ngSubmit)="onSubmit()" class="space-y-4">
      <!-- Constraint Type -->
      <div>
        <label class="block text-data-xs font-medium text-text-secondary mb-1"
          >Constraint Type</label
        >
        <select
          formControlName="type"
          class="w-full px-3 py-2 text-data-sm bg-surface border border-border rounded-md text-text focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent transition-colors"
        >
          @for (opt of constraintTypes; track opt.value) {
            <option [value]="opt.value">{{ opt.label }}</option>
          }
        </select>
      </div>

      <!-- Label -->
      <div>
        <label class="block text-data-xs font-medium text-text-secondary mb-1">Label</label>
        <input
          type="text"
          formControlName="label"
          placeholder="e.g. Max single position weight"
          class="w-full px-3 py-2 text-data-sm bg-surface border border-border rounded-md text-text placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent transition-colors"
        />
        @if (form.get('label')?.invalid && form.get('label')?.touched) {
          <p class="mt-1 text-data-xs text-danger">Label is required.</p>
        }
      </div>

      <!-- Min / Max (for weight_bounds, sector_bounds, tracking_error) -->
      @if (showMinMax()) {
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-data-xs font-medium text-text-secondary mb-1">Min (0-1)</label>
            <input
              type="number"
              formControlName="min"
              step="0.01"
              min="0"
              max="1"
              placeholder="0.00"
              class="w-full px-3 py-2 text-data-sm bg-surface border border-border rounded-md text-text placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent transition-colors"
            />
          </div>
          <div>
            <label class="block text-data-xs font-medium text-text-secondary mb-1">Max (0-1)</label>
            <input
              type="number"
              formControlName="max"
              step="0.01"
              min="0"
              max="1"
              placeholder="0.10"
              class="w-full px-3 py-2 text-data-sm bg-surface border border-border rounded-md text-text placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent transition-colors"
            />
          </div>
        </div>
      }

      <!-- Single Value (for cardinality, turnover) -->
      @if (showSingleValue()) {
        <div>
          <label class="block text-data-xs font-medium text-text-secondary mb-1">
            @switch (form.get('type')?.value) {
              @case ('cardinality') {
                Min Positions
              }
              @case ('turnover') {
                Max Turnover (0-1)
              }
              @default {
                Value
              }
            }
          </label>
          <input
            type="number"
            formControlName="value"
            step="0.01"
            min="0"
            placeholder="0.00"
            class="w-full px-3 py-2 text-data-sm bg-surface border border-border rounded-md text-text placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent transition-colors"
          />
        </div>
      }

      <!-- Target (optional: for sector_bounds) -->
      @if (showTarget()) {
        <div>
          <label class="block text-data-xs font-medium text-text-secondary mb-1"
            >Target Sector (optional)</label
          >
          <input
            type="text"
            formControlName="target"
            placeholder="e.g. Technology (leave blank for all)"
            class="w-full px-3 py-2 text-data-sm bg-surface border border-border rounded-md text-text placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent focus:border-accent transition-colors"
          />
        </div>
      }

      <!-- Actions -->
      <div class="flex justify-end gap-2 pt-2 border-t border-border">
        <button
          type="button"
          (click)="onCancel()"
          class="px-4 py-2 text-data-sm font-medium rounded-lg border border-border text-text-secondary hover:bg-surface-inset transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          [disabled]="form.invalid"
          class="px-4 py-2 text-data-sm font-medium rounded-lg bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Add Constraint
        </button>
      </div>
    </form>
  `,
})
export class ConstraintModalComponent {
  /** Callback invoked when the user confirms adding a new constraint. Provided via ModalService inputs. */
  onConstraintAdded = input<(c: Constraint) => void>();

  private readonly modalService = inject(ModalService);

  readonly constraintTypes: ConstraintTypeOption[] = [
    { value: 'weight_bounds', label: 'Weight Bounds' },
    { value: 'sector_bounds', label: 'Sector Bounds' },
    { value: 'cardinality', label: 'Cardinality (Min Positions)' },
    { value: 'turnover', label: 'Turnover Limit' },
    { value: 'tracking_error', label: 'Tracking Error' },
  ];

  form = new FormGroup({
    type: new FormControl<ConstraintType>('weight_bounds', { nonNullable: true }),
    label: new FormControl('', {
      nonNullable: true,
      validators: [Validators.required, Validators.minLength(1)],
    }),
    min: new FormControl<number | null>(null),
    max: new FormControl<number | null>(null),
    value: new FormControl<number | null>(null),
    target: new FormControl<string>('', { nonNullable: true }),
  });

  showMinMax(): boolean {
    const t = this.form.get('type')?.value;
    return t === 'weight_bounds' || t === 'sector_bounds' || t === 'tracking_error';
  }

  showSingleValue(): boolean {
    const t = this.form.get('type')?.value;
    return t === 'cardinality' || t === 'turnover';
  }

  showTarget(): boolean {
    return this.form.get('type')?.value === 'sector_bounds';
  }

  onSubmit() {
    if (this.form.invalid) return;
    const raw = this.form.getRawValue();

    const constraint: Constraint = {
      id: `c${Date.now()}`,
      type: raw.type,
      label: raw.label,
      enabled: true,
      ...(raw.min != null ? { min: raw.min } : {}),
      ...(raw.max != null ? { max: raw.max } : {}),
      ...(raw.value != null ? { value: raw.value } : {}),
      ...(raw.target ? { target: raw.target } : {}),
    };

    this.onConstraintAdded()?.(constraint);
    this.modalService.close();
  }

  onCancel() {
    this.modalService.close();
  }
}
