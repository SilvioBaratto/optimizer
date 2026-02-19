import { Component, output, OnInit, DestroyRef, inject, ChangeDetectionStrategy } from '@angular/core';
import { ReactiveFormsModule, FormControl } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { debounceTime, distinctUntilChanged } from 'rxjs';

@Component({
  selector: 'app-search-input',
  imports: [ReactiveFormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="relative">
      <svg class="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
      </svg>
      <input [formControl]="searchControl" type="text" placeholder="Search..."
             class="w-full pl-8 pr-3 py-1.5 text-sm bg-surface-raised border border-border rounded-md text-text placeholder:text-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent" />
    </div>
  `,
})
export class SearchInputComponent implements OnInit {
  private readonly destroyRef = inject(DestroyRef);

  searchChange = output<string>();
  searchControl = new FormControl('', { nonNullable: true });

  ngOnInit() {
    this.searchControl.valueChanges.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(value => this.searchChange.emit(value));
  }
}
