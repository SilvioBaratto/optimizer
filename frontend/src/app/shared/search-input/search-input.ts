import { Component, output, OnInit, DestroyRef, inject, ChangeDetectionStrategy } from '@angular/core';
import { ReactiveFormsModule, FormControl } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { debounceTime, distinctUntilChanged } from 'rxjs';

@Component({
  selector: 'app-search-input',
  imports: [ReactiveFormsModule, LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="relative">
      <i-lucide name="search" [size]="14" class="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-tertiary" />
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
