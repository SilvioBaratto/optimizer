import { Injectable, signal, computed } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class GlobalSearchService {
  private _isOpen = signal(false);
  readonly isOpen = computed(() => this._isOpen());

  open() {
    this._isOpen.set(true);
  }

  close() {
    this._isOpen.set(false);
  }

  toggle() {
    this._isOpen.update(v => !v);
  }
}
