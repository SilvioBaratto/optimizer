import { Injectable, signal, computed, Type } from '@angular/core';

export interface ModalConfig {
  title?: string;
  size: 'sm' | 'md' | 'lg';
  component: Type<unknown>;
  inputs?: Record<string, unknown>;
}

@Injectable({ providedIn: 'root' })
export class ModalService {
  private _config = signal<ModalConfig | null>(null);
  readonly config = computed(() => this._config());

  open<T>(config: {
    title?: string;
    size?: 'sm' | 'md' | 'lg';
    component: Type<T>;
    inputs?: Record<string, unknown>;
  }) {
    this._config.set({
      title: config.title,
      size: config.size ?? 'md',
      component: config.component,
      inputs: config.inputs,
    });
  }

  close() {
    this._config.set(null);
  }
}
