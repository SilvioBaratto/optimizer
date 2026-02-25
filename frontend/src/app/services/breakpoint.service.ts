import { Injectable, signal, computed } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class BreakpointService {
  private readonly MOBILE_MAX = 768;
  private readonly TABLET_MAX = 1280;

  readonly isMobile = signal(false);
  readonly isTablet = signal(false);

  readonly chartHeight = computed<number>(() => {
    if (this.isMobile()) return 220;
    if (this.isTablet()) return 280;
    return 380;
  });

  private resizeObserver?: ResizeObserver;

  constructor() {
    this.measureAndUpdate();
    this.initResizeObserver();
  }

  private measureAndUpdate(): void {
    if (typeof window === 'undefined') return;
    const width = window.innerWidth;
    this.isMobile.set(width < this.MOBILE_MAX);
    this.isTablet.set(width >= this.MOBILE_MAX && width < this.TABLET_MAX);
  }

  private initResizeObserver(): void {
    if (typeof window === 'undefined' || !('ResizeObserver' in window)) return;

    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width;
        this.isMobile.set(width < this.MOBILE_MAX);
        this.isTablet.set(width >= this.MOBILE_MAX && width < this.TABLET_MAX);
      }
    });

    this.resizeObserver.observe(document.body);
  }
}
