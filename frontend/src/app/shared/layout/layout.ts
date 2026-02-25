import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { Router, RouterOutlet } from '@angular/router';
import { trigger, transition, style, animate } from '@angular/animations';
import { SidebarComponent } from '../sidebar/sidebar';
import { ContextBarComponent } from '../components/context-bar/context-bar';
import { ToastContainerComponent } from '../notification/toast-container';
import { ModalContainerComponent } from '../modal/modal-container';
import { GlobalSearchComponent } from '../global-search/global-search';
import { GlobalSearchService } from '../global-search/global-search.service';
import { BottomTabBarComponent } from '../bottom-tab-bar/bottom-tab-bar';
import { BreakpointService } from '../../services/breakpoint.service';

const routerTransition = trigger('routerTransition', [
  transition('* <=> *', [
    style({ opacity: 0, transform: 'translateY(4px)' }),
    animate('150ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
  ]),
]);

const PAGE_ROUTES = [
  '/',
  '/portfolio-builder',
  '/optimization-studio',
  '/backtesting',
  '/risk-center',
  '/factor-research',
  '/rebalancing',
  '/attribution',
  '/ai-control-room',
  '/settings',
];

@Component({
  selector: 'app-layout',
  imports: [
    RouterOutlet,
    SidebarComponent,
    ContextBarComponent,
    ToastContainerComponent,
    ModalContainerComponent,
    GlobalSearchComponent,
    BottomTabBarComponent,
  ],
  templateUrl: './layout.html',
  styleUrl: './layout.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
  animations: [routerTransition],
  host: {
    '(document:keydown)': 'onGlobalKeydown($event)',
  },
})
export class LayoutComponent {
  readonly searchService = inject(GlobalSearchService);
  readonly breakpoint = inject(BreakpointService);
  private readonly router = inject(Router);

  readonly isSidebarOpen = signal(false);

  readonly isMobile = computed(() => this.breakpoint.isMobile());

  readonly showOverlay = computed(
    () => this.isSidebarOpen() && (this.breakpoint.isMobile() || this.breakpoint.isTablet()),
  );

  toggleSidebar() {
    this.isSidebarOpen.update((v) => !v);
  }

  closeSidebar() {
    this.isSidebarOpen.set(false);
  }

  onGlobalKeydown(event: KeyboardEvent) {
    // Cmd/Ctrl+K — global search
    if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
      event.preventDefault();
      this.searchService.toggle();
      return;
    }

    // Cmd/Ctrl+1-9 — page navigation
    if ((event.metaKey || event.ctrlKey) && event.key >= '1' && event.key <= '9') {
      const index = parseInt(event.key, 10) - 1;
      if (index < PAGE_ROUTES.length) {
        event.preventDefault();
        this.router.navigateByUrl(PAGE_ROUTES[index]);
      }
    }
  }
}
