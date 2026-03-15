import { Component, ChangeDetectionStrategy } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';

interface BottomTab {
  label: string;
  route: string;
  exact: boolean;
  icon: 'layout-dashboard' | 'briefcase' | 'pie-chart' | 'shield-alert' | 'cpu';
}

const BOTTOM_TABS: BottomTab[] = [
  { label: 'Dashboard', route: '/', exact: true, icon: 'layout-dashboard' },
  { label: 'Portfolio', route: '/portfolio-builder', exact: false, icon: 'briefcase' },
  { label: 'Optimize', route: '/optimization-studio', exact: false, icon: 'pie-chart' },
  { label: 'Risk', route: '/risk-center', exact: false, icon: 'shield-alert' },
  { label: 'AI', route: '/ai-control-room', exact: false, icon: 'cpu' },
];

@Component({
  selector: 'app-bottom-tab-bar',
  imports: [RouterLink, RouterLinkActive, LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <nav
      class="xl:hidden fixed bottom-0 left-0 right-0 z-50 bg-surface-raised border-t border-border pb-[env(safe-area-inset-bottom)]"
      aria-label="Mobile navigation">
      <div class="flex items-stretch">
        @for (tab of tabs; track tab.route) {
          <a
            [routerLink]="tab.route"
            routerLinkActive="text-accent"
            #rla="routerLinkActive"
            [routerLinkActiveOptions]="{ exact: tab.exact }"
            [attr.aria-current]="rla.isActive ? 'page' : null"
            [attr.aria-label]="tab.label"
            class="flex-1 flex flex-col items-center justify-center gap-1 min-h-[44px] py-2 text-text-secondary transition-colors hover:text-text">
            <i-lucide [name]="tab.icon" [size]="20" />
            <span class="text-[10px] font-medium leading-none">{{ tab.label }}</span>
          </a>
        }
      </div>
    </nav>
  `,
})
export class BottomTabBarComponent {
  readonly tabs = BOTTOM_TABS;
}
