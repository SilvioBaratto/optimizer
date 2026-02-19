import { Component, computed, input, output, ChangeDetectionStrategy } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

interface NavItem {
  name: string;
  route: string;
  icon: 'dashboard' | 'universe' | 'data' | 'optimize' | 'macro';
}

@Component({
  selector: 'app-sidebar',
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SidebarComponent {
  isOpen = input<boolean>(false);
  isMobile = input<boolean>(false);

  closeSidebar = output<void>();

  navItems: NavItem[] = [
    { name: 'Dashboard', route: '/', icon: 'dashboard' },
    { name: 'Universe', route: '/universe', icon: 'universe' },
    { name: 'Data', route: '/data', icon: 'data' },
    { name: 'Optimizer', route: '/optimize', icon: 'optimize' },
    { name: 'Macro', route: '/macro', icon: 'macro' },
  ];

  showSidebar = computed(() => !this.isMobile() || this.isOpen());

  onNavClick() {
    this.closeSidebar.emit();
  }
}
