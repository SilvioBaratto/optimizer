import { Component, computed, input, output, inject, ChangeDetectionStrategy } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NAV_GROUPS, NavGroup } from './nav-data';
import { BreakpointService } from '../../services/breakpoint.service';

@Component({
  selector: 'app-sidebar',
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SidebarComponent {
  readonly isOpen = input<boolean>(false);
  readonly isMobile = input<boolean>(false);

  readonly closeSidebar = output<void>();

  readonly navGroups: NavGroup[] = NAV_GROUPS;

  private readonly breakpoint = inject(BreakpointService);

  // Sidebar is permanently visible on desktop (xl+).
  // On mobile and tablet it acts as a drawer, visible only when explicitly opened.
  readonly showSidebar = computed(
    () => (!this.breakpoint.isMobile() && !this.breakpoint.isTablet()) || this.isOpen(),
  );

  onNavClick() {
    this.closeSidebar.emit();
  }
}
