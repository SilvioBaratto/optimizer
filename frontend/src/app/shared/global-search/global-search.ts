import {
  Component,
  inject,
  signal,
  computed,
  ElementRef,
  viewChild,
  afterNextRender,
  ChangeDetectionStrategy,
} from '@angular/core';
import { Router } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';
import { GlobalSearchService } from './global-search.service';
import { NAV_GROUPS } from '../sidebar/nav-data';

interface SearchResult {
  label: string;
  route: string;
  category: string;
}

@Component({
  selector: 'app-global-search',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './global-search.html',
  host: {
    '(keydown)': 'onKeydown($event)',
  },
})
export class GlobalSearchComponent {
  private router = inject(Router);
  private searchService = inject(GlobalSearchService);
  private searchInput = viewChild<ElementRef<HTMLInputElement>>('searchInput');

  query = signal('');
  activeIndex = signal(0);

  private allResults: SearchResult[] = NAV_GROUPS.flatMap(group =>
    group.items.map(item => ({
      label: item.name,
      route: item.route,
      category: group.label,
    })),
  );

  filteredResults = computed(() => {
    const q = this.query().toLowerCase().trim();
    if (!q) return this.allResults;
    return this.allResults.filter(
      r => r.label.toLowerCase().includes(q) || r.category.toLowerCase().includes(q),
    );
  });

  groupedResults = computed(() => {
    const results = this.filteredResults();
    const groups = new Map<string, SearchResult[]>();
    for (const r of results) {
      const list = groups.get(r.category) ?? [];
      list.push(r);
      groups.set(r.category, list);
    }
    return Array.from(groups.entries()).map(([category, items]) => ({ category, items }));
  });

  constructor() {
    afterNextRender(() => {
      this.searchInput()?.nativeElement.focus();
    });
  }

  onQueryChange(value: string) {
    this.query.set(value);
    this.activeIndex.set(0);
  }

  onKeydown(event: KeyboardEvent) {
    const total = this.filteredResults().length;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        this.activeIndex.update(i => (i + 1) % Math.max(1, total));
        break;
      case 'ArrowUp':
        event.preventDefault();
        this.activeIndex.update(i => (i - 1 + total) % Math.max(1, total));
        break;
      case 'Enter':
        event.preventDefault();
        this.selectCurrent();
        break;
      case 'Escape':
        event.preventDefault();
        this.searchService.close();
        break;
    }
  }

  selectResult(result: SearchResult) {
    this.router.navigateByUrl(result.route);
    this.searchService.close();
  }

  getFlatIndex(result: SearchResult): number {
    return this.filteredResults().indexOf(result);
  }

  private selectCurrent() {
    const results = this.filteredResults();
    const idx = this.activeIndex();
    if (results[idx]) {
      this.selectResult(results[idx]);
    }
  }
}
