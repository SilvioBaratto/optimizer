import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { ContextBarComponent } from './context-bar';
import { PortfolioContextService } from '../../../services/portfolio-context.service';
import { ICON_PROVIDER } from '../../../icons';

describe('ContextBarComponent', () => {
  let ctx: PortfolioContextService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ContextBarComponent],
      providers: [provideZonelessChangeDetection(), ICON_PROVIDER],
    });
    ctx = TestBed.inject(PortfolioContextService);
  });

  function render(): HTMLElement {
    const fixture = TestBed.createComponent(ContextBarComponent);
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  function findPresetButton(root: HTMLElement, label: string): HTMLButtonElement {
    const candidates = Array.from(
      root.querySelectorAll<HTMLButtonElement>('[data-testid="preset-btn"]'),
    );
    const match = candidates.find((b) => b.textContent?.trim() === label);
    if (!match) {
      throw new Error(`preset button "${label}" not found`);
    }
    return match;
  }

  it('clicking the 1Y preset sets the global dateRange preset to 1Y', () => {
    const root = render();

    findPresetButton(root, '1Y').click();

    expect(ctx.dateRange().preset).toBe('1Y');
  });

  it('first visible focusable element inside the bar can receive DOM focus', () => {
    const fixture = TestBed.createComponent(ContextBarComponent);
    document.body.appendChild(fixture.nativeElement);
    fixture.detectChanges();
    try {
      const root = fixture.nativeElement as HTMLElement;
      const candidates = Array.from(
        root.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      );
      const firstVisible = candidates.find((el) => el.offsetParent !== null);
      expect(firstVisible).toBeDefined();

      firstVisible!.focus();

      expect(document.activeElement).toBe(firstVisible!);
    } finally {
      document.body.removeChild(fixture.nativeElement);
    }
  });

  it('buttons use transparent-ring focus styles (focus:ring-1 focus:ring-accent)', () => {
    const root = render();
    const select = root.querySelector<HTMLSelectElement>('select');
    expect(select).not.toBeNull();
    // The benchmark <select> uses tailwind focus:outline-none focus:ring-1 focus:ring-accent
    expect(select!.className).toContain('focus:ring-1');
    expect(select!.className).toContain('focus:ring-accent');
  });

  it('Escape closes the date-range custom popover when it is open', async () => {
    const fixture = TestBed.createComponent(ContextBarComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    const customToggle = root.querySelector<HTMLButtonElement>(
      '[data-testid="custom-toggle"]',
    );
    expect(customToggle).not.toBeNull();
    customToggle!.click();
    fixture.detectChanges();
    expect(root.querySelector('[data-testid="custom-panel"]')).not.toBeNull();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    fixture.detectChanges();

    expect(root.querySelector('[data-testid="custom-panel"]')).toBeNull();
  });

  it('benchmark select change updates PortfolioContextService.benchmark', () => {
    const fixture = TestBed.createComponent(ContextBarComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    const select = root.querySelector<HTMLSelectElement>('select');
    expect(select).not.toBeNull();
    select!.value = 'URTH';
    select!.dispatchEvent(new Event('change'));

    expect(ctx.benchmark()).toBe('URTH');
  });
});
