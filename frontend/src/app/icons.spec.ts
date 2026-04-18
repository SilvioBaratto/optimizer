import { LucideIconProvider, LucideIconProviderInterface } from 'lucide-angular';
import { ICON_PROVIDER } from './icons';

const TEMPLATE_NAMES = [
  'x',
  'layout-dashboard',
  'briefcase',
  'pie-chart',
  'clock',
  'shield-alert',
  'flask-conical',
  'arrow-left-right',
  'bar-chart-3',
  'globe',
  'cpu',
  'settings',
  'download',
  'copy',
  'arrow-up',
  'arrow-down',
  'search',
  'menu',
  'triangle-alert',
  'upload',
  'plus',
  'circle-check',
  'circle-x',
  'trash-2',
  'play',
  'chevron-down',
  'refresh-cw',
  'info',
  'sparkles',
  'loader-2',
  'image',
  'file-code',
  'maximize',
  'minimize',
  'table',
  'file-down',
  'trending-up',
  'newspaper',
  'file-text',
  'settings-2',
  'layers',
  'database',
  'target',
  'check-circle',
  'sliders',
  'function-square',
  'printer',
  'activity',
];

function toPascalCase(name: string): string {
  return name.replace(/(\w)([a-z0-9]*)(_|-|\s*)/g, (_match, head, tail) =>
    head.toUpperCase() + tail.toLowerCase(),
  );
}

describe('ICON_PROVIDER', () => {
  let provider: LucideIconProviderInterface;

  beforeEach(() => {
    provider = ICON_PROVIDER.useValue as LucideIconProvider;
  });

  it('exposes the LUCIDE_ICONS injection token as a multi provider', () => {
    expect(ICON_PROVIDER.multi).toBe(true);
    expect(ICON_PROVIDER.useValue).toBeInstanceOf(LucideIconProvider);
  });

  describe('newly registered icons (issue #431)', () => {
    it('resolves the file-down icon for the Dashboard Export Report button', () => {
      expect(provider.hasIcon(toPascalCase('file-down'))).toBe(true);
    });

    it('resolves the trending-up icon for the AI Control Room pipeline', () => {
      expect(provider.hasIcon(toPascalCase('trending-up'))).toBe(true);
    });
  });

  describe('full template + dynamic binding coverage', () => {
    for (const templateName of TEMPLATE_NAMES) {
      it(`resolves "${templateName}" used by templates or dynamic icon bindings`, () => {
        expect(provider.hasIcon(toPascalCase(templateName))).toBe(true);
      });
    }
  });
});
