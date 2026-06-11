import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';
import { ChartToolbarComponent } from './chart-toolbar';

// Render-coverage spec: mounts the template and drives every @if/@else/@for arm
// plus the extractChartTable/formatTableCell helpers through a stubbed chart.
// The logic-only assertions live in chart-toolbar.spec.ts — not duplicated here.

function chartStub(option: unknown): ChartExportable {
  const inst = {
    getOption: () => option,
    getDataURL: () => 'data:image/png;base64,AAAA',
  };
  return { getChartInstance: () => inst as never };
}

async function mount(
  providers: unknown[] = [],
): Promise<ComponentFixture<ChartToolbarComponent>> {
  await configureTestBed({
    imports: [ChartToolbarComponent],
    withHttp: false,
    providers: [ICON_PROVIDER, ...(providers as never[])],
  });
  const fixture = TestBed.createComponent(ChartToolbarComponent);
  fixture.detectChanges();
  return fixture;
}

describe('ChartToolbarComponent render-coverage', () => {
  describe('toolbar toggles and outputs (no chart)', () => {
    const TOGGLES = [
      { input: 'showExportPng', aria: 'Export chart as PNG' },
      { input: 'showExportSvg', aria: 'Export chart as SVG' },
      { input: 'showFullscreen', aria: 'Toggle fullscreen' },
      { input: 'showDataTable', aria: 'Toggle data table' },
    ];

    it('when all toggles are on, every toolbar button renders', async () => {
      const fixture = await mount();
      for (const t of TOGGLES) {
        expect(
          fixture.nativeElement.querySelector(`[aria-label="${t.aria}"]`),
        ).not.toBeNull();
      }
    });

    TOGGLES.forEach((t) => {
      it(`when ${t.input} is false, its button is absent`, async () => {
        const fixture = await mount();
        fixture.componentRef.setInput(t.input, false);
        fixture.detectChanges();
        expect(
          fixture.nativeElement.querySelector(`[aria-label="${t.aria}"]`),
        ).toBeNull();
      });
    });

    it('when isFullscreen flips, the icon swaps maximize to minimize', async () => {
      const fixture = await mount();
      expect(fixture.nativeElement.querySelector('i-lucide[name="maximize"]')).not.toBeNull();
      expect(fixture.nativeElement.querySelector('i-lucide[name="minimize"]')).toBeNull();

      fixture.componentInstance.isFullscreen.set(true);
      fixture.detectChanges();
      expect(fixture.nativeElement.querySelector('i-lucide[name="minimize"]')).not.toBeNull();
      expect(fixture.nativeElement.querySelector('i-lucide[name="maximize"]')).toBeNull();
    });

    it('when the data table is visible, the toggle button takes the active class', async () => {
      const fixture = await mount();
      const btn = fixture.nativeElement.querySelector(
        '[aria-label="Toggle data table"]',
      ) as HTMLElement;
      expect(btn.className).not.toContain('text-chart-1');

      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      expect(btn.className).toContain('text-chart-1');
    });

    it('when the PNG button is clicked, exportPng emits', async () => {
      const fixture = await mount();
      let emitted = false;
      fixture.componentInstance.exportPng.subscribe(() => (emitted = true));
      (fixture.nativeElement.querySelector('[aria-label="Export chart as PNG"]') as HTMLElement).click();
      expect(emitted).toBe(true);
    });

    it('when the SVG button is clicked, exportSvg emits', async () => {
      const fixture = await mount();
      let emitted = false;
      fixture.componentInstance.exportSvg.subscribe(() => (emitted = true));
      (fixture.nativeElement.querySelector('[aria-label="Export chart as SVG"]') as HTMLElement).click();
      expect(emitted).toBe(true);
    });

    it('when the fullscreen button is clicked, fullscreenToggle emits', async () => {
      const fixture = await mount();
      const wrapper = fixture.nativeElement.querySelector('div') as HTMLElement;
      spyOn(wrapper, 'requestFullscreen').and.returnValue(Promise.resolve());
      let value: boolean | undefined;
      fixture.componentInstance.fullscreenToggle.subscribe((v) => (value = v));
      (fixture.nativeElement.querySelector('[aria-label="Toggle fullscreen"]') as HTMLElement).click();
      expect(value).toBe(true);
    });

    it('when the data-table button is clicked, visibility flips and the output emits', async () => {
      const fixture = await mount();
      const states: boolean[] = [];
      fixture.componentInstance.showDataTableToggle.subscribe((v) => states.push(v));
      (fixture.nativeElement.querySelector('[aria-label="Toggle data table"]') as HTMLElement).click();
      expect(fixture.componentInstance.isDataTableVisible()).toBe(true);
      expect(states).toEqual([true]);
    });

    it('when already fullscreen, onFullscreenToggle exits fullscreen', async () => {
      const fixture = await mount();
      const fakeEl = document.createElement('div');
      Object.defineProperty(document, 'fullscreenElement', {
        value: fakeEl,
        configurable: true,
      });
      const exitSpy = spyOn(document, 'exitFullscreen').and.returnValue(Promise.resolve());
      fixture.componentInstance.onFullscreenToggle();
      expect(exitSpy).toHaveBeenCalled();
      Object.defineProperty(document, 'fullscreenElement', {
        value: null,
        configurable: true,
      });
    });

    it('when a fullscreenchange event fires, isFullscreen mirrors document state', async () => {
      const fixture = await mount();
      document.dispatchEvent(new Event('fullscreenchange'));
      expect(fixture.componentInstance.isFullscreen()).toBe(false);
    });
  });

  describe('data table panel — populated', () => {
    it('when visible with a series-bearing chart, the table renders one row per data point', async () => {
      const fixture = await mount([
        {
          provide: CHART_EXPORTABLE,
          useValue: chartStub({
            xAxis: { data: ['Jan', 'Feb'] },
            series: [
              { name: 'Alpha', data: [1, 2000, null, NaN, [1, 2], { a: 1 }] },
              { data: [0.001, 5, 'str'] },
            ],
          }),
        },
      ]);
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();

      expect(fixture.nativeElement.querySelectorAll('thead th').length).toBe(3);
      expect(fixture.nativeElement.querySelectorAll('tbody tr').length).toBe(6);
    });

    it('when the chart xAxis is an array, the table still renders', async () => {
      const fixture = await mount([
        {
          provide: CHART_EXPORTABLE,
          useValue: chartStub({ xAxis: [{ data: ['x'] }], series: [{ name: 'S', data: [9] }] }),
        },
      ]);
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();

      expect(fixture.nativeElement.querySelectorAll('thead th').length).toBe(2);
      expect(fixture.nativeElement.querySelectorAll('tbody tr').length).toBe(1);
    });

    it('when the PNG button is clicked with a chart, the export path runs and emits', async () => {
      const fixture = await mount([
        {
          provide: CHART_EXPORTABLE,
          useValue: chartStub({ series: [{ name: 'S', data: [1] }] }),
        },
      ]);
      let emitted = false;
      fixture.componentInstance.exportPng.subscribe(() => (emitted = true));
      fixture.componentInstance.onExportPng();
      expect(emitted).toBe(true);
    });

    it('when the SVG button is clicked with a chart, the export path runs and emits', async () => {
      const fixture = await mount([
        {
          provide: CHART_EXPORTABLE,
          useValue: chartStub({ series: [{ name: 'S', data: [1] }] }),
        },
      ]);
      let emitted = false;
      fixture.componentInstance.exportSvg.subscribe(() => (emitted = true));
      fixture.componentInstance.onExportSvg();
      expect(emitted).toBe(true);
    });
  });

  describe('data table panel — empty', () => {
    it('when visible with a chart that has no series, the empty-state message renders', async () => {
      const fixture = await mount([
        { provide: CHART_EXPORTABLE, useValue: chartStub({ series: [] }) },
      ]);
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();

      expect(fixture.nativeElement.textContent).toContain('No tabular data available');
    });

    it('when a chart series carries non-array data, the table renders no rows', async () => {
      const fixture = await mount([
        { provide: CHART_EXPORTABLE, useValue: chartStub({ series: [{ name: 'X', data: 5 }] }) },
      ]);
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();
      expect(fixture.nativeElement.textContent).toContain('No tabular data available');
    });

    it('when getOption returns a non-object, the table is empty', async () => {
      const fixture = await mount([
        { provide: CHART_EXPORTABLE, useValue: chartStub(null) },
      ]);
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();
      expect(fixture.nativeElement.textContent).toContain('No tabular data available');
    });

    it('when getOption throws, the table falls back to empty', async () => {
      const throwing = {
        getChartInstance: () => ({
          getOption: () => {
            throw new Error('boom');
          },
          getDataURL: () => '',
        }),
      } as never;
      const fixture = await mount([{ provide: CHART_EXPORTABLE, useValue: throwing }]);
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();
      await fixture.whenStable();
      fixture.detectChanges();
      expect(fixture.nativeElement.textContent).toContain('No tabular data available');
    });

    it('when visible with no resolvable chart, it falls through the async lookup to an empty table', async () => {
      // No CHART_EXPORTABLE provider → resolveChart() is null, exercising the
      // ensureGetInstanceByDom() dynamic-import fallback in refreshDataTable.
      // That import settles asynchronously, so poll the rendered output until
      // the empty-state message appears (bounded) instead of guessing a delay.
      const fixture = await mount();
      fixture.componentInstance.isDataTableVisible.set(true);
      fixture.detectChanges();

      const text = () => fixture.nativeElement.textContent as string;
      for (let i = 0; i < 100 && !text().includes('No tabular data available'); i++) {
        await new Promise((r) => setTimeout(r, 10));
        fixture.detectChanges();
      }

      expect(text()).toContain('No tabular data available');
    });
  });
});
