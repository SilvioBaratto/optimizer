import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ChartToolbarComponent } from './chart-toolbar';

describe('ChartToolbarComponent', () => {
  let fixture: ComponentFixture<ChartToolbarComponent>;
  let comp: ChartToolbarComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [ChartToolbarComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ChartToolbarComponent);
    comp = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('when no chart is provided, exporting PNG falls back to just emitting (degraded path)', () => {
    let emitted = false;
    comp.exportPng.subscribe(() => (emitted = true));
    comp.onExportPng();
    expect(emitted).toBe(true);
  });

  it('when no chart is provided, exporting SVG falls back to just emitting (degraded path)', () => {
    let emitted = false;
    comp.exportSvg.subscribe(() => (emitted = true));
    comp.onExportSvg();
    expect(emitted).toBe(true);
  });

  it('when the data-table toggle is invoked, visibility flips and the toggle output emits', () => {
    const states: boolean[] = [];
    comp.showDataTableToggle.subscribe((v) => states.push(v));
    comp.onDataTableToggle();
    expect(comp.isDataTableVisible()).toBe(true);
    comp.onDataTableToggle();
    expect(comp.isDataTableVisible()).toBe(false);
    expect(states).toEqual([true, false]);
  });

  it('when fullscreen is toggled, the fullscreenToggle output emits', () => {
    const wrapper = fixture.nativeElement.querySelector('div') as HTMLElement;
    spyOn(wrapper, 'requestFullscreen').and.returnValue(Promise.resolve());
    let emitted: boolean | undefined;
    comp.fullscreenToggle.subscribe((v) => (emitted = v));
    comp.onFullscreenToggle();
    expect(emitted).toBe(true);
  });

  it('when showExportPng is false, no PNG button is rendered', () => {
    fixture.componentRef.setInput('showExportPng', false);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[aria-label="Export chart as PNG"]')).toBeNull();
  });

  it('when destroyed, ngOnDestroy is safe', () => {
    expect(() => comp.ngOnDestroy()).not.toThrow();
  });
});
