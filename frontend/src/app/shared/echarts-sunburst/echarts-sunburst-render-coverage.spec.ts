// Render-coverage spec for EchartsSunburstComponent.
// The base spec (echarts-sunburst.spec.ts) covers mount and ngOnDestroy.
// This file exercises buildOption branches:
//   - Single-path smoke: 2 root nodes → series type='sunburst', data.length=2.
//   - Depth-0 nodes get colors from chartColors palette (index-based).
//   - Depth-1 child nodes get parentColor with '99' opacity suffix.
//   - Nodes without children → children field is undefined in output.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
} from '../../../testing';
import {
  EchartsSunburstComponent,
  type SunburstNode,
} from './echarts-sunburst';

// ---------------------------------------------------------------------------
// Cast helper
// ---------------------------------------------------------------------------

function build(
  comp: EchartsSunburstComponent,
  data: SunburstNode[],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(d: SunburstNode[]): EChartsOption;
    }
  ).buildOption(data);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsSunburstComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsSunburstComponent>;
  let comp: EchartsSunburstComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsSunburstComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsSunburstComponent);
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  // ── Series type is 'sunburst' ─────────────────────────────────────────────

  it('series[0].type is "sunburst"', () => {
    const nodes: SunburstNode[] = [
      { name: 'Equity', value: 0.6 },
      { name: 'Fixed Income', value: 0.4 },
    ];
    const opt = build(comp, nodes);
    expect(getSeriesAt(opt, 0)['type']).toBe('sunburst');
  });

  // ── Root data length matches input ────────────────────────────────────────

  it('series[0].data.length equals the number of root nodes', () => {
    const nodes: SunburstNode[] = [
      { name: 'A', value: 0.5 },
      { name: 'B', value: 0.5 },
    ];
    const opt = build(comp, nodes);
    const series = getSeriesAt(opt, 0) as { data: unknown[] };
    expect(series.data.length).toBe(2);
  });

  // ── Root node names preserved ─────────────────────────────────────────────

  it('root node names are preserved in series data', () => {
    const nodes: SunburstNode[] = [
      { name: 'US', value: 0.7 },
      { name: 'EU', value: 0.3 },
    ];
    const opt = build(comp, nodes);
    const series = getSeriesAt(opt, 0) as {
      data: Array<{ name: string }>;
    };
    expect(series.data[0].name).toBe('US');
    expect(series.data[1].name).toBe('EU');
  });

  // ── Depth-0 color comes from chart palette ────────────────────────────────
  // In JSDOM, CSS vars resolve to ''. The palette color at index 0 is
  // readCssVar('--color-chart-1') which equals ''. The color is assigned
  // from chartColors[0] at depth 0.

  it('depth-0 nodes have itemStyle.color from palette (not undefined)', () => {
    const nodes: SunburstNode[] = [{ name: 'Root', value: 1 }];
    const opt = build(comp, nodes);
    const series = getSeriesAt(opt, 0) as {
      data: Array<{ itemStyle: { color: string } }>;
    };
    // Color must be assigned (string, even if empty in JSDOM)
    expect(typeof series.data[0].itemStyle.color).toBe('string');
  });

  // ── Child nodes at depth-1 get parentColor+'99' ───────────────────────────

  it('depth-1 child nodes have itemStyle.color = parentColor+"99"', () => {
    const nodes: SunburstNode[] = [
      {
        name: 'Parent',
        value: 1,
        children: [{ name: 'Child', value: 0.5 }],
      },
    ];
    const opt = build(comp, nodes);
    const series = getSeriesAt(opt, 0) as {
      data: Array<{
        itemStyle: { color: string };
        children: Array<{ itemStyle: { color: string } }>;
      }>;
    };
    const parentColor = series.data[0].itemStyle.color;
    const childColor = series.data[0].children[0].itemStyle.color;
    expect(childColor).toBe(`${parentColor}99`);
  });

  // ── Leaf nodes without children: children field is undefined ─────────────

  it('nodes without children have children=undefined in output', () => {
    const nodes: SunburstNode[] = [{ name: 'Leaf', value: 0.5 }];
    const opt = build(comp, nodes);
    const series = getSeriesAt(opt, 0) as {
      data: Array<{ children: unknown }>;
    };
    expect(series.data[0].children).toBeUndefined();
  });

  // ── Radius configuration ──────────────────────────────────────────────────

  it('series[0].radius is ["20%", "85%"]', () => {
    const opt = build(comp, [{ name: 'X', value: 1 }]);
    const series = getSeriesAt(opt, 0) as { radius: string[] };
    expect(series.radius).toEqual(['20%', '85%']);
  });
});
