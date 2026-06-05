import type { EChartsOption } from 'echarts';

// Assert on the option object, never on rendered pixels. Casts are local to
// these accessors so specs read clean.

type SeriesEntry = Record<string, unknown>;

/** Return the series object at `index` (series is treated as an array). */
export function getSeriesAt(option: EChartsOption, index = 0): SeriesEntry {
  const series = option.series as SeriesEntry[];
  return series[index];
}

/** Return the legend option record. */
export function getLegend(option: EChartsOption): Record<string, unknown> {
  return option.legend as Record<string, unknown>;
}

/** Return the tooltip option record. */
export function getTooltip(option: EChartsOption): Record<string, unknown> {
  return option.tooltip as Record<string, unknown>;
}

/** Return `series[index].label.show`. */
export function seriesLabelShow(option: EChartsOption, index = 0): unknown {
  const series = getSeriesAt(option, index) as { label?: { show?: unknown } };
  return series.label?.show;
}

/** Return `legend.type`. */
export function legendType(option: EChartsOption): unknown {
  return getLegend(option)['type'];
}
