import type { EChartsCoreOption } from 'echarts/core';

export const EMPTY_STATE_OPTION: EChartsCoreOption = {
  graphic: [
    {
      type: 'text',
      left: 'center',
      top: 'middle',
      style: { text: 'No data', fontSize: 13, fill: '#71717a' },
    },
  ],
};

export function safeChartOption<T>(
  data: T | null | undefined,
  builder: (d: NonNullable<T>) => EChartsCoreOption,
): EChartsCoreOption {
  if (data == null || (Array.isArray(data) && data.length === 0)) {
    return EMPTY_STATE_OPTION;
  }
  return builder(data as NonNullable<T>);
}
