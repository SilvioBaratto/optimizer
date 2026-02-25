import { InjectionToken } from '@angular/core';
import type { EChartsType } from 'echarts/core';

export interface ChartExportable {
  getChartInstance(): EChartsType | undefined;
}

export const CHART_EXPORTABLE = new InjectionToken<ChartExportable>('CHART_EXPORTABLE');
