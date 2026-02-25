export function readCssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function getPortfolioTheme(): Record<string, unknown> {
  return {
    color: [
      readCssVar('--color-chart-1'),
      readCssVar('--color-chart-2'),
      readCssVar('--color-chart-3'),
      readCssVar('--color-chart-4'),
      readCssVar('--color-chart-5'),
      readCssVar('--color-chart-6'),
      readCssVar('--color-chart-7'),
      readCssVar('--color-chart-8'),
    ],
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: readCssVar('--color-surface-raised'),
      borderColor: readCssVar('--color-border'),
      borderWidth: 1,
      textStyle: {
        fontFamily: readCssVar('--font-sans'),
        color: readCssVar('--color-text'),
        fontSize: 12,
      },
    },
    categoryAxis: {
      axisLabel: {
        fontFamily: readCssVar('--font-mono'),
        color: readCssVar('--color-text-secondary'),
        fontSize: 10,
      },
      axisLine: { lineStyle: { color: readCssVar('--color-border') } },
      axisTick: { show: false },
      splitLine: {
        lineStyle: {
          color: readCssVar('--color-border-muted'),
          type: 'dashed' as const,
        },
      },
    },
    valueAxis: {
      axisLabel: {
        fontFamily: readCssVar('--font-mono'),
        color: readCssVar('--color-text-secondary'),
        fontSize: 10,
      },
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: {
        lineStyle: {
          color: readCssVar('--color-border-muted'),
          type: 'dashed' as const,
        },
      },
    },
    legend: {
      textStyle: {
        color: readCssVar('--color-text-secondary'),
        fontSize: 11,
      },
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 12,
    },
    line: {
      smooth: false,
      lineStyle: { width: 2 },
      symbolSize: 5,
      symbol: 'circle',
    },
    bar: {
      barMaxWidth: 24,
      itemStyle: {
        borderRadius: [2, 2, 0, 0],
      },
    },
    pie: {
      label: { show: false },
      emphasis: {
        itemStyle: {
          shadowBlur: 6,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0,0,0,0.15)',
        },
      },
    },
    animation: true,
    animationDuration: 300,
    animationEasing: 'cubicOut',
    animationDurationUpdate: 200,
    visualMap: {
      inRange: {
        color: [
          readCssVar('--color-heatmap-1'),
          readCssVar('--color-heatmap-2'),
          readCssVar('--color-heatmap-3'),
          readCssVar('--color-heatmap-4'),
          readCssVar('--color-heatmap-5'),
          readCssVar('--color-heatmap-6'),
          readCssVar('--color-heatmap-7'),
        ],
      },
      textStyle: {
        color: readCssVar('--color-text-secondary'),
        fontSize: 10,
      },
    },
  };
}

export async function registerPortfolioTheme(): Promise<void> {
  const { registerTheme } = await import('echarts/core');
  registerTheme('portfolio', getPortfolioTheme());
}
