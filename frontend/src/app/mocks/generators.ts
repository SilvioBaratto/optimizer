// Shared generator utilities for mock data

export function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
}

export function generateDailyTimeSeries(
  startDate: string,
  days: number,
  drift: number,
  vol: number,
  startValue: number,
  seed = 42,
): TimeSeriesPoint[] {
  const rng = seededRandom(seed);
  const points: TimeSeriesPoint[] = [];
  let value = startValue;
  const start = new Date(startDate);
  let tradingDay = 0;
  let calendarDay = 0;

  while (tradingDay < days) {
    const d = new Date(start);
    d.setDate(d.getDate() + calendarDay);
    const dow = d.getDay();
    calendarDay++;

    if (dow === 0 || dow === 6) continue;

    const dailyDrift = drift / 252;
    const dailyVol = vol / Math.sqrt(252);
    const u1 = rng();
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    value = value * Math.exp(dailyDrift - 0.5 * dailyVol * dailyVol + dailyVol * z);

    points.push({
      date: d.toISOString().split('T')[0],
      value: Math.round(value * 10000) / 10000,
    });
    tradingDay++;
  }

  return points;
}

export function generateCorrelationMatrix(
  n: number,
  intraSectorCorr: number,
  interSectorCorr: number,
  sectorAssignments: number[],
  seed = 123,
): number[][] {
  const rng = seededRandom(seed);
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const base =
        sectorAssignments[i] === sectorAssignments[j]
          ? intraSectorCorr
          : interSectorCorr;
      const noise = (rng() - 0.5) * 0.1;
      const corr = Math.min(0.98, Math.max(-0.3, base + noise));
      const rounded = Math.round(corr * 100) / 100;
      matrix[i][j] = rounded;
      matrix[j][i] = rounded;
    }
  }

  return matrix;
}

export interface MonthlyGridCell {
  year: number;
  month: number;
  value: number;
}

export function generateMonthlyGrid(
  startYear: number,
  endYear: number,
  avgReturn: number,
  vol: number,
  seed = 77,
): MonthlyGridCell[] {
  const rng = seededRandom(seed);
  const cells: MonthlyGridCell[] = [];

  for (let y = startYear; y <= endYear; y++) {
    for (let m = 1; m <= 12; m++) {
      if (y === endYear && m > 2) break;
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const monthlyReturn = avgReturn / 12 + (vol / Math.sqrt(12)) * z;
      cells.push({
        year: y,
        month: m,
        value: Math.round(monthlyReturn * 10000) / 10000,
      });
    }
  }

  return cells;
}
