import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
} from '@angular/core';
import { toObservable, toSignal } from '@angular/core/rxjs-interop';
import { forkJoin, of } from 'rxjs';
import { map, switchMap } from 'rxjs/operators';

import { YfinanceService } from '../../core/services/yfinance.service';
import type { TickerProfile } from '../../core/models/yfinance.model';
import { MacroIntelligenceService } from '../../macro-intelligence/macro-intelligence.service';
import type { BusinessCyclePhase } from '../../core/models/macro-intelligence.model';
import { EchartsDonutComponent, type PieSegment } from '../../shared/echarts-donut/echarts-donut';
import {
  DiversificationViolationsService,
  type SimpleProfile,
  type ViolationKind,
} from './diversification-violations.service';
import { mapCountryToRegion } from '../diversification.helper';

// ── Colour palette ────────────────────────────────────────────────────────

const PALETTE = [
  '#6366F1', '#22D3EE', '#F59E0B', '#10B981',
  '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6',
];

const FLAG_LABELS: Record<ViolationKind, string> = {
  'region-concentration':  'Region > 60%',
  'sector-concentration':  'Sector > 15%',
  'hhi':                   'High concentration (HHI)',
  'top4-concentration':    'Top-4 > 30%',
  'healthcare':            'Healthcare off-target',
  'technology':            'Technology off-target',
  'absent-sector':         'Missing major sector',
};

// ── Component ────────────────────────────────────────────────────────────

@Component({
  selector: 'app-diversification-preview',
  imports: [EchartsDonutComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (profiles().length > 0) {
      <div class="flex flex-col gap-4">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <p class="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Sector</p>
            <app-echarts-donut [segments]="sectorSegments()" data-chart="sector" />
          </div>
          <div>
            <p class="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Region</p>
            <app-echarts-donut [segments]="regionSegments()" data-chart="region" />
          </div>
        </div>

        @if (violations().length > 0) {
          <ul class="flex flex-wrap gap-2" aria-label="Diversification flags">
            @for (flag of violations(); track flag.kind) {
              <li
                [attr.data-flag]="flag.kind"
                [class.flag-escalated]="flag.escalated"
                class="rounded-full border px-2.5 py-0.5 text-xs font-medium
                       border-amber-300 bg-amber-50 text-amber-800"
              >
                {{ flagLabel(flag.kind) }}
              </li>
            }
          </ul>
        }
      </div>
    } @else {
      <p class="text-sm text-gray-400 italic">
        Select assets to preview diversification.
      </p>
    }
  `,
})
export class DiversificationPreviewComponent {
  readonly tickers = input<string[]>([]);

  private readonly yfinance = inject(YfinanceService);
  private readonly macro   = inject(MacroIntelligenceService);
  private readonly vioSvc  = inject(DiversificationViolationsService);

  // ── Data signals ────────────────────────────────────────────────────────

  readonly profiles = toSignal(
    toObservable(this.tickers).pipe(
      switchMap(ids =>
        ids.length === 0
          ? of([] as (TickerProfile | null)[])
          : forkJoin(ids.map(id => this.yfinance.getProfile(id))),
      ),
    ),
    { initialValue: [] as (TickerProfile | null)[] },
  );

  readonly regime = toSignal(
    this.macro.getMacroCalibration().pipe(map(r => r?.phase ?? null)),
    { initialValue: null as BusinessCyclePhase | null },
  );

  // ── Derived signals ─────────────────────────────────────────────────────

  private readonly simpleProfiles = computed(() =>
    this.toSimpleProfiles(this.profiles() ?? []),
  );

  readonly violations = computed(() =>
    this.vioSvc.computeViolations(this.simpleProfiles(), this.regime() ?? ''),
  );

  readonly sectorSegments = computed(() =>
    this.toSegments(this.breakdown(this.simpleProfiles(), 'sector')),
  );

  readonly regionSegments = computed(() =>
    this.toSegments(this.breakdown(this.simpleProfiles(), 'region')),
  );

  // ── Helpers ──────────────────────────────────────────────────────────────

  flagLabel(kind: ViolationKind): string {
    return FLAG_LABELS[kind] ?? kind;
  }

  private toSimpleProfiles(profs: (TickerProfile | null)[]): SimpleProfile[] {
    return profs
      .filter((p): p is TickerProfile => p !== null)
      .map(p => ({ sector: p.sector ?? 'Unknown', region: mapCountryToRegion(p.country) }));
  }

  private breakdown(profs: SimpleProfile[], key: keyof SimpleProfile): Record<string, number> {
    const n = profs.length;
    if (n === 0) return {};
    const w = 1 / n;
    return profs.reduce<Record<string, number>>((acc, p) => {
      acc[p[key]] = (acc[p[key]] ?? 0) + w;
      return acc;
    }, {});
  }

  private toSegments(bd: Record<string, number>): PieSegment[] {
    return Object.entries(bd).map(([label, value], i) => ({
      label,
      value: Math.round(value * 100),
      color: PALETTE[i % PALETTE.length],
    }));
  }
}
