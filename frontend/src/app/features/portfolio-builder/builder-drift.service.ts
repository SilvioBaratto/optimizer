import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import {
  DestroyRef,
  Injectable,
  Injector,
  effect,
  inject,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { type Observable, Subject, merge, of } from 'rxjs';
import { catchError, filter, map, switchMap, tap } from 'rxjs/operators';

import { environment } from '../../../environments/environment';
import type {
  BaseToggle,
  BuilderTrade,
  DriftResponse,
  DriftRow,
  TradeRow,
} from '../../models/drift.model';
import type {
  BuilderDrift,
  BuilderDriftDiagnostics,
  BuilderTrades,
} from './builder-result.model';
import { BuilderStore, type DriftRunner } from './builder.store';

interface DriftPacket {
  readonly seq: number;
  readonly data: DriftResponse | null;
  readonly error: HttpErrorResponse | null;
}

function buildDriftUrl(name: string): string {
  return `${environment.apiUrl}portfolio/${encodeURIComponent(name)}/drift`;
}

function mergeTrades(
  trades: readonly TradeRow[],
  drift: readonly DriftRow[],
): BuilderTrade[] {
  const deltaByTicker = new Map<string, number>(
    drift.map((r) => [r.ticker, r.delta_weight]),
  );
  return trades.map((t) => ({
    ...t,
    delta_weight: deltaByTicker.get(t.ticker) ?? 0,
  }));
}

function toBuilderDrift(data: DriftResponse, name: string): BuilderDrift {
  return { drift: data.drift, totals: data.totals, portfolioName: name };
}

function toBuilderTrades(data: DriftResponse): BuilderTrades {
  return { trades: mergeTrades(data.trades, data.drift) };
}

function toBuilderDriftDiagnostics(
  data: DriftResponse,
): BuilderDriftDiagnostics {
  return { diagnostics: data.diagnostics, requestId: data.request_id };
}

@Injectable()
export class BuilderDriftService implements DriftRunner {
  private readonly store = inject(BuilderStore);
  private readonly http = inject(HttpClient);
  private readonly destroyRef = inject(DestroyRef);
  private readonly injector = inject(Injector);
  private readonly _trigger$ = new Subject<void>();
  private readonly _baseChange$ = new Subject<BaseToggle>();
  private _initialized = false;
  private _seq = 0;
  private _autoFired = false;
  private _lastBase: BaseToggle = 'invested';

  init(): void {
    if (this._initialized) return;
    this._initialized = true;
    this._lastBase = this.store.deployableBase();
    this._registerAutoFireEffect();
    this._registerBaseChangeEffect();
    this._subscribe();
  }

  runExplicit(): void {
    this._trigger$.next();
  }

  private _subscribe(): void {
    merge(this._trigger$, this._baseChange$)
      .pipe(
        filter(() => this._guardsPass()),
        map(() => this._beginRun()),
        switchMap((seq) => this._fetch(seq)),
        filter((p) => p.seq === this._seq),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((p) => this._applyPacket(p));
  }

  private _beginRun(): number {
    const seq = ++this._seq;
    this.store.setDriftStatus('running');
    return seq;
  }

  private _registerAutoFireEffect(): void {
    effect(
      () => {
        const shouldFire =
          this.store.currentView() === 'drift' &&
          this.store.resultStatus() === 'ok';
        if (shouldFire && !this._autoFired) {
          this._autoFired = true;
          this._trigger$.next();
        } else if (!shouldFire) {
          this._autoFired = false;
        }
      },
      { injector: this.injector },
    );
  }

  private _registerBaseChangeEffect(): void {
    effect(
      () => {
        const curr = this.store.deployableBase();
        if (this._lastBase === curr) return;
        this._lastBase = curr;
        this._baseChange$.next(curr);
      },
      { injector: this.injector },
    );
  }

  private _guardsPass(): boolean {
    return (
      this.store.portfolioName() !== null &&
      this.store.resultStatus() === 'ok'
    );
  }

  private _fetch(seq: number): Observable<DriftPacket> {
    const name = this.store.portfolioName() as string;
    const headers = new HttpHeaders({ 'X-Drift-Request-Id': String(seq) });
    return this.http
      .get<DriftResponse>(buildDriftUrl(name), {
        params: { base: this.store.deployableBase() },
        headers,
      })
      .pipe(
        map((data): DriftPacket => ({ seq, data, error: null })),
        catchError((err: HttpErrorResponse) =>
          of<DriftPacket>({ seq, data: null, error: err }),
        ),
      );
  }

  private _applyPacket(packet: DriftPacket): void {
    if (packet.error || !packet.data) {
      this.store.setDriftStatus('error');
      return;
    }
    const name = this.store.portfolioName() as string;
    this.store.setDrift(toBuilderDrift(packet.data, name));
    this.store.setTrades(toBuilderTrades(packet.data));
    this.store.setDriftDiagnostics(toBuilderDriftDiagnostics(packet.data));
    this.store.setDriftStatus('ok');
  }
}
