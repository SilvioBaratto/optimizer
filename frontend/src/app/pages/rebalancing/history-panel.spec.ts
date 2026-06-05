import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeSnapshotDto } from '../../../testing';
import { HistoryPanelComponent } from './history-panel';

describe('HistoryPanelComponent', () => {
  let fixture: ComponentFixture<HistoryPanelComponent>;
  let comp: HistoryPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [HistoryPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(HistoryPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when snapshots are empty, there are no rows', () => {
    expect(comp.tableRows()).toEqual([]);
  });

  it('when snapshots exist, they are sorted by date descending', () => {
    fixture.componentRef.setInput('snapshots', [
      makeSnapshotDto({ snapshot_date: '2026-01-01' }),
      makeSnapshotDto({ snapshot_date: '2026-03-01' }),
    ]);
    expect(comp.tableRows()[0]['snapshot_date']).toBe('2026-03-01');
  });

  it('when turnover is null, the cell shows the dash; a number is percent-formatted', () => {
    fixture.componentRef.setInput('snapshots', [
      makeSnapshotDto({ turnover: null }),
      makeSnapshotDto({ snapshot_date: '2026-02-01', turnover: 0.1234 }),
    ]);
    const rows = comp.tableRows();
    expect(rows.find((r) => r['turnover'] === '—')).toBeTruthy();
    expect(rows.find((r) => r['turnover'] === '12.34%')).toBeTruthy();
  });

  it('when rows are mapped, the snapshot_type passes through for the badge map', () => {
    fixture.componentRef.setInput('snapshots', [makeSnapshotDto({ snapshot_type: 'rebalance' })]);
    expect(comp.tableRows()[0]['snapshot_type']).toBe('rebalance');
  });
});
