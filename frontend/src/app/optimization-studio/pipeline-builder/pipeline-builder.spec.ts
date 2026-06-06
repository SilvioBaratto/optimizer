import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { PipelineBuilderComponent } from './pipeline-builder';
import type { PipelineNode, PipelineNodeStatus } from '../../core/models/optimization.model';

function node(id: string, status: PipelineNodeStatus): PipelineNode {
  return { id, label: id, status, detail: '' };
}

describe('PipelineBuilderComponent', () => {
  let fixture: ComponentFixture<PipelineBuilderComponent>;
  let comp: PipelineBuilderComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [PipelineBuilderComponent], withHttp: false });
    fixture = TestBed.createComponent(PipelineBuilderComponent);
    comp = fixture.componentInstance;
  });

  it('when nodes have mixed statuses, completed and total counts derive from them', () => {
    fixture.componentRef.setInput('nodes', [
      node('a', 'completed'),
      node('b', 'completed'),
      node('c', 'running'),
    ]);
    expect(comp.completedCount()).toBe(2);
    expect(comp.totalCount()).toBe(3);
  });

  const cases: ReadonlyArray<[PipelineNodeStatus, string, string]> = [
    ['pending', 'bg-text-tertiary', 'Pending'],
    ['running', 'bg-accent animate-pulse', 'Running'],
    ['completed', 'bg-gain', 'Done'],
    ['error', 'bg-loss', 'Error'],
  ];

  for (const [status, color, label] of cases) {
    it(`when status is ${status}, statusColor and statusLabel match`, () => {
      expect(comp.statusColor(status)).toBe(color);
      expect(comp.statusLabel(status)).toBe(label);
    });
  }

  it('when a node card is clicked, nodeSelect emits its id', () => {
    fixture.componentRef.setInput('nodes', [node('a', 'pending')]);
    fixture.detectChanges();
    let selected: string | undefined;
    comp.nodeSelect.subscribe((id) => (selected = id));
    (fixture.nativeElement.querySelector('button') as HTMLButtonElement).click();
    expect(selected).toBe('a');
  });
});
