import {
  ACCORDION_LABELS,
  ACCORDION_SECTIONS,
  type AccordionSectionId,
  BUILDER_STAGES,
  type BuilderStage,
  SECTION_STEP_MAP,
  STAGE_LABELS,
  STAGE_STEP_MAP,
  sectionForStep,
  stepsForSection,
  stepsForStage,
} from './builder-stage';
import type { PipelineStepId } from '../../core/models/pipeline-builder.model';

const ALL_STEP_IDS: readonly PipelineStepId[] = [
  'load',
  'screen',
  'clean_returns',
  'build_history',
  'validate_is',
  'validate_oos',
  'coverage_gate',
  'regime',
  'optimize',
  'rebalance_decision',
  'cost',
  'report',
  'persist',
];

describe('builder-stage', () => {
  it('when BUILDER_STAGES is read, it lists 6 stage ids in canonical order', () => {
    expect(BUILDER_STAGES).toEqual([
      'universe',
      'objective',
      'constraints',
      'optimize',
      'review',
      'rebalance',
    ] as readonly BuilderStage[]);
    expect(BUILDER_STAGES.length).toBe(6);
  });

  it('when STAGE_LABELS is indexed, every stage id maps to a non-empty display label', () => {
    for (const id of BUILDER_STAGES) {
      expect(STAGE_LABELS[id]).toBeTruthy();
    }
    expect(Object.keys(STAGE_LABELS).length).toBe(6);
  });

  it('when ACCORDION_SECTIONS is read, it lists 5 section ids in canonical order', () => {
    expect(ACCORDION_SECTIONS).toEqual([
      'universe',
      'objective',
      'constraints',
      'moments',
      'horizon',
    ] as readonly AccordionSectionId[]);
    expect(ACCORDION_SECTIONS.length).toBe(5);
  });

  it('when ACCORDION_LABELS is indexed, every section id maps to a non-empty display label', () => {
    for (const id of ACCORDION_SECTIONS) {
      expect(ACCORDION_LABELS[id]).toBeTruthy();
    }
    expect(Object.keys(ACCORDION_LABELS).length).toBe(5);
  });

  it('when SECTION_STEP_MAP is unioned, all 13 PipelineStepId values are reachable', () => {
    const reachable = new Set<PipelineStepId>();
    for (const section of ACCORDION_SECTIONS) {
      for (const step of SECTION_STEP_MAP[section]) {
        reachable.add(step);
      }
    }
    expect(reachable.size).toBe(13);
    for (const id of ALL_STEP_IDS) {
      expect(reachable.has(id)).toBe(true);
    }
  });

  it('when sections are partitioned, no PipelineStepId appears in two sections', () => {
    const counts = new Map<PipelineStepId, number>();
    for (const section of ACCORDION_SECTIONS) {
      for (const step of SECTION_STEP_MAP[section]) {
        counts.set(step, (counts.get(step) ?? 0) + 1);
      }
    }
    for (const [, count] of counts) {
      expect(count).toBe(1);
    }
  });

  it('when stepsForSection is called with each section id, it returns the mapped tuple', () => {
    expect(stepsForSection('universe')).toEqual(['load', 'screen']);
    expect(stepsForSection('objective')).toEqual([
      'validate_is',
      'validate_oos',
      'coverage_gate',
    ]);
    expect(stepsForSection('constraints')).toEqual([
      'regime',
      'rebalance_decision',
    ]);
    expect(stepsForSection('moments')).toEqual([
      'clean_returns',
      'build_history',
    ]);
    expect(stepsForSection('horizon')).toEqual([
      'optimize',
      'cost',
      'report',
      'persist',
    ]);
  });

  it('when sectionForStep is roundtripped through stepsForSection, every PipelineStepId is recovered', () => {
    for (const step of ALL_STEP_IDS) {
      const section = sectionForStep(step);
      expect(stepsForSection(section)).toContain(step);
    }
  });

  it('when STAGE_STEP_MAP is read, each strip stage maps to its expected step subset', () => {
    expect(STAGE_STEP_MAP.universe).toEqual(SECTION_STEP_MAP.universe);
    expect(STAGE_STEP_MAP.objective).toEqual(SECTION_STEP_MAP.objective);
    expect(STAGE_STEP_MAP.constraints).toEqual(SECTION_STEP_MAP.constraints);
    expect(STAGE_STEP_MAP.optimize).toEqual(['optimize']);
    expect(STAGE_STEP_MAP.review).toEqual(['cost', 'report']);
    expect(STAGE_STEP_MAP.rebalance).toEqual(['rebalance_decision', 'persist']);
  });

  it('when STAGE_STEP_MAP is unioned, every relevant PipelineStepId is reachable from some stage', () => {
    const reachable = new Set<PipelineStepId>();
    for (const stage of BUILDER_STAGES) {
      for (const step of STAGE_STEP_MAP[stage]) reachable.add(step);
    }
    const expected: readonly PipelineStepId[] = [
      'load',
      'screen',
      'validate_is',
      'validate_oos',
      'coverage_gate',
      'regime',
      'rebalance_decision',
      'optimize',
      'cost',
      'report',
      'persist',
    ];
    for (const id of expected) {
      expect(reachable.has(id)).toBe(true);
    }
  });

  it('when stepsForStage is called with each BuilderStage id, it returns STAGE_STEP_MAP[id]', () => {
    for (const stage of BUILDER_STAGES) {
      expect(stepsForStage(stage)).toBe(STAGE_STEP_MAP[stage]);
    }
  });

  it('when sectionForStep is called with each PipelineStepId, it returns the owning section', () => {
    expect(sectionForStep('load')).toBe('universe');
    expect(sectionForStep('screen')).toBe('universe');
    expect(sectionForStep('clean_returns')).toBe('moments');
    expect(sectionForStep('build_history')).toBe('moments');
    expect(sectionForStep('validate_is')).toBe('objective');
    expect(sectionForStep('validate_oos')).toBe(
      'objective',
    );
    expect(sectionForStep('coverage_gate')).toBe(
      'objective',
    );
    expect(sectionForStep('regime')).toBe('constraints');
    expect(sectionForStep('rebalance_decision')).toBe(
      'constraints',
    );
    expect(sectionForStep('optimize')).toBe('horizon');
    expect(sectionForStep('cost')).toBe('horizon');
    expect(sectionForStep('report')).toBe('horizon');
    expect(sectionForStep('persist')).toBe('horizon');
  });
});
