import type { PipelineStepId } from '../../models/pipeline-builder.model';

export type BuilderStage =
  | 'universe'
  | 'objective'
  | 'constraints'
  | 'optimize'
  | 'review'
  | 'rebalance';

export const BUILDER_STAGES: readonly BuilderStage[] = [
  'universe',
  'objective',
  'constraints',
  'optimize',
  'review',
  'rebalance',
] as const;

export const STAGE_LABELS: Record<BuilderStage, string> = {
  universe: 'Universe',
  objective: 'Objective',
  constraints: 'Constraints',
  optimize: 'Optimize',
  review: 'Review',
  rebalance: 'Rebalance',
} as const;

export type AccordionSectionId =
  | 'universe'
  | 'objective'
  | 'constraints'
  | 'moments'
  | 'horizon';

export const ACCORDION_SECTIONS: readonly AccordionSectionId[] = [
  'universe',
  'objective',
  'constraints',
  'moments',
  'horizon',
] as const;

export const ACCORDION_LABELS: Record<AccordionSectionId, string> = {
  universe: 'Universe',
  objective: 'Objective',
  constraints: 'Constraints',
  moments: 'Moments',
  horizon: 'Horizon',
} as const;

export const SECTION_STEP_MAP: Record<
  AccordionSectionId,
  readonly PipelineStepId[]
> = {
  universe: ['load', 'screen'],
  objective: ['validate_is', 'validate_oos', 'coverage_gate'],
  constraints: ['regime', 'rebalance_decision'],
  moments: ['clean_returns', 'build_history'],
  horizon: ['optimize', 'cost', 'report', 'persist'],
} as const;

export function stepsForSection(
  id: AccordionSectionId,
): readonly PipelineStepId[] {
  return SECTION_STEP_MAP[id];
}

export function sectionForStep(id: PipelineStepId): AccordionSectionId {
  for (const section of ACCORDION_SECTIONS) {
    if (SECTION_STEP_MAP[section].includes(id)) return section;
  }
  throw new Error(`No accordion section owns step "${id}"`);
}

// Stage→steps map for the 6 strip stages. Reuses the matching accordion
// section subset for universe / objective / constraints; the trailing three
// stages map to the panels they surface in the user-flow journey:
//   optimize  → [optimize]
//   review    → [cost, report]
//   rebalance → [rebalance_decision, persist]
// `clean_returns` / `build_history` live in the `moments` accordion section
// only and are intentionally not surfaced as a strip stage.
export const STAGE_STEP_MAP: Record<BuilderStage, readonly PipelineStepId[]> = {
  universe: SECTION_STEP_MAP.universe,
  objective: SECTION_STEP_MAP.objective,
  constraints: SECTION_STEP_MAP.constraints,
  optimize: ['optimize'],
  review: ['cost', 'report'],
  rebalance: ['rebalance_decision', 'persist'],
} as const;

export function stepsForStage(stage: BuilderStage): readonly PipelineStepId[] {
  return STAGE_STEP_MAP[stage];
}
