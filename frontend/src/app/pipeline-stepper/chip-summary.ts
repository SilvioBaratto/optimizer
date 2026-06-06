import type { PipelineStepId } from '../models/pipeline-builder.model';
import {
  type AccordionSectionId,
  type BuilderStage,
  stepsForSection,
  stepsForStage,
} from '../portfolio-builder/models/builder-stage';
import { stepSummary } from './step-summary';

type ResultMap = Map<PipelineStepId, Record<string, unknown> | null>;

const CHIP_SEPARATOR = ' · ';

function joinStepSummaries(
  steps: readonly PipelineStepId[],
  results: ResultMap,
): string {
  return steps
    .map((id) => stepSummary(id, results.get(id) ?? null))
    .filter((s) => s.length > 0)
    .join(CHIP_SEPARATOR);
}

export function sectionSummary(
  sectionId: AccordionSectionId,
  results: ResultMap,
): string {
  return joinStepSummaries(stepsForSection(sectionId), results);
}

export function stageSummary(stage: BuilderStage, results: ResultMap): string {
  return joinStepSummaries(stepsForStage(stage), results);
}
