# pipeline-stepper

Host-less panel library. Contains 14 step panels used by the portfolio-builder pipeline wizard.

## Role

This module is a panel library with no host component. It does not define a route or standalone page.

## Sanctioned consumer

`portfolio-builder/inputs-pane` is the only authorised importer of panels from this library.

Other imports (`stage-strip → chip-summary`, `builder.store → models/step-params.model`) are also sanctioned but scoped to specific utilities.

## Reverse edge (do not widen)

`chip-summary.ts` imports from `../portfolio-builder/models/builder-stage`. This is the one documented reverse edge (pipeline-stepper → portfolio-builder). Do not add new imports from pipeline-stepper to portfolio-builder.

## Structure

- `step-*-panel/` — trio subfolders (`.ts` + `.html` + `.spec.ts`) for each pipeline step panel
- `run-config-panel/` — trio subfolder for the run configuration panel
- `step-section/` — nested section component (already trio, leave as-is)
- `models/step-params.model.ts` — per-step parameter types
- `chip-summary.ts` — pure formatter utility (no template)
- `step-summary.ts` — pure formatter utility (no template)
