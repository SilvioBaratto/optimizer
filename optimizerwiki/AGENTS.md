<!-- OPENWIKI:START -->

## OpenWiki

This repository has a generated `openwiki/` evidence index. It is optional just-in-time context, not required startup reading.

- Treat source code and tests as authoritative. A brief's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

## GitHub math rendering

GitHub renders Markdown math (`$...$`, `$$...$$`) only after CommonMark/GFM inline parsing, so an LLM-authored formula silently fails whenever its `$` delimiters are consumed first: by an emphasis pair (`*...*` / `_..._`, e.g. `w^*` or `R^{opt}_{t+1}`), an escaped `\$`, a `$` glued to adjacent punctuation, or a multi-line `$$` block not surrounded by blank lines. Both the hand-written `docs/` and the generated `openwiki/` therefore use the GitHub-safe forms:

- inline math as `` $`...`$ `` (the backticks shield the body from markdown);
- display math as a `$$` block with a blank line before and after (single-line content may stay glued to the delimiters);
- literal currency as `\$` (e.g. `\$20M`), never a bare `$`.

`scripts/fix-github-math.mjs` applies these transforms deterministically and is idempotent. The OpenWiki page-writer should emit these forms directly; the update workflow also runs the script after every regeneration so any slips are corrected before the PR. After regenerating OpenWiki by any other path, run `node scripts/fix-github-math.mjs openwiki` from the wiki root before committing.

<!-- OPENWIKI:END -->
