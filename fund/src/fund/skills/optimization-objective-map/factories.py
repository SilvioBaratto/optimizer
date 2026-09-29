"""Illustrative optimizer factory skeletons for `optimization-objective-map`.

READ-ONLY reference for the allocator agent: it shows the exact skfolio/optimizer
call shape per objective so the LLM emits a matching *config*, never weights. This
file is a text resource — its directory name has a hyphen, so it is NOT importable;
the real optimizer wiring lives behind the `optimize_portfolio` @tool (Phase 3).

Load-bearing rule: the LLM chooses the objective / measure / knobs; skfolio computes
the weights. Feed LINEAR returns (`prices_to_returns`); `shuffle=False` in any CV.
"""
