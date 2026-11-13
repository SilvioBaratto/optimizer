# Requirements registry

The list of requirement ids that tests cite (standard 6.6). A requirement is one
testable statement about the behaviour of a named module, with the standard clause or
design decision it answers.

- Ids have the form `REQ-<AREA>-NNN`, where `AREA` is `CORE`, `DB`, `ING`, `FUND` or `OPS`
  and `NNN` is a three-digit number that is never reused.
- A test binds to a requirement with `@pytest.mark.criterion("REQ-...")`.
- A check in CI will fail when a registered requirement has no test, or when a test cites
  an id that is not registered. It is not built yet; until it is, this registry is filled
  in by hand and the gap is recorded as a deviation in `CONTRIBUTING.md`.
- Do not invent an id outside this table.

| id | statement | module | standard |
|---|---|---|---|
