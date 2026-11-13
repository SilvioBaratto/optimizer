## Summary

<!-- Brief description of changes, and why -->

## Related Issues

<!-- Link to related issues: Closes #XX, Fixes #YY -->

## Checklist

- [ ] Tests added/updated, and a regression test fails without the change
- [ ] `make lint` passes (ruff check + format --check on every workspace member)
- [ ] `make typecheck` passes (mypy)
- [ ] `CHANGELOG.md` updated

### Security and boundaries

- [ ] Input from outside the process (HTTP, scraped HTML, LLM output, CLI, files) is validated at the boundary
- [ ] No secret, API key or personal data in code, logs, error messages or `repr()`
- [ ] Import boundaries still hold (`portopt-core` and `portopt_db` do not import each other, `ingestion` does not import `optimizer`)

### Data and contracts

- [ ] A new or changed migration has a working `downgrade()`, or declares `IRREVERSIBLE_REASON` and says where the restore source is
- [ ] A change to a public function, config field or CLI flag is described in `CHANGELOG.md`
