# ADR-0007: Branching, review and secrets policy for a single maintainer

- Status: Accepted
- Date: 2026-11-13

## Context

The repository has one maintainer on a personal GitHub account. The code standard asks
for pull requests with mandatory review into the main branches (4.1, 5.2.2), protected
branches with no direct commits (5.2.1), `feature/`, `bugfix/` and `hotfix/` branch
prefixes (4.1) and a documented branching strategy (5.3).

The standards review of 2026-11-13 found that `development` had no protection, `main`
required three check names that no longer exist (`test (3.10)`, `test (3.11)`,
`test (3.12)`), none of the last fifteen pull requests had a review, and recent work used
`feat/` and `fix/` prefixes. It also found API keys, `.env` files and a database dump
inside the Syncthing folder that replicates to two other machines.

GitHub does not let an author approve their own pull request, so a mandatory-approval
rule would block every merge.

## Decision

**Branches.** `development` is the integration branch and `main` receives releases. Work
happens on `feature/`, `bugfix/`, `hotfix/` and `release/` branches (GitFlow without the
ceremony, standard 5.3.2). A pull request is the normal way in.

**Protection** (applied 2026-11-13 to `development` and `main`): the twelve CI jobs are
required, administrators are included (`enforce_admins`), force-push and deletion are off.
CI also runs on pushes to the prefixed branches, so a commit can carry the required
statuses without a pull request.

**Deviation from standards 4.1, 5.2.1 and 5.2.2, recorded under standard 1.3.** No pull
request and no approval is required. Compensating controls: the required checks (lint,
format, mypy, pyright, the four test suites with coverage floors, `pip-audit`, the image
build and the Postgres integration job), the complexity and length ratchets once they
land, and the standards-reviewer agent that reviews every changed file before a change is
finished. Review this deviation by 2027-05-13, or earlier if a second maintainer joins.

**Branch-name enforcement.** GitHub's branch-name rule returned HTTP 422 for this
repository, which GitHub restricts to organization-owned repositories, so the prefix
convention is kept by the local commit gate and by review, not by the server.

**Secrets.** Keys, `.env` files, exports and database dumps are never synced between
machines: `/Volumes/External SSD/.stignore` ignores `optimizer/secrets/*` (except its
`.gitignore`), `optimizer/.env`, `optimizer/exports` and `optimizer-backups/*.dump`. Each
machine provisions its own keys. Nothing secret is committed.

**Changelog.** Every change to shipped code adds an `[Unreleased]` entry (standard 5.1.2).

## Alternatives considered

- **Require one approval.** Closest to the standard, but it needs a second GitHub account
  or collaborator, or switching the rule off at every merge.
- **Required checks and a pull request, but no approval.** Rejected by the maintainer in
  favour of checks alone, so a fast-forward or merge of a commit that already carries
  green statuses is allowed.
- **No protection, rely on discipline.** The state found by the review: by its count,
  over two hundred first-parent commits reached `development` without a pull request
  between mid-September and mid-November (counted by commit subject and spot-checked).

## Consequences

- A commit without green required checks cannot reach `development` or `main`, for
  administrators too. The escape hatch is one call:
  `gh api -X DELETE repos/SilvioBaratto/optimizer/branches/<branch>/protection/enforce_admins`.
- The review-by date keeps the deviation from becoming permanent by default.
- A new CI job must be added to the required list on both branches, or it only advises.
- Old copies of keys and dumps already replicated to the other two machines stay there
  until they are deleted by hand.
