# Contributing to Portfolio Optimizer

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

The repository is a uv workspace of four packages that share one virtualenv (see
[ADR-0002](docs/adr/0002-four-package-uv-workspace.md)). Install through `uv`: the
daemon and the fund bridge depend on sibling packages that only uv can resolve.

```bash
# Clone the repository
git clone https://github.com/SilvioBaratto/optimizer.git
cd optimizer

# One environment for all four packages, with every extra
uv sync --all-packages --all-extras

# Install pre-commit hooks
pre-commit install
```

## Running Tests

Run each suite from the directory CI runs it from. From the repository root, `pytest`
always collects the library suite, whichever `--package` you pass.

```bash
# Library suite (portopt-core), from the repository root
uv run --package portopt-core pytest tests/ -v
uv run --package portopt-core pytest -k "test_name"

# Member suites, each from its own directory
cd ingestion && uv run pytest tests/
cd packages/portopt-db && uv run pytest
cd fund && uv run pytest tests/
```

### PostgreSQL integration suites

`portopt-db` and `fund` have tests that need a real server (`ON CONFLICT ON CONSTRAINT`,
the append-only audit triggers, the LangGraph schema). They skip unless a scratch
database is configured, and they refuse any database that is the live one. Never point
them at `optimizer_db`.

```bash
docker run -d --rm --name portopt_test_pg -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=portopt_test -p 127.0.0.1:54399:5432 postgres:16

# portopt-db: the database name must end in _test
cd packages/portopt-db && \
  PORTOPT_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:54399/portopt_test \
  uv run pytest tests/integration -m integration

# fund: FUND_TEST_DATABASE_URL, see fund/tests/integration
docker stop portopt_test_pg
```

## Code Quality

```bash
# Lint (the library plus ingestion/, fund/ and packages/portopt-db/)
ruff check packages/portopt-core/optimizer/ tests/ ingestion/ fund/ packages/portopt-db/

# Auto-fix lint issues
ruff check packages/portopt-core/optimizer/ tests/ ingestion/ fund/ packages/portopt-db/ --fix

# Format
ruff format packages/portopt-core/optimizer/ tests/ ingestion/ fund/ packages/portopt-db/

# Type check
mypy packages/portopt-core/optimizer/
```

Or use the Makefile, which runs through `uv` (run `make sync` once to build the
workspace venv):

```bash
make lint            # ruff check + format check (core, ingestion, fund and portopt-db)
make typecheck       # mypy on portopt-core, plus strict mypy on portopt-db and ingestion
make pyright         # pyright over all four implementation trees
make test            # portopt-core pytest with the 90% coverage floor
make test-db         # portopt-db pytest with the 90% coverage floor
make test-ingestion  # ingestion pytest with the 80% coverage floor
make test-fund       # fund pytest with the 80% coverage floor
make all             # every target above: a local mirror of the CI jobs
```

`make all` gates every workspace member, so it runs far longer than the
portopt-core slice. It leaves out `pip-audit` and the Python 3.12/3.13 matrix, which
only CI runs.

## Coding Standards

### Architecture Pattern

Every module follows the same pattern: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**.

- **Configs** hold only primitives, enums, and nested frozen dataclasses (serializable)
- **Non-serializable objects** (estimator instances, numpy arrays, callables) are passed as factory `**kwargs`
- All transformers follow the sklearn `BaseEstimator + TransformerMixin` API

### Example

```python
from dataclasses import dataclass
from enum import Enum

class MyMethodType(str, Enum):
    METHOD_A = "method_a"
    METHOD_B = "method_b"

@dataclass(frozen=True)
class MyConfig:
    method: MyMethodType = MyMethodType.METHOD_A
    threshold: float = 0.05

    @classmethod
    def for_conservative(cls) -> "MyConfig":
        return cls(method=MyMethodType.METHOD_A, threshold=0.01)

def build_my_estimator(config: MyConfig, **kwargs):
    """Factory function that creates the estimator from config."""
    ...
```

### Style

- **ruff** for linting and formatting (line-length 88, target py310)
- **mypy** strict mode for type checking
- Use lazy `%s`/`%d` formatting for log calls, never f-strings
- Follow sklearn naming conventions: `X`, `y` for data parameters

## Pull Request Process

The branching, review and secrets policy, including the one place it departs from the
code standard, is in
[ADR-0007](docs/adr/0007-branching-review-and-secrets-policy.md).

1. Create a branch from `development` named `feature/<slug>`, `bugfix/<slug>` or
   `hotfix/<slug>`
2. Make your changes with tests
3. Ensure all checks pass: `make all` (or the targets for the packages you touched)
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Push the branch. CI runs on every push to a prefixed branch, and `development` and
   `main` accept only commits whose twelve required checks are green
6. Open a pull request into `development` and merge it once the checks pass

### PR Checklist

- [ ] Tests added/updated for new functionality
- [ ] `ruff check` passes with no errors
- [ ] `ruff format --check` passes
- [ ] `make typecheck` passes
- [ ] `CHANGELOG.md` updated

### Commit Messages

Use conventional commit style:

- `feat:` new feature
- `fix:` bug fix
- `test:` adding or updating tests
- `docs:` documentation changes
- `ci:` CI/CD changes
- `refactor:` code refactoring
- `deps:` dependency updates

## Releasing

`main` receives releases only; direct commits to it are blocked.

1. Branch `release/0.x.0` from `development`
2. Set the version to `0.x.0` in the root `pyproject.toml` and keep the three member
   `pyproject.toml` files equal to it, then refresh `uv.lock`
3. Move `[Unreleased]` entries in `CHANGELOG.md` to a new `[0.x.0]` section
4. Open a pull request from the release branch into `main` and merge it once green
5. Tag the merge commit on `main`: `git tag -a v0.x.0` and `git push origin v0.x.0`
6. The release workflow rejects the tag unless its commit is on `main` and the tag
   equals `v` plus the `pyproject.toml` version, re-runs the full CI workflow, and
   only then publishes `portopt-core` to PyPI

## Test traceability (documented deviation from standard 6.6)

Standard 6.6 asks every test to cite the requirement ID it verifies. This
repository has no requirement registry yet, so most tests name only the code they
cover in their docstring. That is a known non-compliance (standard 1.3).

Corrective action: a registry of requirement IDs (`docs/requirements.md`, ids of the
form `REQ-<AREA>-NNN`), the `criterion(id)` pytest marker on the tests of each
critical function, and a CI step that fails when a registered requirement has no
test or a marker cites an unknown id. Until it lands, a new regression test names
the function it guards in its docstring and the fix is described in the
`CHANGELOG.md` entry. Do not invent ids outside the registry.

## Questions?

Open a [GitHub Discussion](https://github.com/SilvioBaratto/optimizer/discussions) or file an [issue](https://github.com/SilvioBaratto/optimizer/issues).
