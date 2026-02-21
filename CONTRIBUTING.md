# Contributing to Portfolio Optimizer

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/SilvioBaratto/optimizer.git
cd optimizer

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with all development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific module
pytest tests/rebalancing/ -v

# Run a single test by name
pytest -k "test_name"

# Run with coverage
pytest tests/ --cov=optimizer --cov-report=term-missing
```

## Code Quality

```bash
# Lint
ruff check optimizer/ tests/

# Auto-fix lint issues
ruff check optimizer/ tests/ --fix

# Format
ruff format optimizer/ tests/

# Type check
mypy optimizer/
```

Or use the Makefile:

```bash
make lint        # ruff check + format check
make typecheck   # mypy
make test        # pytest with coverage
make all         # lint + typecheck + test
```

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

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass: `make all`
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Submit a pull request

### PR Checklist

- [ ] Tests added/updated for new functionality
- [ ] `ruff check` passes with no errors
- [ ] `ruff format --check` passes
- [ ] `mypy optimizer/` passes
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

1. Update the version in `pyproject.toml`
2. Move `[Unreleased]` entries in `CHANGELOG.md` to the new version section
3. Commit: `git commit -m "release: v0.x.0"`
4. Tag: `git tag v0.x.0`
5. Push: `git push origin main --tags`
6. The release workflow will automatically publish to PyPI

## Questions?

Open a [GitHub Discussion](https://github.com/SilvioBaratto/optimizer/discussions) or file an [issue](https://github.com/SilvioBaratto/optimizer/issues).
