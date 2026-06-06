# `core/`

`core/` holds app-wide singletons (services) and infrastructure (guards,
interceptors, error-handling, config) plus app-wide DTO models. **No UI. No
barrels.**

## Dependency rule

- `core/` and `shared/` never import a feature (`pages/`).
- `core/` imports only `core/models` (plus shared UI-infra such as
  `shared/notification`).

## Target structure

| Subfolder | Holds |
|---|---|
| `services/` | App-wide singleton services (`providedIn: 'root'`). |
| `guards/` | Route guards. |
| `interceptors/` | HTTP interceptors. |
| `error-handling/` | Global error handler. |
| `config/` | App-wide config / feature flags. |
| `models/` | App-wide / cross-feature DTO models. |

Subfolders are created by the first `git mv` of each Cycle 1 move issue
(infra → services → models); empty folders are intentionally not committed
(git does not track them, and the project forbids `.gitkeep` and barrel
`index.ts` files).
