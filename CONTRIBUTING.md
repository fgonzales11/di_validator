# Maintaining DI Validator

## Code map

| Area | Location | Responsibility |
| --- | --- | --- |
| Application entry point | `di_validator/api.py` | App lifecycle, access middleware, error responses, router registration, static files |
| Workflow endpoints | `di_validator/routes/` | Dataset/import, label/inventory, experiment, event, job, and workspace HTTP contracts |
| Specialized endpoints | `di_validator/forecasting/api.py`, `di_validator/faults/api.py`, `di_validator/notebooks.py` | Forecasting, COMTRADE, and notebook HTTP contracts |
| Job lifecycle | `di_validator/jobs.py` | Claiming, cancellation, reruns, completion, failures, and restart recovery |
| Worker processes | `di_validator/worker.py` | Dispatching jobs, exclusive worker ownership, child processes, and termination |
| Persistence | `di_validator/store.py`, `di_validator/cloud.py` | Workspace paths, catalog objects, queue records, progress, and database connections |
| Analysis and ingestion | `di_validator/ingest.py`, `classification.py`, `events.py`, `query.py`, `forecasting/`, `faults/` | Data validation, analysis, and reproducible result artifacts |
| Frontend pages | `frontend/src/pages/` | Workflow-specific forms, queries, and result views |
| Shared UI | `frontend/src/components/` | UI primitives, modal lifecycle, Plotly charts, virtual tables, and job activity |
| Frontend utilities | `frontend/src/api.ts`, `format.ts`, `hooks/useStored.ts` | HTTP requests/uploads, display formatting, and versioned browser preferences |

`di_validator.api:app` remains the API entry point. Python analysis interfaces described
in the README remain available from their original modules. The API URL prefix is
`/api/v1`; moving endpoints into routers does not change their URLs or schemas.

## Making changes

- Add ordinary endpoints to the relevant workflow router. Register a new router in
  `api.py` when adding a separate workflow. Keep the frontend fallback route last.
- Put analysis logic in domain modules so the worker and Python callers can use it
  without importing the FastAPI app. Validate requests with the existing Pydantic
  contracts in `schemas.py` or the relevant domain package.
- Use `jobs.py` for queue state transitions shared by HTTP and worker code. Use
  `store.connection()` for database access so local and hosted behavior stays aligned.
  The hosted worker relies on its PostgreSQL advisory lease for exclusive ownership.
- Freeze source, label, annotation, and relationship evidence before enqueuing
  analyses. Rerunning a job must retain its saved configuration and snapshots.
- Import frontend components from their defining files. The small `components/ui.tsx`
  module contains shared presentation elements; charts, tables, dialogs, and job
  activity have separate modules. Keep Plotly's dynamic import and page lazy loading.
- Preserve the `di.v1.` local-storage prefix and existing keys when moving UI state.
  Notebook mounting also preserves the browser kernel when changing tabs.

## Formatting and checks

From the repository root:

```powershell
uv sync --frozen
uv run ruff format di_validator tests
uv run ruff format --check di_validator tests
uv run ruff check di_validator tests
uv run pytest
```

From `frontend/`:

```powershell
npm ci
npm run format
npm run format:check
npm run typecheck
npm run build
```

Ruff formats maintained Python application code and tests. Prettier is pinned in
the frontend lockfile and formats `src/`; `.editorconfig` supplies common editor
settings. TypeScript checks also reject unused imports, variables, and parameters.
Formatting source files changes the code fingerprints recorded by new
forecast runs, even when numerical behavior is unchanged.

Python tests use temporary workspaces. The PostgreSQL integration test requires
`DI_TEST_DATABASE_URL` pointing to an isolated test database; it clears that database's
catalog tables. It is skipped when that variable is absent.

Browser tests use `http://127.0.0.1:8765` and require a running API and worker. Run
`npm run test:e2e` from `frontend/`. Use a separate `DI_WORKSPACE` for both services
when testing imports, saved presets, or worker jobs. Some tests need an imported
AMI dataset, a completed classification experiment, synthetic recordings, imported
COMTRADE recordings, or a built JupyterLite site; see the fixtures required by each
file in `frontend/e2e/` and the setup instructions in the README.

`reference/`, `fault-distance/`, and `data/` hold reference or source material.
Generated output and local runtimes live under `runtime/`, `frontend/dist/`, and
`dist/`. Keep those separate from application refactors.
