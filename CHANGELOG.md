# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This section covers the dependency refresh of 2026-11 (`feature/dependency-refresh-2026-11`),
the second pass over the packages that no `pyproject.toml` declares (`feature/bump-transitive-refresh`)
and the dependency updates merged since 0.2.0. Versions are given as they were in 0.2.0 and
as they are now; the 0.2.0 section below keeps the pins as they were when it was written.
The entries under Security affect privacy or configuration and come first.

### Security

- **fund**: the daemon log no longer shows the `HTTP Request: <method> <url>` lines of
  the SDKs that use the `httpx2` stack (`openai`, which also serves OpenRouter and Azure
  OpenAI, `anthropic`, `langsmith` and, from 2.0, `huggingface-hub`). The line carries the
  full URL, so a query string or credentials in a base URL reached the log on every
  model call. `silence_http_request_logs` now also covers `httpx2` and `httpcore2`.
  Log files written before this fix may already contain such lines: review them and
  rotate or delete the ones that do.
- **fund**: `build_chat_model` now refuses to build a chat model for any provider while the
  environment variable `LANGSMITH_GATEWAY` is set to anything but `false`, `0`, `no` or empty.
  The LangChain packages read that variable and, when a model has no explicit endpoint, send
  its prompts and the provider API key to the LangSmith gateway instead of the provider's own
  host. That applies to `openai` (without `OPENAI_BASE_URL`), `anthropic`, `google` and
  `aws`, and, from `langchain-groq` 1.2, to `groq`. `openai` with a base URL, `openrouter`,
  `microsoft`, `ollama`, `nvidia` and `huggingface` do not use it. The variable is unset
  unless somebody sets it, and this repository never sets it, but the check fails closed:
  if you set it on purpose for other tools, unset it for the fund process. The error does not
  repeat the value, because a gateway URL can carry credentials (REQ-FUND-027;
  `docs/integrations.md`, `.env.example`).
- **fund**: usage telemetry to NVIDIA is switched off. `langchain-nvidia-ai-endpoints` 1.5.0
  sends hourly, content-free usage totals (request, token and error counts, latency buckets,
  connector and model family) to NVIDIA by default when it talks to the hosted API; a
  self-hosted NIM sends nothing. The `nvidia` builder now passes
  `usage_telemetry_enabled=False`, which also wins over the SDK's own
  `NVIDIA_USAGE_TELEMETRY_ENABLED` variable (REQ-FUND-026).
- **deps**: `certifi` 2026.10.23 -> 2026.11.20 removes five root certificates from the bundle
  (Buypass Class 2 and Class 3, QuoVadis Root CA 1 G3 and 3 G3, Certigna Root CA). A host
  whose chain ends in one of them fails TLS verification until it moves to a root that is
  still included. `portopt setup --corp-ca` still adds the operating-system roots on top of
  certifi. `pydantic-settings` 2.16.3 no longer follows symlinks that leave a `secrets_dir`
  (this project sets none), and the `psycopg2-binary` 2.9.13 wheel bundles libpq 17.11 and
  OpenSSL 3.5.8 and rejects over-long int64 array elements instead of overflowing a buffer.
- **fund**: no new telemetry reaches the daemon from the SDK moves, and tests now pin that.
  `anthropic` 2.x can emit OpenTelemetry spans and trace headers, but only through a tracer
  provider the process registers; the fund registers none and the lock holds only
  `opentelemetry-api`, so nothing is recorded or sent (`ANTHROPIC_OPEN_TELEMETRY=false`
  switches it off if one is ever added; REQ-FUND-029). `groq` 1.3 reads `GROQ_CUSTOM_HEADERS`
  and adds those headers to every Groq request; nothing here sets it, and a test pins the exact
  headers a Groq request may carry (REQ-FUND-030). The `google` provider never asks Google's
  auth library for Application Default Credentials (REQ-FUND-032). `hf-xet` 1.7 and later
  report each Hugging Face transfer to the Hub, but the chat path never loads it
  (REQ-FUND-037). A new guard keeps the network-capable development tools (`virtualenv`,
  `nodeenv`, `pip-audit`, `pre-commit`, `pyright`) out of everything that ships
  (`tests/test_lockfile_dev_tooling_isolated.py`).
- **fund**: the portfolio-manager, profiler and Ask agents can no longer write, edit or delete
  a file whose path has a dot-prefixed part (`/.git/config`, `/docs/.x.md`,
  `/code/fund/.hidden.py`). Their one permission rule denies writes to `/**`, and `deepagents`
  matches globs without the dotfile flag, so the rule never matched such a path; in a scripted
  run the Ask agent wrote, edited and deleted them. The writes landed in the virtual backend
  (the staged copies and scratch), never in the checkout. A glob cannot cover a path of any
  depth, so the backend now refuses the three mutating calls itself (REQ-FUND-038). The
  offloaded history and tool results use plain names and are unaffected.
- **fund**: the `huggingface` provider switches the Hugging Face hub's telemetry off. With it
  on, `huggingface-hub` 2.x put the name of the AI agent harness it runs under and the installed
  torch version into the user agent of every request, and fetched the harness list from
  huggingface.co (REQ-FUND-039).
- **fund**: `GROQ_LOG`, `OPENAI_LOG` and `ANTHROPIC_LOG` set to `info` or `debug` no longer
  bring the `HTTP Request: <method> <url>` line back after `silence_http_request_logs` ran.
  Each SDK sets its own logger and the `httpx` logger of its stack when it is first imported,
  which happens after the entry point silenced the loggers, and at DEBUG the SDK logger
  writes the URL, the response headers and the whole request body, which is the prompt.
  `build_chat_model` now undoes that once the model is built (REQ-FUND-028).
- **fund**: the `google` provider is always the Gemini Developer API. An ambient
  `GOOGLE_GENAI_USE_VERTEXAI=true` (or `GOOGLE_GENAI_USE_ENTERPRISE=true`) used to turn the
  model into a Vertex AI client even though an API key was passed, which put Application
  Default Credentials and the metadata server on the request path (REQ-FUND-041).
- **deps**: security fixes in packages that the lock pulls in: `websockets` 17.4 rejects a
  message split into more than 4096 frames (a memory-exhaustion attack; this repository opens
  no WebSocket itself, and `WEBSOCKETS_MAX_FRAGMENTS` tunes the limit), `lxml` 6.1.3 no longer
  resolves external parameter entities by default (reached through yfinance's `read_html`),
  `ormsgpack` 1.13.1 no longer spends memory and time on truncated MessagePack arrays (the
  LangGraph checkpoints), `botocore` 1.44 refuses bearer-token authentication on a plain-HTTP
  endpoint, `markdown-it-py` 4.3.0 and `mdit-py-plugins` 0.7.0 carry security releases (unused
  here), and `wcmatch` 12.0.4 keeps the fix for the ReDoS in 11.x.

### Added

- **portopt-db**: `portopt_db.config.pin_postgres_driver(url)`, public and exported. It
  rewrites a URL that starts with `postgresql://` to `postgresql+psycopg2://` and leaves every
  other URL (an explicit driver, SQLite, an empty string) untouched. The engine, Alembic's
  `env.py` and the live-Postgres test fixtures use it, see SQLAlchemy 2.1 below.
- **docs**: requirement REQ-FUND-018 (the API key or token of the OpenAI family, Groq,
  Anthropic, NVIDIA, Google, AWS and Hugging Face models stays out of `repr`, `str` and the
  serialised form), REQ-FUND-026 (NVIDIA telemetry off), REQ-FUND-027 (the gateway check),
  REQ-FUND-028 (no request line in the log), REQ-FUND-029 and
  REQ-FUND-030 (what an `anthropic` and a `groq` request may carry), REQ-FUND-032 (the `google`
  provider never uses Application Default Credentials) and REQ-FUND-037 (the Hugging Face path
  never loads `hf_xet`), each with tests, in `docs/requirements.md` and `docs/test-plan.md`.
  REQ-FUND-038 (no write to a dot-prefixed path), REQ-FUND-039 (Hugging Face hub telemetry
  off), REQ-FUND-040 (a Groq request has a deadline) and REQ-FUND-041 (`google` ignores the
  Vertex switches) follow, and REQ-FUND-028 now also covers the SDK log variables.
- **README**: a `## Notices` section right after the badges. It states that the output is
  research and education material and not investment advice, that the software places no
  orders with any broker (`fund` records paper orders only), that `fund` sends the data
  you give it to the LLM provider you configure, that market data comes from third parties
  under their own terms, and under which licence the code is published.
  `tests/test_legal_notices.py` guards it.
- **ingestion**: an acknowledgement of Yahoo's terms. Market data comes through
  yfinance, an unofficial Yahoo Finance client, and Yahoo's terms restrict automated
  access and commercial use. `portopt setup` now shows the terms first and asks to
  accept them (default No; declining stops setup before any key or setting is stored),
  `--accept-yahoo-terms` (env `PORTOPT_ACCEPT_YAHOO_TERMS`) accepts them without the
  prompt and is required by a first `--non-interactive` setup, and the answer is kept as
  `yahoo_terms_acknowledged` in `~/.portopt/config.toml`, so a re-run does not ask
  again. `portopt start` passes it to the scheduler container as
  `YAHOO_TERMS_ACKNOWLEDGED` (the new `Settings.yahoo_terms_acknowledged`, default
  false, documented in `ingestion/.env.example`), which `docker-compose.yml` hands
  through from the compose environment, and stops with a message naming
  `portopt setup` when neither the stored config nor its own environment acknowledges
  the terms. The CI `docker-images` job sets the variable for its `up --wait` boot. The
  setup wizard's LLM provider prompt now says that the portfolio data you enter
  (holdings, constraints, optimisation inputs) is sent to the provider you choose,
  and `fund profile --help` says the answers go to the configured LLM provider
  (REQ-ING-026, REQ-ING-027, REQ-ING-028, REQ-ING-029, REQ-ING-030).
- **fund**: the building blocks for presenting four neutral candidate portfolios to
  you, which a rebalance run now uses (see the BREAKING entry under Changed). The
  pure tool `fund.tools.candidates.compute_candidates` fits `hrp`, `mean_risk`,
  `min_risk` and `risk_parity` on one returns panel with the same bounds, position cap,
  budget and moment estimators, always in that alphabetical id order, and reports per
  candidate only annualised volatility, CVaR 95%, largest position, number of positions,
  turnover from current holdings and the constraint check; a method that fails or
  breaches a constraint comes back `feasible=False` with its breaches beside the others.
  That tool's `feasible` is the bounds-only verdict; `fund.tools.candidates` also has
  `constraint_check_from`, `apply_constraint_check` and `recheck_candidates`, which
  re-judge a set by another check and keep each method's own breaches. The schemas are
  in `fund.schemas.candidates` (`CandidateSet`, `Candidate`, `CandidateMetrics`,
  `ConstraintCheck`, `CANDIDATE_IDS`), and none of them has a field that ranks or scores
  a candidate. `AgentRunRepository` gains `append_candidate_set`,
  `latest_candidate_set` (it logs a warning and reads as no set when the stored row is
  corrupt), `append_candidate_selection` and `latest_selected_weights`;
  `fund.audit.repository` gains `current_adviser_choice` and the decision kinds
  `SELECT_CANDIDATE_DECISION` and `EDIT_DECISION`, which the weight edit now writes.
  `fund.observe` gains `candidate_set` and `selected_candidate_id`, and the toolset binds
  a `compute_candidates` tool that applies the risk gate's own check to each candidate
  and persists the set as one `allocator` / `compute_candidates` decision.
  `optimize_portfolio` now shares its returns panel and its mean-risk
  fit (`ReturnsPanel`, `load_returns_panel`, `fit_mean_risk` in `fund.tools.optimize`)
  with the new tool; its results and error messages are unchanged, except that its
  practically unreachable no-weights path now reports
  `ValueError: optimizer produced no weights` instead of the bare text
  (REQ-FUND-043, REQ-FUND-044, REQ-FUND-045, REQ-FUND-046).
- **fund**: a guard that only a human resumes a paused run with approve.
  `fund/tests/unit/hygiene/test_no_unattended_approval.py` parses `fund/src` and lists
  every reference on the resume chain whose decision can be an approve, from
  `Command(resume=...)` through `apply_action`'s dispatch table (`ACTION_HANDLERS`) up
  to `fund approve`, the Decisions keys, the candidate confirm,
  the weight editor and a confirmed Ask action, each with its reason; a new reference
  fails it until a reason is written down. The daemon (`fund.scheduler`, `fund.worker`),
  the agent graphs with their toolsets and subagents, the Ask agent and `fund.tools`
  hold none, and the resume code in `fund.agents.graph`, `graph_hitl` and `profiler`
  only forwards the decision it is handed (REQ-FUND-051).

### Changed

Behaviour changes to know about:

- **fund (BREAKING)**: a rebalance round no longer produces one proposal for you
  to approve. The allocator runs `compute_candidates`, which returns four
  candidate portfolios (`hrp`, `mean_risk`, `min_risk`, `risk_parity`), always listed in
  that id order and never ranked; the agents present and compare them and never choose
  one (their prompts, role descriptions and skills say so, and no longer use
  recommendation language). The `place_orders` gate now carries the whole candidate set
  (its hash, date and universe, every candidate with its label, feasibility, weights,
  metrics, constraint check and breaches, and the `selectable` ids) instead of one
  weight vector and its risk verdict, and a round whose PM reaches the gate with no
  candidate set or no feasible candidate ends `incomplete` (`no_candidate_set` /
  `no_feasible_candidate`) without opening a gate; so does a re-gate after an
  approve, and after a respond, that lands on such a gate. Approving the
  `place_orders` gate now requires naming one candidate: `fund approve <run_id>
  --candidate <id>`, or the `candidate_id` of `fund.actions.apply_action`,
  `resume_run`, `resume_fund` and `FundRun.resume`. There is no default: an approve
  with no id, an unknown id or an infeasible one is refused with a message naming the
  choices and writes nothing, so the gate stays open. Which rule applies depends on
  the gate that is pending, not on the stored set: approving a gate a mandate adds
  ahead of `place_orders` (`compute_candidates`, `risk_check`, `backtest`,
  `get_portfolio_history`) needs no id, records no choice and commits no weights, and
  an id named there is refused. A valid id is re-checked under the mandate in force at
  approve time; that refusal comes after the approve's audit row is appended to the
  caller's session, so nothing is committed and the caller's rollback drops the row.
  On success the passing verdict (`source="candidate_selection"`) and the choice (an
  allocator `optimize_portfolio` row with `hitl_decision.decision="select_candidate"`)
  are recorded, the approve's audit row carries `candidate_id`, `weights`,
  `weights_hash` and `candidate_set_hash`, and paper orders derive only from that
  candidate's weights. `place_orders` places nothing without the user's choice (or a
  weight edit) newer than the run's latest candidate set, so an agent can never place
  a candidate; `reject`, `respond` and `edit` work as before, `edit` approves without a
  candidate id, and an id on any other decision or on a profile gate is refused.
  `risk_check` and `backtest` now take a `candidate_id` and record no verdict, and the
  general-purpose subagent gets an explicit empty tool list, so it can no longer reach
  `place_orders`. `optimize_portfolio` is no longer an agent tool (it leaves
  `fund.tools.__all__`, which gains `compute_candidates`; the function stays in
  `fund.tools.optimize` for the readouts), and `build_gate_weights_middleware` is now
  `build_gate_candidates_middleware`. New public names: the module
  `fund.agents.graph_candidates`, `AgentRunRepository.adviser_choice_in_force`,
  `fund.observe.is_profiler_run` and `fund.agents.toolsets.describe_breaches`. The TUI
  approves a rebalance gate through the Decisions view's candidate table and its
  confirm, and Ask refuses to (see the next entry). **Upgrading**: runs left paused at
  their gate from before this change have no candidate set and cannot be approved;
  reject them and run a fresh round. A mandate whose `hitl_gates` names
  `optimize_portfolio` loses that gate silently, because no agent binds the tool any
  more: name `compute_candidates` instead. When an approve's transaction is lost after
  `place_orders` ran (a consumed gate), the retry must name the candidate again and the
  replacement ticket is placed from the one it names, which may differ from the first
  attempt's; the PM's transcript still shows the first attempt (REQ-FUND-047,
  REQ-FUND-048, REQ-FUND-049, REQ-FUND-050, REQ-FUND-051, REQ-FUND-052).
- **fund**: you choose a candidate in `fund-tui` and see the candidates in
  `fund report`. The Decisions view lists a rebalance gate's four candidates in a table
  (`#decisions-candidates`), in id order and never sorted, each row with the same
  columns (label, feasible or infeasible with its breaches, volatility, CVaR 95%,
  largest position, number of positions, turnover, constraint check, every status a
  dot and a word). Nothing is selected for you and no row is highlighted until
  the cursor moves into the table. `↵` on a feasible row selects it (an infeasible row
  cannot be selected), and `a` then opens a confirm naming the candidate's id, label,
  weights and measures; only Confirm sends the approve, with that id. `a` without a
  selection is refused with a status line, so is an approve of a gate whose stored
  candidate set differs from the one the gate was opened with, and the selection is
  cleared whenever the selected run, the gate id or the candidate set changes. At a
  gate other than `place_orders` (a profile gate, a gate a mandate adds ahead of it)
  `a` approves at once without a candidate, as before. `e` opens the weight editor on
  the selected candidate's weights. Ask now refuses to approve a rebalance gate before
  any confirm and points to Decisions; its agent is told to describe the candidates
  neutrally, in id order, and never to recommend or rank one. `fund report` lists every
  candidate of the run's set in id order with the same columns, then the candidate
  you chose once a choice is in force. The Deliberation view reads out the
  `compute_candidates` step (every candidate with its measures and weights) and shows
  the mean-risk frontier replay only for a `mean_risk` choice, labelled as such; for
  another candidate or an edit it is hidden, and a replay of a choice uses the
  constraint set its candidate set recorded (a run from before candidates existed is
  replayed as before, under the replayed row's own constraint set, none for a
  weight edit). A gate whose detail or candidate set cannot be read shows a one-line
  reason and the error goes to the log. The gate text no longer says an approve
  places the audited proposal (a gate with no candidate set says an approve is
  refused), and `fund.tui.views` no longer exports `_pm_weights_differ` (REQ-FUND-053,
  REQ-FUND-054).
- **fund**: the interface, the agents' prompts and skills and the docs no longer use
  advisory, suitability or mandate wording, so nothing the fund shows you or tells its
  agents presents it as investment advice, a suitability assessment or portfolio
  management: the optimizer computes candidate portfolios, you choose, and orders are paper
  tickets only. The person running the tool is "you" or "the user", never "the adviser"
  or "the client". Visible renames: the onboarding form "MiFID suitability
  questionnaire" is now "Optimisation inputs", and its ESG list is labelled "Sectors to
  exclude (optional)" (none ticked is valid); the TUI view "Mandate" is now
  "Constraints" (rail `Co`), with the sections RUN SETTINGS, RISK LIMITS (from your
  inputs), HITL GATES and SAVED INSTRUCTIONS, the gate box "Save inputs (always on)"
  and the keys "save constraints", "override risk limit", "remove instruction" and
  "steer the run"; an out-of-band override reads "⚠ outside the risk-limit band";
  `fund report` prints `candidates (listed by id; you choose):`, `chosen by you: <id>`
  (was `chosen by the adviser: <id>`) and `current settings:` (was `current
  mandate:`); `fund mandate set` prints `settings saved for portfolio <id>`, and
  `fund mandate show` and `fund run` say `no settings for portfolio <id>`; the profile
  gate reads "Save the optimisation inputs for portfolio <id> … Approve to save these
  inputs, or reject to discard them."; `place_orders` without a choice is refused with
  "no candidate has been chosen for this run; the user chooses one at the place_orders
  gate". A steer answered at the gate now reaches the orchestrator opening with
  `USER DIRECTIVE` (was `ADVISER DIRECTIVE`), and the run instruction labels the
  portfolio line `Settings:` (was `Mandate:`). The orchestrator, economist, allocator,
  risk and executor prompts describe roles that compute, check and present scenarios
  (the economist's views are "background context", no longer "advisory"), and the
  profiler prompt records inputs instead of conducting a MiFID suitability assessment;
  the numerics, the tools, the procedure and the rule that no agent chooses a candidate
  are unchanged. Identifiers, stored data, JSON keys, enum values, the skill folder
  `mifid-profiling` and the CLI commands and flags (`fund mandate set`, `fund profile`,
  `--candidate`) are unchanged. `fund/tests/unit/test_no_advice_language.py` fails on
  the old phrases and on "adviser" in any user-visible string (REQ-FUND-052).
- **fund**: the comments and docstrings of `fund` (the source the Ask agent stages as
  `/code` and quotes to you) no longer call the person running the tool by the old
  role title: they say "the user", and the `SuitabilityBreachError` docstring says
  "hard limit conflict". Prose only, no identifier or behaviour changes; the guard
  `fund/tests/unit/test_no_advice_language.py` now scans those comments and docstrings
  too, and `portopt_db`'s (REQ-FUND-052).
- **fund**: excluding sectors on ESG grounds is optional, and an empty list is a
  complete answer. Inputs with no sector excluded no longer list `esg.exclusions` or
  `esg.look_through_tolerance` in `MiFIDAnswers.unanswered` (the schema drops them,
  whatever the extraction model or a stored record put there), so the profile gate shows
  no `unanswered_input:esg.exclusions` flag. The extraction prompt and the ESG field
  descriptions no longer call an empty list a conservative value: with no sector
  excluded they tell the model to list neither ESG datum as unanswered. A non-empty
  exclusion list the model marks as unstated is still flagged, and excluding every
  sector still blocks the profile (REQ-FUND-055).
- **fund**: the last regulated-service wording is gone. The `portopt-fund` package
  summary now reads "Bridge between LLM agents and the optimizer: the optimizer computes
  candidate portfolios, you choose; paper tickets only". The docstrings and comments the
  Ask agent reads in `fund` and `portopt_db` no longer describe the profiler, the inputs
  it saves, the risk gate or the audit trail as a regulated, suitability-assessment or
  record-keeping service (identifiers, the `mifid_profiles` table and stored data are
  unchanged), the setup wizard says "optimisation inputs" instead of "questionnaire
  answers", and the README tagline calls `fund` a bridge.
  `fund/tests/unit/test_no_advice_language.py` now also scans every package
  description, the `portopt_db` source and this section of the changelog. Two guards
  are stricter: `fund/tests/unit/hygiene/test_candidate_selection_writers.py` also
  limits who records a risk verdict, who writes the weight-edit kind and who writes a
  raw choice or verdict row (REQ-FUND-051), and the new
  `tests/test_no_ticker_isin_lookup.py` fails on any `.isin` read that is not a method
  call, any `get_isin()` call and any `getattr(..., "isin")` in `ingestion/app` or
  `fund/src`, except an `isin` column read on the instruments model or row it lists, so
  a Ticker from `yf.Ticker(...)` or from the repo's own factories is caught under any
  other name (REQ-OPS-005). The README Notices now say that the news step downloads
  articles from the publishers' sites under their terms, and tests pin the
  LLM-provider, third-party-terms and no-hosted-service statements. The root and
  `portopt-db` READMEs give the CI downgrade window as seven revisions, the root README
  tells upgraders to delete files an earlier version left in `./secrets/`, and the
  project skills warn that `Ticker.isin` is never called and name the current Alembic
  head.
- **ingestion (BREAKING)**: the ingestion daemon (`python -m app.worker`, the
  `scheduler` container) now refuses to start until Yahoo's terms are acknowledged: the
  worker logs why and exits with status 2 before it serves metrics, opens its database
  connection or calls Yahoo, and the container restarts into the same refusal. The
  container's entrypoint still waits for the database and applies the migrations
  first; only then does the worker refuse. `portopt start` checks the acknowledgement
  before it touches Docker and stops with a message naming `portopt setup` when it is
  missing. **Upgrading**: an existing install must re-run `portopt setup` once (it asks
  for the acknowledgement and keeps everything else) and then `portopt start`, or set
  `YAHOO_TERMS_ACKNOWLEDGED=1` for the daemon: in the environment of
  `portopt start`, in the shell or the root `.env` for a plain `docker compose up`, in
  the shell or `ingestion/.env` for a daemon run on the host. A script that runs
  `portopt setup --non-interactive` for a first install must add
  `--accept-yahoo-terms`. Manual one-shot commands (`python -m app.cli daily`,
  `docker compose exec scheduler python -m app.cli ...`) are unaffected.
- **ingestion**: the universe builder left the `trading212` package it was named after when
  Trading 212 sourced the universe. `UniverseBuilder`, `UniverseBuilderConfig`,
  `classify_instrument`, the asset-class enums, the protocols and the ETF filters now import
  from `app.services.universe.builder`, and `UniverseBuildRequest` from
  `app.schemas.universe.build_request`. The builder's source protocol `Trading212ApiClient`
  is now `UniverseSourceClient`. No behaviour change; code importing the old paths must
  follow.
- **licence**: the README and the packages now call the licence what it is, source-available
  under PolyForm Noncommercial 1.0.0 (not an OSI-approved licence). `ingestion`, `fund` and
  `portopt-db` declare `license` and the `License :: Other/Proprietary License` classifier in
  their `pyproject.toml`, as the library already did, so a built wheel of any of the four
  carries the same terms. No version or dependency changes.
- **pandas 3.0.6 -> 3.1.1** in the library, `portopt-db` and `ingestion` (exact pins). The
  library adds explicit casts where pandas 3.1's inline typing widened a result type. The one
  casualty is the Tel Aviv trading calendar, see "Known / not done".
- **portopt-db, ingestion, fund**: SQLAlchemy 2.1 makes psycopg 3 the default driver for a
  bare `postgresql://` URL. This project declares, tunes and tests against psycopg2 only
  (connection keepalives, the circuit breaker's error classification), so the engine and
  Alembic now keep psycopg2 through `pin_postgres_driver`. Existing `DATABASE_URL` values do
  not need to change. A URL that names its own driver is respected as written.
- **library**: skfolio 1.14: `prices_to_returns(fill_nan=False)` now keeps the rows after a
  delisting as NaN returns where 1.12 dropped them (`ReturnsConfig.fill_nan` is documented
  accordingly); `OnlineGridSearch.best_estimator_` ends on the final weights; views with a
  negative correlation now fit. Seeded optimizer runs were bit-identical on about 75 probes.
- **ingestion**: yfinance 1.8.0 reads `Ticker.news` from Yahoo's GraphQL news API again
  (nested articles). The Yahoo search fallback stays for an empty feed and for yfinance's
  "Yahoo is down" error, and the news readers accept both article shapes. The release may
  send one extra 2-year hourly request while it repairs prices, outside the rate limiter,
  and its level-shift repair can change stored prices for a few tickers. It also caches an
  empty feed answer per `Ticker` object, so a transient feed fault means search-only news
  for that symbol while the object lives. `multitasking` is no longer installed, and
  `curl-cffi` moves to 0.17.0.
- **ingestion**: typer 0.29: `--help` shows a `<bool>` or `<list[str]>` column for each
  option, validation errors are worded differently and typer now depends on pydantic. Exit
  codes are unchanged (2 for a usage error).
- **ingestion**: exchange-calendars 4.13.3 corrects session data: Hong Kong has two fewer
  sessions in 2023 (the weather closures of 1 and 8 September) and Korea one fewer in 2026, so
  the history-length guard expects slightly fewer sessions for those two venues.
- **fund**: Google SDK 2.x -> 3.x (`google-genai` 2.23.0 -> 3.6.1, pulled by
  `langchain-google-genai` 4.5): the SDK's default API version is now `v1`, which
  `langchain-google-genai` pins back to `v1beta` so requests are unchanged. TLS is no longer
  verified with certifi: the SDKs on the `httpx2` stack (`google-genai`, `anthropic`, `openai`
  and `huggingface-hub` 2.x; see the Security entry on request lines) use the operating-system
  trust store on macOS and Windows and, since `httpx2` 2.15, the system CA files on Linux
  (OpenSSL's default location, for example `/etc/ssl/certs`, and the usual bundle paths).
  `SSL_CERT_FILE` or `SSL_CERT_DIR` still wins. A host behind a corporate CA may need
  `SSL_CERT_FILE` at runtime.
- **fund**: the `anthropic` builder no longer sends `temperature=0` to `claude-mythos-5-2` or
  `claude-sonnet-5-5`: `ChatAnthropic` refuses any temperature but 1 for those models, as it
  already did for `claude-fable-5*`. With `langchain-anthropic` 1.7.13, `LLM_PROVIDER=anthropic`
  and `claude-sonnet-5-5` failed every request before this change, and so did the fallback
  model that goes through the same builder.
- **fund**: `anthropic` 1.6.0 -> 2.14.0 (through `langchain-anthropic`) sends the API key as
  `Authorization: Bearer` instead of `X-Api-Key`; the API accepts both. A proxy, gateway or
  log filter that matches on `X-Api-Key` has to read `Authorization`. The request keeps the
  `anthropic-version` of `2023-06-01` that `langchain-anthropic` sets, which is what lets the
  `temperature` setting work; a wire test fails if that changes.
- **fund**: AWS Bedrock calls retry less: `botocore` 1.44 defaults to its standard retry
  mode, so a throttled or failing call is tried 3 times instead of 5, with shorter waits (about
  3 s at most, was about 15 s). `AWS_MAX_ATTEMPTS` and `AWS_RETRY_MODE` change it.
- **fund**: a Groq request now has a 180 second deadline, the same as the other providers
  that set one. `ChatGroq` had none, so a call that hung blocked the run for good.
- **fund**: an SDK error says less. `openai` 3.38 puts only the server's message in
  `str(error)` (the body stays on `error.body`), so an alert that matches the text
  `Error code: 4xx - {...}` now sees `Error code: 4xx - <message>`, and `openai` 3.59 strips the
  `Authorization` header from the request attached to the error.
- **fund**: the Ask agent's `glob` and `grep` listings can only shrink with `wcmatch` 12:
  `*.env` and `*.*` no longer list dotfiles and `**/` no longer lists regular files. Over
  182,546 pattern and path cases nothing started matching, the permission checks decided
  1,416 cases identically, and a scripted Ask run gave the same write, edit and delete
  outcomes.
- **fund**: LangGraph checkpoints written by the earlier versions read back unchanged under
  `ormsgpack` 1.13.1 and `langgraph-checkpoint` 4.3.2, and the reverse also holds, so a
  rollback can read what the new versions wrote (checked on the real Postgres saver and store).
- **library**: the solver stack moved (`cvxpy-base` 1.9.3 with `sparsediffpy` 0.6.1, `joblib`
  1.6.0, `threadpoolctl` 3.8.0, `tzdata` and `pytz` 2026.6) and results did not: 122 seeded
  optimizer problems (mean-risk with 14 risk measures, risk budgeting, distributionally robust
  CVaR, maximum diversification, robust uncertainty sets, the benchmark tracker) gave weights
  identical to the last digit. `joblib` 1.6.0 needs `cloudpickle`, a new package in the lock.
  The time-zone data changes Cuba, Manitoba, Inuvik and some history before 1967; no zone
  that the library, the scheduler or an exchange calendar uses is affected.
- **fund**: langchain-core 1.8.5 is used by every provider package. The `gpt-5.3-chat-latest`
  profile no longer states `max_input_tokens`, and the NVIDIA profile table in
  `langchain-nvidia-ai-endpoints` 1.5 dropped 54 earlier ids, so deepagents' summarization
  falls back to its fixed thresholds (170,000 tokens for the OpenAI model) for models
  without a profile; `FUND_SUMMARY_TOKEN_THRESHOLD` overrides it.
- **dev tools**: mypy 2.4 uses its native parser by default and flagged one duplicate
  `type: ignore` in `ingestion/app/repositories/jobs/background_job_repository.py`, now
  removed. pyyaml 6.0.4 changes the precedence of YAML merge keys; the repository has none.
  `plotly` 7.2.0 embeds plotly.js 4.1.1 in the HTML that `save_figure_html` writes, and its
  wheel is about 0.6 MB larger. `coverage` 7.17.5 measures the same percentages on all four
  suites, `click` 8.5.1 leaves the `--help` screens of both CLIs as they were, `wcwidth` 0.9
  ships an optional compiled extension (`WCWIDTH_PYTHON=1` turns it off) and `virtualenv`
  21.15.9 and later also asks the package indexes in pip's configuration about yanked files
  on its periodic update (`VIRTUALENV_NO_PERIODIC_UPDATE=1` stops all contact).

Version moves, from 0.2.0 to now:

| Member | Package | From | To |
|---|---|---|---|
| library (`portopt-core`) | skfolio | 1.12.5 | 1.14.6 |
| library | plotly | 7.0.0 | 7.2.0 |
| library, ingestion | numpy | 2.5.4 | 2.5.5 |
| ingestion, portopt-db | pydantic | 2.14.1 | 2.14.4 |
| ingestion | python-dotenv | 1.2.3 | 1.2.4 |
| ingestion | pydantic-settings | 2.15.0 | 2.16.3 |
| ingestion | yfinance | 1.7.0 | 1.8.0 |
| ingestion | typer | 0.27.1 | 0.29.3 |
| ingestion | certifi | 2026.10.23 | 2026.11.20 |
| ingestion | exchange_calendars | 4.13.2 | 4.13.3 |
| ingestion, portopt-db | SQLAlchemy | 2.0.52 | 2.1.9 |
| portopt-db | alembic | 1.19.1 | 1.20.1 |
| portopt-db | psycopg2-binary | 2.9.12 | 2.9.13 |
| fund | deepagents | 0.7.35 | 0.7.42 |
| fund | langgraph (and langgraph-prebuilt) | 1.2.20 (1.1.0) | 1.2.24 (1.1.3) |
| fund | langgraph-checkpoint-postgres | 3.2.0 | 3.2.1 |
| fund | psycopg (and psycopg-binary) | 3.3.5 | 3.3.9 |
| fund | langchain-ollama (ollama client) | 1.1.0 (0.6.2) | 1.2.0 (0.6.3) |
| fund | langchain-huggingface | 1.2.3 | 1.3.0 |
| fund | huggingface-hub (tokenizers) | 1.33.0 (0.23.2) | 2.4.0 (0.23.3) |
| fund | langchain-anthropic | 1.7.10 | 1.7.13 |
| fund | langchain-groq | 1.1.3 | 1.2.1 |
| fund | langchain-nvidia-ai-endpoints | 1.4.3 | 1.5.0 |
| fund | langchain-openai | 1.6.5 | 1.8.5 |
| fund | langchain-google-genai (google-genai) | 4.4.3 (2.23.0) | 4.5.3 (3.6.1) |
| fund | langchain-aws | 1.9.3 | 1.10.0 |
| fund, ingestion (tests) | pytest-asyncio | 1.4.1 | 1.5.0 |
| dev | ruff (also the pre-commit hook `rev`) | 0.17.3 | 0.17.6 |
| dev | pre-commit | 4.6.2 | 4.7.0 |
| dev | mypy | 2.3.1 | 2.4.1 |
| dev | hypothesis | 6.168.5 | 6.170.17 |
| dev | pyyaml | 6.0.3 | 6.0.4 |
| CI | astral-sh/setup-uv | v10.3.3 | v10.4.0 |

The lock also moves, without a declaration of its own: langchain 1.4.9 -> 1.5.2, langchain-core
1.7.7 -> 1.8.5, langchain-protocol 0.0.20 -> 0.0.22, langsmith 0.14.17 -> 0.16.2,
curl-cffi 0.16.2 -> 0.17.0; `greenlet` and `multitasking` are no longer installed.
The comments in `pyproject.toml` and `packages/portopt-db/pyproject.toml` now name
`numpy==2.5.5`. deepagents 0.7.41 and later need langgraph 1.2.24 or newer, which is why
the two move together; the Ask agent, summarization, backend, skills and containment tests
pass unchanged on 0.7.42.

The second pass moved every other package in the lock to its newest release. None has a
declaration of its own, and `uv lock --upgrade` now has nothing left to move:

- Provider SDKs: anthropic 1.6.0 -> 2.14.0 (adds `opentelemetry-api` 1.45.2), groq 0.37.1 -> 1.7.0,
  openai 3.19.2 -> 3.60.0, boto3 and botocore 1.43.101 -> 1.44.12, s3transfer 0.19.2 -> 0.19.3,
  google-auth 2.58.0 -> 2.63.1.
- Other majors: websockets 16.1.1 -> 17.4, multidict 6.9.1 -> 7.2.1, wcmatch 11.2 -> 12.0.4;
  uuid-utils 0.17.1 -> 0.19.0, jsonpatch 1.33 -> 1.48, jsonpointer 3.1.1 -> 3.12.
- HTTP and runtime: httpx2 and httpcore2 2.13.0 -> 2.16.2, anyio 4.14.2 -> 4.16.0, idna 3.19 -> 3.20,
  charset-normalizer 3.5.1 -> 3.5.3, truststore 0.10.4 -> 0.10.5, orjson 3.12.0 -> 3.13.0, ormsgpack
  1.12.2 -> 1.13.1, jiter 0.17.0 -> 0.19.0, lxml 6.1.2 -> 6.1.3, regex 2026.9.10 -> 2026.11.25,
  soupsieve 4.1 -> 4.1.3, hf-xet 1.6.0 -> 1.8.0, huggingface-hub 2.4.0 -> 2.5.0, langgraph-checkpoint
  4.3.0 -> 4.3.2, langgraph-sdk 0.4.11 -> 0.4.14, annotated-doc 0.0.5 -> 0.0.7, typing-inspection
  0.4.4 -> 0.4.5, toolz 1.1.0 -> 1.2.0.
- Numerics and time zones: cvxpy-base 1.9.2 -> 1.9.3, sparsediffpy 0.3.0 -> 0.6.1, joblib 1.5.3 -> 1.6.0
  (adds cloudpickle 3.1.3), threadpoolctl 3.6.0 -> 3.8.0, narwhals 2.25.0 -> 2.29.0, xxhash 4.0.1 -> 4.1.0,
  pycparser 3.0 -> 3.12, pyparsing 3.3.2 -> 3.3.3, protobuf 7.36.0 -> 7.36.2, pytz 2026.3.post1 -> 2026.6,
  tzdata 2026.3 -> 2026.6, tzlocal 5.4.4 -> 5.4.5.
- Tooling and text: coverage 7.15.4 -> 7.17.5, pluggy 1.6.0 -> 1.7.0, iniconfig 2.3.0 -> 2.3.1,
  cyclonedx-python-lib 11.12.0 -> 11.13.0, py-serializable 2.1.0 -> 2.2.0, pip-api 0.0.34 -> 0.0.35,
  pip 26.2.1 -> 26.3, cachecontrol 0.14.4 -> 0.15.0, packaging 26.3 -> 26.4, virtualenv 21.14.13 ->
  21.15.13, filelock 4.2.4 -> 4.3.1, platformdirs 4.11.4 -> 4.14.8, distlib 0.4.3 -> 0.4.4,
  python-discovery 1.6.4 -> 1.9.7, identify 2.6.19 -> 2.6.20, nodeenv 1.10.0 -> 1.12.0, click 8.5.0 ->
  8.5.1, markdown-it-py 4.2.0 -> 4.3.0, mdit-py-plugins 0.6.1 -> 0.7.0, mako 1.4.1 -> 1.4.3,
  markupsafe 3.0.3 -> 3.0.4, tomli 2.4.1 -> 2.5.0, wcwidth 0.8.2 -> 0.9.5, peewee 4.4.0 -> 4.5.3.

### Removed

- **The `docs/` folder is no longer published.** The architecture and decision records
  (`ARCHITECTURE.md`, `DECISIONS.md`, `SPEC-RECONCILIATION.md`, `VALIDATION.md`,
  `WRITE-PATH-AUDIT.md`), the ADRs, `integrations.md`, `nfr.md`, `test-plan.md` and the
  requirements registry `requirements.md` left version control. The maintainer keeps them in a
  git-ignored local folder, and earlier commits still hold them. The operator knobs that
  `integrations.md` listed with their defaults are defined in `ingestion/app/config.py` and
  `fund/src/fund/config.py` and named in this file.
- **The requirement-traceability gate**, which only worked with that registry: the
  `Requirement traceability` step and its `traceability` artifact in the `lint` job,
  `scripts/ci/traceability.py` (also the pytest plugin the four suites loaded with
  `-p traceability`, together with their `pythonpath` entries for `scripts/ci`) and the
  guard tests of the removed files (`tests/ci/test_traceability.py`,
  `tests/test_adr_format.py`, `tests/test_integration_docs.py`,
  `tests/test_nfr_and_test_plan_docs.py`). CI no longer fails a test that cites an unknown
  `REQ-` id or a requirement with no test. The `criterion` and `defect` markers stay
  registered, because tests use them, but they are now plain labels that nothing verifies.
  CONTRIBUTING.md loses its "Test traceability" section and its two ADR links.
- **fund**: the Trading 212 broker scope. `PortfolioMandate.tradeable_only`, the
  `fund mandate set --tradeable-only` / `--no-tradeable-only` options, the "tradeable"
  line of `fund mandate show`, the Mandate view's "Tradeable on Trading 212 only" box, the
  `tradeable_only` parameter of `resolve_eligibility` and `list_universe`, the
  `not_t212_tradeable` bar reason, `Eligibility.has_allowlist`, `Eligibility.untradeable`
  and `Eligibility.has_broker_bar`, the mandate-save warnings about names the broker lacks,
  and the broker paragraphs of the PM seed are gone. A mandate stored with a
  `tradeable_only` key still loads and the key has no effect, so a portfolio that used the
  broker scope loses that restriction silently and may now propose names the broker does
  not carry; list the instruments you want in `--eligible` instead. **portopt-db**: the
  `UniverseReader` port no longer declares `get_t212_untradeable_tickers` and
  `get_t212_unevaluated_tickers` (the repository methods go with the columns, below).
- **ingestion**: the Trading 212 integration. The Trading 212 client with its ticker mapper
  and cache, the `t212_annotate` step that followed `universe_build` (and its job type),
  `portopt add-instrument --t212-ticker`, the Trading 212 prompts of `portopt setup` with
  its `--t212-key` / `--t212-secret` options and live key check, the `TRADING_212_API_KEY`,
  `TRADING_212_SECRET_KEY`, `TRADING_212_MODE`, `BREAKER_TRADING_212_MAX_ATTEMPTS`,
  `TRADING_212_REQUEST_ATTEMPTS` and `TRADING_212_REQUEST_TIMEOUT_SECONDS` settings, the
  Trading 212 circuit breaker, the `trading_212_api_key` / `trading_212_secret_key` compose
  secrets, the scheduler's `.env.scheduler` env file (it only carried `TRADING_212_MODE`)
  and `ingestion/scripts/debug_universe_scale.py`. The weekly `universe_build` job now runs
  the universe step alone. Nothing writes `instruments.t212_ticker`,
  `instruments.is_t212_tradeable` or `exchanges.t212_id` any more; the `portopt-db` entry
  below drops the columns. The per-ticker ISIN lookup the annotation step made is gone
  too. **Upgrading**: nothing is needed for the stack to start. A secret store written by
  an older `portopt setup` may still hold the two Trading 212 keys; setup and
  `portopt start` ignore them and render neither. The `./secrets/trading_212_api_key`,
  `./secrets/trading_212_secret_key` and `./.env.scheduler` files an older `portopt start`
  rendered are no longer used, and `portopt stop --all` no longer deletes them: delete them
  at will.
- **portopt-db**: the three Trading 212 columns `exchanges.t212_id`,
  `instruments.t212_ticker` and `instruments.is_t212_tradeable`. The new Alembic revision
  `2266e24c509f` (after `b8d0f2a4c6e1`) drops them, and the ORM attributes and the
  `UniverseRepository` methods `set_t212_ticker`, `set_t212_tradeable`,
  `get_t212_tradeable_tickers`, `get_t212_untradeable_tickers` and
  `get_t212_unevaluated_tickers` are gone with them. `save_exchange` ignores any key but
  `name`. **Upgrading**: the migration deletes the data in the three columns, and the
  scheduler and fund containers run `alembic upgrade head` when they start, so pulling this
  version and restarting the stack applies it. Take the database dump **before** you pull
  the new version and restart. `alembic downgrade -1` re-adds the columns as nullable and
  empty; the dropped values are not restored. A repository test
  (`tests/test_removed_integrations.py`) now fails when a tracked file names the removed
  integration, except this changelog, the Alembic history, the tests of those migrations
  and the three tests that assert the removal itself (the mandate, the setup wizard and
  the universe builder import checks).
- **ingestion**, **portopt-db**: the yfinance ISIN lookup against a third-party site.
  `yf.Ticker.isin` is experimental: it searches Business Insider by company name, a site
  whose terms ban scrapers, and the profile step called it once per ticker. Gone are
  `MetadataClient.fetch_isin` and its protocol entry, the ISIN injection into the profile
  write and the `isin:` error entry it could add to a sweep, the ISIN backfill that ran
  after each bulk yfinance sweep, `UniverseRepository.backfill_isin_from_profiles` and
  `set_isin_if_missing`, and the helpers `is_trusted_isin` and `has_exchange_suffix`.
  `has_valid_checksum` stays, in `app.services.market_data.yfinance.ticker.isin_checksum`.
  A new profile carries no ISIN and the screener supplies none, so an ISIN is now stored
  only when an operator sets it with `portopt add-instrument <TICKER> --isin <ISIN>`; the
  `instruments.isin` and `ticker_profiles.isin` columns stay and keep the values they
  hold. **Upgrading**: nothing is needed; an instrument that never had an ISIN stays
  without one until you set it. `tests/test_removed_integrations.py` now also rejects the
  site's name in a tracked file.
- **portopt-db**: the six tables the removed macro scrapers filled, `economic_indicators`,
  `economic_indicator_observations`, `trading_economics_indicators`,
  `trading_economics_observations`, `bond_yields` and `bond_yield_observations`. The new
  Alembic revision `8ed6504aee6a` (the head, after `2266e24c509f`) drops them. The six ORM
  models, the `MacroRegimeRepository` methods that read or wrote them (the economic
  indicator, Trading Economics and bond-yield `upsert_*` and `get_*` methods, plus
  `get_country_summary` and `get_distinct_countries`), their three entries in `APP_TABLES`
  and the `portopt_db.coerce` module (`parse_reference_date`, which only those methods
  called) are gone with them. The FRED observations, the macro news tables and their
  models and methods are unchanged. **Upgrading**: the migration deletes the stored macro
  and bond-yield history, and the scheduler and fund containers run `alembic upgrade head`
  when they start, so pulling this version and restarting the stack applies it. Take the
  database dump **before** you pull the new version and restart. `alembic downgrade -1`
  recreates the six tables empty, with their columns, constraints and indexes; the deleted
  rows are not restored. The deliberate data loss is a recorded deviation from the
  reversible-migration rule, with the reason and the backup step, under "Destructive
  migrations" in `packages/portopt-db/README.md`.

### Removed (BREAKING — ingestion CLI, scheduler and settings)

- **ingestion**: the Il Sole 24 Ore and Trading Economics macro scrapers. Gone are the two
  scrapers and `ParseStructureError`, the Il Sole and Trading Economics parts of
  `MacroRegimeService` (`fetch_fred_series` and `fetch_macro_news` stay), the
  `run_bulk_macro_fetch` function, the `run_macro_step` scheduler step with its `macro_fetch`
  job type, the `python -m app.cli macro` command, the `MacroFetchRequest`,
  `EconomicIndicatorResponse`, `TradingEconomicsIndicatorResponse`, `BondYieldResponse`
  and `CountryMacroSummary` schemas, the two circuit breakers, and the
  `BREAKER_TRADINGECONOMICS_MAX_ATTEMPTS`, `BREAKER_ILSOLE_MAX_ATTEMPTS`,
  `TRADINGECONOMICS_REQUEST_ATTEMPTS`, `ILSOLE_REQUEST_ATTEMPTS`,
  `TRADINGECONOMICS_REQUEST_TIMEOUT_SECONDS` and `ILSOLE_REQUEST_TIMEOUT_SECONDS`
  settings. `daily_pipeline` and `weekly_refetch` run no macro step any more, so macro
  series now come from FRED only; the FRED steps and the macro news fetch are unchanged.
  **Upgrading**: a script or cron entry that runs `python -m app.cli macro` now fails with
  an unknown-command error and must drop the call. A removed setting left in `.env` is
  ignored. The tables the scrapers filled are dropped by the **portopt-db** entry under
  "Removed" above. A `macro_fetch` job row left by a worker that died is no longer
  re-dispatched by the orphan reaper.
  `tests/test_removed_integrations.py` now also rejects both names in a tracked file.

### Known / not done

- **The Tel Aviv (`XTAE`) trading calendar does not build on pandas 3.1.**
  `exchange_calendars` 4.13.3 raises while building it, so for Tel Aviv listings the
  history-length check is skipped (the daemon logs a warning per instrument) until a release
  contains upstream pull request 633. The two tests that cover that venue are strict expected
  failures in `ingestion/tests/_fixtures/upstream_gaps.py`; once `exchange_calendars` is
  fixed they pass, which fails the run, and the entry is deleted there.
- **python-dateutil 4.0.0** is not adopted: `botocore`, reached through `langchain-aws` (the
  `aws` and `all-llm` extras), caps it below 3.0.
- **Releases that another package's cap holds back**: `uuid-utils` 1.x (`langsmith` requires
  below 1.0, and `langchain-core` and `langsmith` still call the 0.x signature),
  `sparsediffpy` 0.7 to 0.9 (`cvxpy-base` 1.9.3 requires below 0.7) and `soupsieve` 4.1.4 (on
  GitHub, not on PyPI).
- **At DEBUG, `botocore` and `websockets` log URLs, headers and credential ids.** The daemon
  logs at INFO and has no DEBUG switch, so that is not reachable today.
- **Test isolation in `portopt-db`**: running its unit tests and `tests/integration` in one
  process with `PORTOPT_TEST_DATABASE_URL` set fails two log-capture tests, because Alembic's
  `env.py` disables the existing loggers. CI runs them separately and passes.
- What only a live provider can confirm was not exercised, because the tests use fakes and
  local stubs: the Anthropic API accepting the key as `Authorization: Bearer` with the
  `2023-06-01` version header, Gemini, Bedrock, Groq and NVIDIA NIM structured output and tool calls, the
  extra yfinance repair request and its effect on stored prices, and the news feed's article
  ids against search ids (a story may appear twice per ticker).

## [0.2.0] - 2026-11-18

> **Upgrading from 0.1.0.**
>
> - **Library users:** the distribution is now `portopt-core` (it was `portopt` in 0.1.0;
>   the import name `optimizer` is unchanged, so uninstall the old `portopt` before you
>   install it). It needs Python 3.12 or newer, pins exact dependency versions, and is
>   licensed under PolyForm Noncommercial 1.0.0 instead of BSD-3-Clause. The `optimizer`
>   console command and `optimizer.pipeline` are gone (compose runs from the primitives;
>   `build_portfolio_pipeline` now lives in `optimizer.pre_selection`), and so are the HMM
>   and DMM moment blending and several other names the subpackages used to export.
>   Several `optimizer.factors` configs validate at construction, and `CPCVConfig` purges
>   and embargoes by default. The two BREAKING sections below list the main breaks: the
>   removed names, the packaging and dependency changes and some behaviour changes. They are
>   not complete. The configs and builders that are still exported also lost or renamed
>   fields, presets and parameters and changed some defaults, and most of those are not
>   itemised: for example `MomentEstimationConfig` takes `ew_mu_half_life` and
>   `ew_cov_half_life` where it took `ew_mu_alpha` and `ew_cov_alpha`, `RiskBudgetingConfig`
>   no longer has `risk_measure`, `DistanceConfig` has `estimator` where it had
>   `distance_type`, and `WalkForwardConfig.purged_size` defaults to 5 where it was 0. Check
>   the signatures you call against the 0.1.0 source and run your code against this
>   release before you rely on it.
> - **Docker users:** dump the database first, because four migrations cannot be
>   downgraded (`d1e2f3a4b5c6`, which drops 17 tables, and three later ones,
>   `a5ac0fa9b0ec`, `a6f1c8d3e2b7` and `b9c8d7e6f5a4`) and `a5ac0fa9b0ec` deletes rows
>   from `trading_economics_indicators` and `trading_economics_observations`. Every
>   compose service now sits behind a profile, so a bare `docker compose up` starts
>   nothing (use `--profile ingestion` or `--profile fund`). The scheduler reads its API
>   keys from `.env`, `.env.scheduler` or the `./secrets/` files, so a key exported only
>   in the shell no longer reaches it. The database (54320) and Adminer (18081) ports are
>   published on `127.0.0.1` only, so other hosts can no longer reach them.
>   `docker-compose.yml` pins the project name to `optimizer`, so a checkout under
>   another directory name needs the steps in the **Fixed** entry on that pin before it
>   upgrades.

> **Many entries below concern the repository's ingestion service (`ingestion/`,
> formerly `api/`) and its tooling.** The itemised library breaking changes are in
> **Removed (BREAKING — library)** and **Changed (BREAKING — library)** immediately
> below; the `fund` bridge, the shared `portopt-db` layer and CI have entries of
> their own further down. The **Added** entries that cite #36 to #53 were already
> listed as unreleased in the 0.1.0 changelog, and some of what they describe (the
> release workflow, the Makefile, the `NullHandler`) is in the 0.1.0 tag too, so they
> are not new in 0.2.0.

### Removed (BREAKING — library)

- **`optimizer.pipeline` deleted entirely.** The library is now a set of
  composable, DB-agnostic primitives with no fixed end-to-end runner. Removed
  public symbols: `run_full_pipeline`, `run_full_pipeline_with_selection`,
  `optimize`, `backtest`, `tune_and_optimize`, `compute_net_backtest_returns`,
  `PortfolioResult`. `build_portfolio_pipeline` (pre-selection → optimiser
  composition) moved to `optimizer.pre_selection`. Opinionated, DB-connected
  orchestration (FX conversion, delisting correction, universe/factor selection,
  rebalancing decisions, persistence) now belongs to the `fund/` bridge,
  keeping `optimizer` DB-agnostic. Compose runs from primitives:
  `build_portfolio_pipeline(optimizer)` + `optimizer.validation.run_cross_val`.
- **The `optimizer` console command and the `cli` package.** 0.1.0 installed an
  `optimizer` command (`cli:app`) and a top-level `cli` package next to the
  `optimizer` import package. The library distribution now installs neither; the
  `portopt` command belongs to the separate ingestion distribution. The `portopt setup`
  wizard puts a launcher that is also named `optimizer` on your `PATH` by default (it
  brings the fund stack up and opens its terminal shell); it is not the removed command.
- **HMM and DMM moment blending.** `optimizer.moments` no longer exports `HMMConfig`,
  `HMMResult`, `HMMBlendedCovariance`, `HMMBlendedMu`, `fit_hmm`, `select_hmm_n_states`,
  `blend_moments_by_regime`, `DMMConfig`, `DMMResult`, `fit_dmm` or `blend_moments_dmm`.
  The `hmmlearn` dependency and the `dmm` extra (`torch`, `pyro-ppl`) went with them.
- **Names `optimizer.optimization` no longer exports:** `RegimeRiskConfig`,
  `build_regime_risk_budgeting`, `compute_regime_budget`, `compute_blended_risk_measure`,
  `build_regime_blended_optimizer`, `RobustConfig`, `bootstrap_covariance_uncertainty`,
  `CovarianceUncertaintyResult`, `ClusteringConfig`, `build_clustering_estimator`,
  `build_distance_estimator`, `DistanceType` and `ExtraRiskMeasureType`. `DistanceConfig`
  and `LinkageMethodType` are importable from `optimizer.distance` and `optimizer.cluster`
  instead (`DistanceConfig` changed shape on the way: `estimator` replaces `distance_type`,
  and `threshold` defaults to `None` instead of 0.5). Where the capability remains it has
  new entry points, not drop-in
  replacements, so read their signatures: `build_regime_blended_mean_risk` and
  `RobustMeanRiskConfig` in `optimizer.optimization`, `build_covariance_uncertainty_set`
  and `build_mu_uncertainty_set` in `optimizer.uncertainty_set`,
  `HierarchicalClusteringConfig` and `build_hierarchical_clustering` in `optimizer.cluster`,
  and `build_distance` in `optimizer.distance`.
- **The `docs` extra** (MkDocs) and the `Documentation` project URL, with the `examples/`
  directory and the MkDocs site they served (see the entry further down).
- **`optimizer.factors` config knobs that no code path reads any more.** Setting
  them implied a protection or a switch, so they are removed and passing them now
  raises `TypeError` (or `ImportError` for the names).
  - `FactorIntegrationConfig.max_weight` — never applied; cap positions with
    `MeanRiskConfig.max_weights`.
  - `FactorConstructionConfig.publication_lag` — nothing reads it, so it gives no
    point-in-time protection. In 0.1.0 its only reader was the removed
    `run_full_pipeline_with_selection`, which used `macro_days` (default 63) to lag
    macro data before regime classification, so lag the macro data yourself before
    `classify_regime`. Shift fundamentals with `align_to_pit(lag_days=...)`, taking
    the lag from the matching `PublicationLagConfig` field, before computing factors.
  - `FactorValidationConfig.fmp_top_pct` / `fmp_bottom_pct` — mimicking-portfolio
    legs are set through the builder's `quantile` argument.
  - `GroupICAggregationConfig`, `ICWeightingMethod`, `ICNegativeFilterPolicy`,
    `GroupICResult` — no group-IC aggregation by t-stat weight or negative-IC
    filter existed.

  `FactorIntegrationConfig.max_weight` and the four group-IC names did not exist in
  0.1.0 (they were added after it and removed again before this release), so only the
  `publication_lag` and `fmp_*` knobs can affect a 0.1.0 user.

### Changed (BREAKING — library)

Some entries below are about things that did not exist in 0.1.0: the
`SectorRegimeBandsConfig` line of the construction-time validation entry, and the
entries on `resolve_sector_bands`, `RegimeThresholdConfig.for_rolling_percentile` and
`MomentEstimationConfig.for_regime_adjusted_ew`, plus the factor exposure band entry
(nothing in 0.1.0 read the `exposure_*_bound` fields) and the `RobustMeanRiskConfig`
entry (its presets
share their names with the `RobustConfig` presets that 0.1.0 had and this release
removes, see above). They were added after 0.1.0 and changed again before this
release, so they record later changes and ask nothing of a 0.1.0 user. Other entries
mix older and newer names, and only the newer ones are later additions:
`PreSelectionConfig.zero_variance_threshold`, `CleaningConfig` and `for_low_volatility`,
`CompositeScoringConfig.ic_decay_halflife`, `implied_annualization_factor`,
`apply_no_trade_band`, `min_sector_weights`, `to_returns`, the `optimizer.online`
searches, `market_weights` and `AvailableAssetViewsPrior` did not exist in 0.1.0, while
`winsorize_threshold`, `ICIR`, `CovEstimatorType.IMPLIED`, `compute_drifted_weights` and
`SyntheticDataConfig` did, so read those entries in full.

- **The library distribution is renamed from `portopt` to `portopt-core`.** The import
  name `optimizer` is unchanged. In this repository the name `portopt` now belongs to
  the ingestion daemon's distribution (`ingestion/`), which the release workflow does
  not publish. Switch a dependency on the library to `portopt-core`; `portopt` gets no
  further library releases. The 0.1.0 `portopt` distribution and `portopt-core` both
  install the `optimizer` package, so uninstall `portopt` before you install `portopt-core`.
- **The Python floor rises from 3.10 to 3.12** (`requires-python = ">=3.12"`). The
  pinned `numpy` and `scipy` need Python 3.12 or newer; CI runs 3.12 and 3.13.
- **Runtime dependencies are exact pins and fewer.** `numpy==2.5.4`, `pandas==3.0.6`,
  `scipy==1.18.1`, `scikit-learn==1.9.1` and `skfolio==1.12.5` replace the 0.1.0 ranges
  (`numpy>=1.24`, `pandas>=2.0`, `scipy>=1.11`, `scikit-learn>=1.6`, `skfolio>=0.5`), so
  installing the library fixes those versions (pandas 3 in particular) in the
  environment. `skfolio` is pinned to the 1.x release 1.12.5, whose API differs from the
  0.x versions its 0.1.0 range allowed (for example the EW estimators take `half_life`
  where they took `alpha`), and that shows through the estimators the builders return.
  `jinja2==3.1.6` is a new runtime dependency, and `hmmlearn` and `arch` are no longer
  declared.
- **The licence changes from BSD-3-Clause to PolyForm Noncommercial 1.0.0** (see
  `LICENSE`).
- **`optimizer.factors` configs now validate at construction and raise
  `ConfigurationError`** for values that previously failed late or produced
  silently wrong output:
  - `FactorConstructionConfig`: every lookback must be `>= 1` and
    `0 <= momentum_skip < momentum_lookback - 1`.
  - `CompositeScoringConfig`: `ic_lookback >= 1`, tier weights finite and
    non-negative with a positive sum.
  - `SelectionConfig`: `target_count >= 1`, quantiles and `sector_tolerance` in
    `[0, 1]`, `exit_quantile <= target_quantile`, `buffer_fraction >= 0`.
  - `SectorRegimeBandsConfig`: repeated `(regime, sector)` rows are rejected, and
    a regime listing all 11 sectors must have caps summing to at least 1.0.
- **`ICIR` composite weighting now honours `ic_lookback` and
  `ic_decay_halflife`**, as `IC_WEIGHTED` already did. With the default
  `ic_lookback=36`, group IC histories longer than 36 periods are trimmed, so
  existing ICIR-weighted composites change; set a larger `ic_lookback` to keep
  the old whole-history behaviour.
- **Factor exposure band is relative to the selected set.** In 0.1.0 nothing applied
  `FactorIntegrationConfig.exposure_lower_bound` / `exposure_upper_bound`.
  `build_factor_integration`, added after 0.1.0, bounds the exposure within them
  around the equal-weight exposure of the selected tickers, not around the universe
  mean (z-score 0), because absolute bounds were infeasible for any top-N selection; a
  full-universe selection is unchanged. `build_factor_exposure_constraints`, which
  0.1.0 already had, still takes absolute bounds.
- **`resolve_sector_bands` honours `SectorRegimeBandsConfig.enable`.** A disabled
  config (`SectorRegimeBandsConfig.disabled()`) now yields no sector constraints
  instead of an 11-sector, 33%-floor set, and an enabled config with no bands is
  filled with the default regime matrix instead of a flat 3%-16% band.
- **`standardize_factor` honours `neutralize_country` on its own.**
  `neutralize_country=True` with `neutralize_sector=False` now demeans within
  each country instead of returning un-neutralized scores.
- **`RegimeThresholdConfig.for_rolling_percentile` keeps
  `spread_2s10s_inversion` at its default of 0**; only the steep threshold is
  percentile-based, so a normal upward-sloping curve is no longer classified as
  inverted.
- **Pre-selection no longer drops low-volatility funds, and clips real crashes
  less.** `PreSelectionConfig.zero_variance_threshold` now defaults to `0.0`
  (was `1e-8`) and drops only columns that are constant up to float noise, so
  money-market and overnight-rate ETFs stay in the universe. The default
  `winsorize_threshold` (`PreSelectionConfig`, `OutlierTreater`, `CleaningConfig`)
  is now 8 sigma (was 3): a genuine single-name crash is no longer flattened
  before CVaR and drawdown are estimated. The `for_conservative` and
  `for_low_volatility` presets keep 2.5.
- **`RobustMeanRiskConfig.for_conservative`, `for_moderate` and `for_aggressive`
  maximise worst-case mean-variance utility, not the Sharpe ratio.** The ratio
  problem is infeasible whenever the worst-case excess return is non-positive,
  which is the normal case on 1-2 years of daily data.
- **The implied covariance estimator needs `implied_annualization_factor`.** 0.1.0's
  `build_cov_estimator` returned a bare `ImpliedCovariance()` for
  `CovEstimatorType.IMPLIED`; it now raises `ConfigurationError` unless
  `MomentEstimationConfig.implied_annualization_factor` (new since 0.1.0) is positive,
  for example 252 for daily returns. skfolio never inferred it from the index (it uses
  252), so weekly data was de-annualised by the wrong factor.
- **`MomentEstimationConfig.for_regime_adjusted_ew()` uses the EW mean estimator**,
  so it fits a ragged panel like its covariance does.
- **Rebalancing treats weights that sum below 1 as cash.** `apply_no_trade_band`
  works to the target's invested level and `compute_drifted_weights` lets the
  cash sleeve earn 0, so a partly invested book no longer trades its cash away or
  reports drift with no market move.
- **Stricter inputs that used to fail silently.** A positive `min_sector_weights`
  floor for a sector absent from the mapping raises `ValueError`; a benchmark or
  factor `y` whose index differs from `X` (same length) raises `DataError`; the
  online grid and randomized searches raise `OptimizationError` when every
  candidate scores NaN and now apply `risk_free_rate=0.0` to every candidate;
  `to_returns` warns when `prices_to_returns` drops all-NaN columns and raises
  when none or the benchmark is left.
- **A weight cap that cannot reach the budget is reported at fit time.**
  `build_mean_risk` and `build_max_diversification` return estimators that warn
  (`UserWarning`) when `n_assets * max_weights` is below the budget, naming the cap,
  the asset count and the number of assets needed, instead of leaving only the
  solver's generic infeasibility error. The presets that default to
  `max_weights=0.10` need at least ten assets. The returned estimators are the
  subclasses `CapacityCheckedMeanRisk` and `CapacityCheckedMaximumDiversification`
  (0.1.0 returned skfolio's own `MeanRisk` and `MaximumDiversification`); fitting,
  cloning and failure handling are otherwise unchanged.
- **`beta_neutral` factor-mimicking portfolios hedge with a lagged expanding
  beta** (NaN for the first 12 shared dates) instead of a full-sample beta, which
  used later data on earlier dates. `scores` must lead `returns`.
- **`CPCVConfig` purges and embargoes by default.** `purged_size` and
  `embargo_size` default to 5 (they were 0), presets included, so out-of-sample
  scores are no longer inflated by autocorrelated returns at the fold boundaries.
  The sizes count observations of whatever frequency is fed in; pass `1` for a
  monthly panel, or `0` to switch the guard off.
- **Black-Litterman helpers**: `market_weights` also accepts a ticker-indexed
  `pd.Series` (aligned to the fitted columns, so a pre-selection drop no longer
  breaks the shape); `AvailableAssetViewsPrior` subsets `empirical_omega` with
  the views and falls back to the equilibrium prior when no view is left.
  `SyntheticDataConfig()` always builds a vine copula with the log-transform
  guard.

### Added

- **validation**: probabilistic and deflated Sharpe ratio (Bailey and Lopez de Prado,
  2014). `probabilistic_sharpe_ratio` gives the probability that the true Sharpe ratio
  beats a benchmark, given the sample length, skewness and kurtosis;
  `expected_max_sharpe` gives the best Sharpe ratio luck alone produces across a number
  of trials; `minimum_track_record_length` gives the observations needed for a
  confidence level. `run_deflated_sharpe` evaluates one selected return series against
  its search, and `run_deflated_sharpe_from_trials` takes the matrix of every trial
  (a grid search, or the candidates of a cross-validation), keeps the best one and
  takes the number of trials and their dispersion from the columns.
- **Requirement traceability (standard 6.6).** `docs/requirements.md` registers 35
  requirements, `REQ-<AREA>-NNN`, each one statement about one module that the tests
  citing it assert: the critical functions of `docs/test-plan.md`, the secret-hiding
  `repr`s, the irreversible-migration reasons, the first-run universe bootstrap, and the
  traceability check and its wiring. A test cites a requirement with the `criterion(id)`
  marker, now registered in all four suites with one description, and a new
  `defect(issue)` marker links a regression test to its GitHub issue; issue 1125 is
  linked this way. `scripts/ci/traceability.py` is both a pytest plugin and the `check`
  command. Each suite's pytest config puts `scripts/ci` on its `pythonpath` and loads the
  plugin with `-p traceability`, which adds `--traceability-out=PATH`: one CSV row per
  (`criterion` id, collected test), plus one row with an empty id for a test that carries
  a `defect` marker and no `criterion` marker. The `lint` job collects the four suites
  with it and uploads the four files as the `traceability` artifact. The check fails when
  a registered requirement has no test, a test cites a `REQ-` id the registry does not
  list or a test carries a `defect` marker but no `REQ-` id, and when the registry or a
  trace file cannot be read or parsed. The `scope-N` and `global-N` labels that
  ingestion and portopt-db tests already carried in the `criterion` marker stay in the
  CSV; the check does not look them up in the registry, but a `defect` marker on a test
  whose only ids are such labels still fails it. Tests written before the registry were
  tagged only where they assert one of the 35 requirements, so most tests cite no id.
  Running a suite's pytest config now needs `scripts/ci` from the checkout: without it,
  pytest stops because it cannot import `traceability`.
- **risk**: new `optimizer.risk` module with a Value-at-Risk calculator and a backtest
  of its forecasts. `compute_var` returns the VaR and the expected shortfall of a
  return sample, as positive loss fractions, by the historical, parametric (normal),
  Cornish-Fisher or bootstrap Monte Carlo method (`VarMethod`, `VarConfig`), at a
  confidence level and a horizon of your choice; `cornish_fisher_quantile` exposes the
  skewness and kurtosis adjustment on its own. `rolling_var_forecasts` makes one-step
  forecasts from the window before each date, with no look-ahead. `backtest_var` counts
  the days the loss beat the forecast and runs the Kupiec proportion-of-failures test,
  the Christoffersen independence and conditional coverage tests, and the Basel
  traffic-light zones (99% over 250 days). The Cornish-Fisher method raises `DataError`
  when the sample's skewness and kurtosis put it outside the expansion's range.
- **portopt-db: read ports for the repository reads `fund` runs.** `portopt_db.ports`
  defines three `runtime_checkable` protocols and the frozen dataclasses they return:
  `PriceReader` (`instrument_for`, `profile_for` and `price_bars_for`, returning
  `InstrumentSummary`, `ProfileSummary` and `PriceBar`), `FxRateReader` (`rates_for`,
  returning `FxRatePoint`) and `UniverseReader` (`get_t212_untradeable_tickers`,
  returning a list of Yahoo symbols, or `None` when the overlay has evaluated no row,
  and `get_t212_unevaluated_tickers`, returning a list of Yahoo symbols).
  `YFinanceRepository`, `FxRatesRepository` and `UniverseRepository` implement them. The
  four new repository methods call `get_instrument_by_yfinance_ticker`, `get_profile`,
  `get_price_history` and `get_rates` and copy the rows into the dataclasses, with the
  decimal columns as `float`. No existing repository method changed its signature or
  return type. `packages/portopt-db/tests/test_ports.py` checks that each repository
  has its protocol's parameters and type hints and that no protocol method's type hints
  name an ORM model.
- **simulation**: new `optimizer.simulation` package for Monte Carlo scenarios and
  Kelly sizing. `simulate_gbm` takes a `GbmConfig` and draws price paths of one asset
  under geometric Brownian motion with the exact log-normal step and optional
  antithetic variates; `simulate_correlated_gbm` does the same for several assets from
  a covariance matrix (singular matrices are handled); `calibrate_gbm` estimates drift
  and volatility from a price series and returns a `GbmFit`. `simulate_bootstrap`
  takes a `BootstrapConfig` and compounds resampled returns into wealth paths, singly
  or in moving blocks that keep short-range autocorrelation; `summarize_paths` returns
  a `PathSummary` with the terminal distribution, the chance of a loss and the
  drawdowns. `kelly_fraction_gaussian` and `kelly_fraction_empirical` give the
  growth-optimal fraction, and `simulate_kelly_growth` shows growth, median wealth and
  ruin probability over a grid of fractions. Settings raise `ConfigurationError` and
  data arrays raise `DataError`.
- **ingestion: the external-call breakers, attempts and timeouts are settings.** The
  ingestion daemon reads them from its environment; `docs/integrations.md` lists them with
  what they set. The defaults are the values the code used before, except
  `BREAKER_TRADING_212_MAX_ATTEMPTS`: the Trading 212 client had no breaker. They are not
  in `ingestion/.env.example` yet.
  - Breaker `max_attempts`: `BREAKER_YFINANCE_MAX_ATTEMPTS` (10),
    `BREAKER_FRED_MAX_ATTEMPTS` (8), `BREAKER_TRADINGECONOMICS_MAX_ATTEMPTS` (5),
    `BREAKER_ILSOLE_MAX_ATTEMPTS` (5), `BREAKER_TRADING_212_MAX_ATTEMPTS` (5).
  - Attempts per request, the first included: `FRED_REQUEST_ATTEMPTS` (5),
    `TRADINGECONOMICS_REQUEST_ATTEMPTS` (3), `ILSOLE_REQUEST_ATTEMPTS` (3),
    `TRADING_212_REQUEST_ATTEMPTS` (5).
  - Request timeouts in seconds: `FRED_REQUEST_TIMEOUT_SECONDS` (30),
    `TRADINGECONOMICS_REQUEST_TIMEOUT_SECONDS` (20), `ILSOLE_REQUEST_TIMEOUT_SECONDS`
    (10), `TRADING_212_REQUEST_TIMEOUT_SECONDS` (30).
  - Base and cap of FRED's retry backoff for errors `is_transient_network_error` does not
    flag: `FRED_RETRY_BASE_DELAY_SECONDS` (1.0) and `FRED_RETRY_MAX_DELAY_SECONDS` (60.0).

  An attempt count that is not an integer of at least 1, or a duration that is not a
  finite number above 0, fails at startup with an error that names the variable. A
  scraper built with an explicit `timeout` keeps it.
- **strategies**: new `optimizer.strategies` package with moving-average and momentum
  strategies and a cost-aware weight backtest. `moving_average` (simple or exponential)
  and `moving_average_crossover_positions` give a trend position per asset;
  `momentum_scores` is the 12-1 momentum (return from `t-252` to `t-21` by default) and
  `cross_sectional_momentum_weights` holds the top fraction of assets at equal weight,
  optionally dollar neutral, rebalanced every N periods. `backtest_weights` simulates
  holding any weight table: a weight decided at a date earns returns only from
  `execution_lag` periods later (1 by default; 0 leaks the future and is for measuring
  that), costs are `CostConfig.proportional_bps` of the drifted-weight turnover, and the
  result has gross and net returns, held weights, turnover, equity curve, Sharpe
  (risk-free rate 0), drawdown and the yearly return lost to costs.
- **ingestion: metrics for the external HTTP sources.** `external_breaker_open{service}`
  reads 1 while the process-wide breaker of FRED, Trading Economics, Il Sole 24 Ore or
  Trading 212 would refuse a request, and `external_request_seconds{service,outcome}` times
  the HTTP requests those four clients send (`outcome` is `ok` or `error`).
- **A circuit breaker on `DatabaseManager` sessions, so a dead database fails jobs
  fast.** `DatabaseManager.get_session` (portopt-db) now asks a per-manager breaker
  first. The paths that reach the database without it (the APScheduler job stores,
  fund's LangGraph pool, startup checks and migrations) are listed in ADR-0008.
  New portopt-db surface:
  - `DatabaseManager(config, breaker=...)` takes an optional breaker, and
    `DatabaseManager.breaker` exposes it;
  - `DbConfig.breaker_failure_threshold` and `DbConfig.breaker_reset_timeout_s`;
  - `portopt_db.resilience`: the `CircuitBreaker` protocol, `DbCircuitBreaker`,
    `BreakerConfig`, `DatabaseUnavailableError`, `BreakerState`, `ProbeTicket`,
    `is_connectivity_failure`, the defaults, and the validator and environment
    parsers.

  After 5 consecutive connectivity failures the breaker refuses new sessions at once
  with `DatabaseUnavailableError` instead of waiting out a connect timeout per call.
  After 30 s it lets a probe session through, one at a time, which connects before the
  caller's code runs and is settled at that connect. Both values are set
  from the environment, for the ingestion daemon and for fund alike:
  `DATABASE_BREAKER_FAILURE_THRESHOLD` (default 5) and `DATABASE_BREAKER_RESET_TIMEOUT_S`
  (default 30.0). Both processes parse them the same strict way: the threshold is
  plain digits and the timeout digits with an optional fraction. Anything else
  (`5.0` for the threshold, a sign, `inf`, an empty value) fails at startup.
  An error is classified in this order: (1) a `DisconnectionError` counts; (2) a
  `DBAPIError` that SQLAlchemy marked `connection_invalidated` counts, tested before the
  SQLSTATE is read; (3) a `DBAPIError` whose SQLSTATE is in class 08 or equal to 57P01,
  57P02 or 57P03 counts; (4) an `OperationalError` without a SQLSTATE raised outside a
  statement counts; (5) any other `DBAPIError` with a SQLSTATE is recorded as a success,
  because the server answered; (6) anything else is inconclusive and leaves the count
  alone: a `DBAPIError` without a SQLSTATE that steps 2 and 4 do not cover, and any other
  exception.
  The ingestion daemon logs a step the breaker stopped without a traceback (ADR-0008
  lists the exceptions under Known limits), counts it in the new
  `jobs_db_unavailable_total` counter, counts a refused failed-status write in
  `jobs_failed_status_refused_total`, and retries on the next scheduled run unless the
  step's row is still active. When the
  failed-status write of a step the breaker stopped does land, `background_jobs.error`
  starts with the new `db_unavailable:` prefix. When it is refused, the row stays
  `running` or `pending` until the orphan reaper, or the reconciliation at the next
  worker start, fails it as `orphaned`. The daemon also
  exposes the breaker as the `portopt_db_breaker_state` gauge (0 closed, 1 open,
  2 half-open).
  The fund cockpit greys out its write actions, and refuses Ask's decision verbs, while
  the breaker is not closed as of the last poll, and says why on the status bar. The
  running daemons pick this up once their images are rebuilt and restarted. See
  `docs/adr/0008-database-circuit-breaker.md`.
- **regression**: a new `optimizer.regression` module. `ols` is a least squares fit
  with t-statistics, p-values and optional Newey-West standard errors for
  autocorrelated residuals. `capm_regression` reports Jensen's alpha and beta of an
  asset against the market net of a risk-free rate, with the Treynor ratio, tracking
  error and information ratio; `rolling_beta` gives the trailing-window beta without
  look-ahead; `security_market_line` fits the cross-sectional line and each asset's
  distance from it. `multifactor_regression` and `fama_french_regression` regress an
  excess return on factor series the caller supplies (the module loads no data) and
  return the alpha, the loadings and their t-statistics. Missing, infinite or
  misaligned data raises `DataError` instead of being dropped.
- **Decision log, integration docs and a changelog check.** `docs/adr/` starts an
  architecture decision log with seven records (the strip to an ingestion pipeline,
  the four-package workspace, the removal of `optimizer.pipeline`, the heartbeat-lease
  reaper, the LangGraph schema, the provider registry and the branching, review and
  secrets policy), and `tests/test_adr_format.py` keeps their shape. `docs/integrations.md`
  documents every external system (Yahoo Finance, FRED, Trading 212, Trading
  Economics, Il Sole 24 Ore, Business Insider and the LLM providers) with its API,
  authentication, failure impact, risk and mitigation, plus a communication diagram
  and two sequence diagrams, and a test checks that each cited source file exists.
  `docs/nfr.md` records dated scalability, performance and maintainability baselines
  against their targets, `docs/test-plan.md` maps the critical functions to their
  tests, and `docs/requirements.md` holds the (still empty) requirement registry.
  An advisory `changelog` CI job (`scripts/ci/check_changelog.py`, pull requests only)
  fails a pull request that changes shipped code (sources, migrations, Dockerfiles,
  entrypoints, installers, dependency pins, agent prompts) without a `CHANGELOG.md`
  entry; the `skip-changelog` label overrides it and Dependabot is exempt. The `lint`
  job, `make lint` and pre-commit now also cover `scripts/`.
- **plotting**: new `optimizer.plotting` package of plotly charts. `correlation_heatmap`
  draws a correlation matrix (Pearson, Spearman or Kendall, or one you pass in) on a
  diverging scale centred at zero, optionally ordered by hierarchical clustering so
  related assets sit together; `efficient_frontier_figure` plots a frontier with named
  highlighted portfolios, and `frontier_from_weights` turns a table of portfolio weights
  into its risk and expected-return points; `returns_volatility_figure` stacks the
  cumulative return, the rolling annualised volatility and the drawdown;
  `security_market_line_figure` plots assets against the CAPM line; `save_figure_html`
  writes any of them to a standalone HTML page. `ChartTheme` sets the template and
  size and `CorrelationOrder` names the heatmap orderings (plain strings work too).
  The package is imported only by `import optimizer.plotting`, never by
  `import optimizer`. `plotly==7.0.0` is now declared as a runtime dependency of
  `portopt-core`: skfolio already pulled it in, so the lockfile only gains the
  declaration and no installed version changes.
- **portopt-db PostgreSQL integration suite and migration checks in CI.** SQLite
  drops the `ON CONFLICT ON CONSTRAINT` target and never installs the
  `human_audit_log` triggers, so neither had a test. `packages/portopt-db/tests/
  integration` now runs `_upsert` (constraint and index targets, SAVEPOINT
  isolation), the append-only UPDATE/DELETE/TRUNCATE guards and a
  constraint-exists-in-the-migrated-schema check against a real server, keyed on
  `PORTOPT_TEST_DATABASE_URL` (skips without it, refuses a database not named
  `*_test`). A static test checks the 37 `_upsert` conflict names against the
  models, and the `postgres-integration` CI job now also runs `alembic check` and a
  downgrade/upgrade round trip over the five most recent revisions. The six
  revisions whose `downgrade` raises now declare an `IRREVERSIBLE_REASON`, which a
  test requires of any future one-way revision.
- **derivatives**: option pricing and hedging in the new `optimizer.derivatives`
  package. `black_scholes_price` and `black_scholes_greeks` price European calls and
  puts with a continuous dividend yield (Greeks: delta, gamma, vega per 1.00 of
  volatility, theta per year, rho per 1.00 of rate) and accept numpy arrays that
  broadcast, so a strike by volatility grid is one call. `implied_volatility` inverts
  the price with Brent's method and rejects prices outside the no-arbitrage bounds.
  `binomial_price` is a Cox-Ross-Rubinstein tree, European or American, that
  converges to Black-Scholes. `simulate_delta_hedge` sells an option, hedges it with
  the Black-Scholes delta at a chosen number of dates, with proportional trading
  costs and a hedge volatility that may differ from the true one, and returns the
  profit and loss of every simulated path.
- **Ask is a read-only deep agent with a saved conversation.** A question is now
  answered by an agent that reads the fund's data (read-only tools over
  `fund.observe`) and a read-only staged copy of the docs and source, deciding its
  own steps (shown live on the status line), capped by `FUND_ASK_MAX_STEPS` and
  `FUND_ASK_TIMEOUT_SECONDS`. Follow-up questions work: each scope keeps one
  checkpointer thread that deepagents compacts automatically. The exact
  conversation is saved per portfolio (and for the fund-wide view) and shown when
  Ask opens; `/clear`, behind a confirm, deletes the saved messages, the agent's
  memory and its offloaded history files.
- **credit**: new `optimizer.credit` package with the Merton (1974) structural credit
  model. `solve_merton` reads a firm's equity as a call option on its assets and
  inverts the observed equity value and equity volatility into the asset value and
  volatility, then reports the distance to default, the default probability (the
  risk-neutral one, or the real-world one with `MertonConfig.asset_drift`), the
  market value of the debt, its credit spread and the leverage.
  `estimate_equity_volatility` annualises the volatility of an equity price series,
  optionally over the latest `window` returns, and `merton_from_market` solves the
  model from a market capitalisation, a price history and the total debt. A solve
  that cannot reach `solver_tolerance` raises `ConvergenceError` instead of
  returning an approximation.
- **The Daily brief and Ask text is laid out to be read.** The market digest is a
  table (one macro series per line with value, date and signed change, then the
  markets and news themes), and Ask answers render their headings, bold, bullets
  and dividers instead of showing the markdown marks, with a blank line between
  turns.
- Structured logging with `NullHandler` pattern across all modules (#36)
- CI Python version matrix testing (3.10, 3.11, 3.12) (#38); the matrix is now 3.12
  and 3.13 (see **Changed (BREAKING — library)**)
- Test coverage reporting with Codecov integration (#39)
- This CHANGELOG file (#40)
- CI restructure: parallel jobs, concurrency control, format checking (#41)
- Packaging: project URLs, dependency groups, version strategy (#45)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CITATION.cff (#46)
- GitHub issue and PR templates (#47)
- Dependabot for dependency and GitHub Actions updates (#48)
- Property-based testing with Hypothesis (#49)
- Developer experience: Makefile (#50)
- Automated release workflow: tag check, CI gate and `portopt-core` build (#52)
- Pyright compatibility configuration (#53)
- **api**: `app/worker.py` — the ingestion daemon entrypoint. Serves Prometheus
  metrics, reaps orphaned jobs, bootstraps benchmarks off-thread, then runs
  APScheduler until SIGTERM
- **api**: `app/cli.py` — manual runs (`python -m app.cli daily`, `yfinance`,
  `macro`, `fred`, `news`, `summarize`, `calibrate`, `universe`,
  `reference-indices`, `refetch-all`). Each goes through the same job-slot and
  heartbeat path the scheduler uses, so a manual run is refused rather than
  double-fetching when the scheduler is mid-step
- **api**: `universe_build` promoted to a scheduled job (Sun 02:00, ahead of
  `weekly_refetch` — every other step iterates the `instruments` table, so a
  stale universe silently caps what yfinance fetches)
- **fund / portopt-db / ingestion**: the **eligible universe**, so a portfolio can
  be restricted to a hand-picked set of instruments (the first real ETF portfolio
  could not run: its ETFs ranked below 7,000 by market cap, had no sector and
  failed every ESG exclusion)
  - `PortfolioMandate.eligible_tickers` — a hard allowlist, empty meaning
    unrestricted; set with `fund mandate set --eligible T ... | --clear-eligible`
    or the TUI Mandate view's "eligible tickers" field. Saving warns once about
    tickers without a priced instrument and syncs the portfolio's pins
  - `fund.tools.eligibility` resolves the allowlist once per run (run, resume and
    the adviser edit) into `RunContext.eligible`; `list_universe` returns only the
    allowed set (with `instrument_type` / `asset_class`), `universe_filter`,
    `estimate_moments` and `optimize_portfolio` refuse an outside ticker,
    `risk_check` records a `not_eligible` violation, and `place_orders` and the
    adviser edit refuse it. The PM seed lists the eligible and barred names
  - ESG look-through for ETFs: a fund is judged by its latest `etf_sector_weights`
    snapshot against the client's new `look_through_tolerance` (strict 0 by
    default); commodity and government / inflation-linked bond ETFs pass, other
    funds with a missing, stale (`FUND_ESG_SECTOR_WEIGHTS_MAX_AGE_DAYS`, 180
    days), partial or unmappable snapshot are barred `esg_unclassified`. Under an
    allowlist the stored-sector screen leaves ETFs to the resolver
  - `pinned_instruments` table (Alembic `e3a5c7d9b1f4`) and
    `PinnedInstrumentRepository`: a pinned ticker is never marked delisted by the
    universe build and keeps its classification and `t212_ticker`
  - `portopt add-instrument <TICKER> [--asset-class ...] [--pin]` validates one
    Yahoo ticker, stores the instrument with 5y of prices and the ETF fund data,
    and takes no job slot; ETF fund data is now fetched for every ETF, and
    `commodity` is a new asset class
  - `ETFMetadataRepository.get_latest_sector_weights` batches the latest sector
    snapshots for a list of tickers
  - `PortfolioMandate.tradeable_only`, an opt-in broker scope on top of the
    allowlist (default off, mandate JSON only, no migration): `fund mandate set
    --tradeable-only | --no-tradeable-only` or the TUI Mandate view's "Tradeable
    on Trading 212 only" box. The resolver bars an instrument whose
    `is_t212_tradeable` is an explicit False as `not_t212_tradeable`; NULL is not
    barred, and when nothing has been evaluated the universe stays whole with a
    note in the PM seed and the mandate-save warning. It also works without an
    allowlist, as a refusal list over the whole universe (`list_universe` drops the
    False rows). `UniverseRepository.get_t212_untradeable_tickers` is the new read
- **Function length and complexity ratchet.** `scripts/ci/check_complexity_ratchet.py`
  (a step of the `lint` job) holds standard 2.8 (at most 30 lines per function,
  cyclomatic complexity below 10) over the four source trees. The 274 functions
  that are already too long and the 50 that are too complex are recorded in
  `scripts/ci/baselines/`; the check fails a new offender and a recorded one that
  grew. `--update` on an existing baseline can only lower a stored value or drop
  an entry, never raise a value or add one, so the recorded debt shrinks as
  functions are split. Deleting a baseline file and running `--update` starts a
  fresh baseline from the current measurement, so a deleted or emptied baseline
  file in a diff deserves a reviewer's attention. A source root that is missing
  fails the check instead of measuring nothing.
- **statarb**: new research package `optimizer.statarb` for pairs trading and PCA
  statistical arbitrage. `engle_granger` runs the two-step cointegration test: an OLS
  hedge ratio, then an Augmented Dickey-Fuller test on the residuals with the lag count
  chosen by the Akaike criterion. It uses the asymptotic MacKinnon critical values with
  no finite-sample correction, so its decision is approximate for short samples.
  `ou_half_life` gives the half-life of mean reversion of any series.
  `kalman_hedge_ratio` tracks a drifting hedge ratio and returns the one-step-ahead
  spread and its z-score, and `zscore_positions` turns a z-score into positions with
  entry, exit and stop bands. `fit_pca_residual_model` and `pca_signals` implement the
  Avellaneda and Lee (2010) eigenportfolio residual model: the Ornstein-Uhlenbeck fit
  of each residual, its s-score and the paper's trading thresholds. These are research
  tools that ignore costs and borrow limits and are not wired to the fund, whose
  mandate is long-only and paper-only. They need numpy and scipy only.
- **fund: execution venues, paper-ticket pricing without a session, and NAV marking
  without a session.** `fund.pricing` (`BPS_PER_UNIT`, `DUST_WEIGHT`, `CostModel`,
  `FillQuote`, `TicketLine`, `PricedTicket`, `held_weights`, `price_ticket`) and
  `fund.execution` (`ExecutionVenue`, `OrderRequest`, `PlacedOrder`,
  `FillQuoteSource`, `PaperVenue`) are new modules. `place_orders` takes an optional
  keyword-only `venue` and `RunContext` has a `venue` field; both default to `None`,
  the paper venue, so a non-paper venue can be passed in without editing the tool.
  `fund.performance` adds `load_nav_inputs`, `compute_nav`, `NavInputs` and
  `NavSeries`.

### Changed

- **docs**: the README, `CLAUDE.md` and the `optimizer` package docstring list the
  eight research tools added to the library (`risk`, `simulation`, `strategies`,
  `regression`, `plotting`, `derivatives`, `credit`, `statarb`), so the library now has
  26 submodules, and `validation` is described with the deflated Sharpe ratio.
  `CLAUDE.md` no longer says `arch` arrives through skfolio: it does not, and neither
  `arch`, `statsmodels`, `hmmlearn` nor `matplotlib` is installed.
- **portopt-db: `newest_close_per_instrument` is renamed `newest_close_dates`** on the
  `PriceReader` port and on `YFinanceRepository`, since it returns bar dates, not
  closes. `fund.observe` calls the new name; a caller of the old name gets an
  `AttributeError`. The statement it runs is the same.
- **fund, ingestion, docs: names, docstrings and tests that say what the code does.**
  `fund.ask_context.short_id` is the one definition of the compact id that
  `fund.tui.ask` and the cockpit views that show one (through `_short` in
  `tui/_views/shell.py`) use, `build_context` no longer takes the `question` it never
  read, `fill_shortfalls` and the paper venue's quote lookup share one helper, and
  `_cost_params_match` names its tolerances (same values). Docstrings now say that a
  re-run of a stored paper ticket with other cost params is refused as far as the
  ticket's stored lines show (the commission of a line whose commission base is 0 is
  not compared), that `fund.performance` reads `nav_history` and `instruments`
  rather than `positions` and that the fund worker runs the daily NAV mark; three
  ingestion docstrings no longer tie a refusal to an open breaker, since a half-open
  one refuses too. `docs/WRITE-PATH-AUDIT.md` describes the `exchanges.name`
  constraint the model and migrations have and points rows 2-10 and 12 at the
  modules that now hold their upserts. Fourteen repository-method tests name the
  method in their docstrings, the two re-stamp tests are one, and the heartbeat test
  waits for its pulses instead of sleeping a fixed 0.4 s.
- **ingestion, ci: messages that say what the code does.** While its failure count is at
  or above its limit inside a back-off window, an external-source breaker refuses with
  `<service> circuit breaker open: failure count N, limit M, inside its back-off
  window.` instead of `<service> rate limit persists after N attempts. ...`, because
  what counts as a failure is up to the code that trips it. The log line for a
  failed-status write the database breaker refused, and the help text of
  `jobs_failed_status_refused_total`, now say that once the row's lease has expired, an
  orphan reap (the periodic one or the one at worker start) that the breaker lets reach
  the database fails it; they used to give "once the database answers" as the
  condition. The description of `FRED_RETRY_BASE_DELAY_SECONDS` says how a wait is
  drawn, and that a flagged error sleeps a short jitter this base does not scale before
  the next attempt goes through the breaker's check. The `--traceability-out` help text
  now also names the row with an empty id that a test with `defect` markers and no
  `criterion` marker gets. `scripts/ci/traceability.py check` exits 1, naming the file,
  when a `defects` cell of a trace CSV holds anything but positive issue numbers in
  plain digits without leading zeros; it used to accept 0, negative numbers and leading
  zeros, and to fail on a non-integer without naming the file. These source changes
  keep behaviour: the rate-limit jitter bounds of
  `retry_with_backoff` are named constants with the same values, and
  `UniverseRepository.get_instrument_by_yfinance_ticker` and `instrument_classes_for`
  share one sort-key tuple that compiles to the same SQL. Renamed private names and
  tests, and reworded docstrings and comments, change no behaviour either.
- **ingestion: the requests `Trading212Client` sends go through `retry_with_backoff` and a
  circuit breaker.** A 408, a 429 or a 5xx answer, and a `requests` failure that is not an
  HTTP error, such as a timeout, are passed to the breaker and retried while attempts
  remain, unless the breaker refuses the attempt; any other 4xx is still raised at once, and
  so is an HTTP error without a response. The wait a 429 reports can lengthen the breaker's
  window, which stays capped at 120 s. The clients that `Trading212Client.from_settings`
  builds share one breaker per process. When it refuses with `CircuitOpenError`,
  `t212_annotate` fails before it reads or writes any instrument, as for any other
  unreachable catalogue. A `Retry-After` of 0 or below used to mean no wait (0) or the
  exponential backoff (below 0); it now sets no floor on the window.
- **ci**: the CI workflow also runs on pushes to `feature/**`, `bugfix/**`,
  `hotfix/**` and `release/**`, so a change merged without a pull request still
  carries the status checks that branch protection on `main` and `development`
  requires; `tests/test_ci_triggers.py` guards the trigger list.
- **api**: the service is now a headless ingestion daemon. APScheduler runs
  in-process; there is no HTTP API. Job state is read from the `background_jobs`
  table, the logs, and Prometheus on `METRICS_PORT` (default 9000, also the
  container healthcheck target)
- **api**: the scheduler is composed of public per-step functions
  (`run_yfinance_step`, `run_macro_step`, …). The cron pipelines and the CLI both
  build on those, instead of maintaining parallel code paths that drift
- **api**: no longer depends on the `optimizer` library. The daemon ingests; it
  does not optimize. The image no longer carries the sklearn/skfolio/scipy stack,
  and CI enforces the split by installing only `api/requirements.txt`
- **api**: `scheduler/fetch.sh` and `refetch_all.sh` are thin wrappers over the
  CLI rather than curl-and-poll drivers against HTTP endpoints
- **docker**: compose service `api` → `scheduler`; the API port is gone and 9000
  (metrics) is published instead
- Type-check target raised to Python 3.12 (see Fixed). The runtime floor is 3.12
  as well (see **Changed (BREAKING — library)**)
- **api**: renamed to **`ingestion/`** — the folder is a schedule-driven worker,
  not an HTTP service, so the name now describes the domain rather than a
  removed transport layer. Docker (`ingestion/Dockerfile`), CI (job renamed
  `api-test` → `ingestion-test`), the `scheduler/*.sh` wrappers, and the docs
  were all updated to match; no code paths changed
- **docs**: the architecture and decision records (`ARCHITECTURE.md`,
  `DECISIONS.md`, `SPEC-RECONCILIATION.md`, `VALIDATION.md`, `WRITE-PATH-AUDIT.md`)
  moved from the git-ignored `todo/` folder to a tracked `docs/`, so they are
  versioned and reviewable. Pointers to `SPEC.md`, `deep_agent.md` and
  `tasks/plan.md`, which were never committed, were dropped from `CLAUDE.md`,
  `fund/README.md`, `ci.yml` and code comments, and
  `tests/test_doc_references_resolve.py` fails on any new one. `CONTRIBUTING.md`
  now describes the uv workspace, the per-member test commands, branching from
  `development` and releasing through a `release/` branch, and the `portopt-db` README
  documents the migration process (validation, rollback, one-way revisions) with an ER
  diagram; new Alembic revisions get a dated file name.
- **ci**: the `fund-test` job timeout goes from 30 to 45 minutes. The job takes 25 to
  29 minutes, and a pull-request run competing with the push run was cancelled at
  30m17s although the same job passed on the same commit.
- **ingestion**: the yfinance sweep decides which categories are due for a fetch
  through a pure, unit-tested `FetchPlan`
  (`app/services/market_data/yfinance/fetch_plan.py`), built once per ticker,
  instead of repeating the incremental-mode check before every category.
  `StalenessThresholds` moved into that module and is still importable from
  `yfinance_data_service`. Behaviour is unchanged.
- **ingestion**: the yfinance sweep refreshes its six financial-statement groups,
  its fundamentals and its corporate-action, analyst and holder categories through
  one `run_category` (`app/services/market_data/yfinance/categories.py`), driven by
  a `CategorySpec` per category, instead of a copy-pasted fetch, store and error
  block for each. The savepoint every repository write runs in moved into that
  module, and `FetchPlan.is_due` no longer takes a `fresh_hours` override, which
  only repeated each event marker's own window. Counts, errors, skips and the
  memory of empty answers are unchanged; the failure warning of a migrated
  category now comes from the `categories` logger and names it by its key (for
  example `price_targets`, `financials.balance_sheet.annual`).
- **ingestion**: the yfinance sweep's `_fetch_and_store` is split into profile,
  price, statement, fundamentals, event and news steps and now only builds the
  sweep's context and plan, in 27 lines instead of about 470. Whether a fetched
  full-period price history may replace the stored series is decided by a pure,
  unit-tested `decide_price_storage`
  (`app/services/market_data/yfinance/price_policy.py`). The pause before each
  scraped news article is the new `YFINANCE_NEWS_SCRAPE_DELAY_SECONDS` setting
  (default 0.5, the value it always had; 0 to 10). The unused
  `FetchPlan.all_fresh` is removed, and the complexity ratchet's baselines no
  longer list `_fetch_and_store`. Behaviour is unchanged.
- **ingestion**: four more long functions are split, each after characterization
  tests pinned its current results. `MacroNewsFetcher.fetch_all` (151 lines) is now
  ticker and search collection, a budget merge and a pure theme pass.
  `MacroRegimeService.fetch_country` (131 lines) runs one step per source.
  `run_t212_annotate` (122 lines) reads the catalogue, snapshots the universe, runs
  the batched annotation loop and reports. The two Trading Economics
  `country-list` getters and their parsers, which were copies, share one
  implementation driven by an entry per page. The four news paths read a Yahoo
  item's link through one `extract_article_link`
  (`app/services/market_data/yfinance/news/_links.py`), each passing its own
  fallback where the old copies disagreed. The complexity ratchet's baselines
  drop those functions: 266 long and 46 complex functions remain. Behaviour is
  unchanged.
- **fund**: three long functions of the agents and actions are split, each after
  characterization tests pinned its current results. The Ask agent's
  `build_ask_tools` (210 lines) builds its eleven data tools from one
  `make_<tool>_tool` factory each, and the read-only session, the portfolio
  resolution and the one-portfolio confinement they share live on a frozen
  `AskToolContext`. `run_fund` and `stream_fund` close a round through one
  `_settle_round` instead of two copies. `apply_action` (130 lines) groups its
  keywords into a frozen `ActionRequest` and dispatches it through
  `ACTION_HANDLERS`, one handler per `ActionKind`; its signature, its replay of a
  repeated idempotency key and its error for an unknown action are unchanged. The
  complexity ratchet's baselines drop `build_ask_tools` and `apply_action` and
  lower `run_fund` and `stream_fund`: 264 long and 44 complex functions remain.
  Behaviour is unchanged.
- **library**: three long functions are split, each after characterization tests
  pinned its current results. `run_regime_validation` (159 lines) aligns the
  regimes, measures the sub-periods and the regimes, and computes the alpha
  shares in one private helper each, and the NaN metrics of a slice under
  `min_regime_obs` come from one shared constant instead of three copies.
  `build_mean_risk` (102 lines) builds its sector caps, floors and regime bands in
  `_compose_sector_constraints`, which returns the `groups` and
  `linear_constraints` keywords; explicit keywords still win. The composite macro
  regime score (`_score_composite`, 63 lines) scores each indicator through one
  `_band_score` instead of four copies. A sector cap without a sector mapping
  still only logs a warning, while floors without one raise and bands raise
  when there is neither a mapping nor an explicit `groups`. The complexity
  ratchet's baselines drop the three functions: 261 long and 41 complex
  functions remain. Behaviour is unchanged.
- **fund**: the three largest modules are split, and every import path still
  works. `fund/tui/views.py` (3,345 lines) keeps the sidebar specs and
  `build_views` and re-exports the views, now one per module in
  `fund/tui/_views/`. `fund/tui/app.py` (3,341 lines) keeps `ShellApp`'s state,
  layout, navigation, `CSS` and `BINDINGS`; its modal screens moved to
  `fund/tui/screens.py` and its sections to plain mixins in `fund/tui/_mixins/`,
  which are type-checked against the type-only `ShellHost` declarations in
  `fund/tui/_mixins/host.py`. `fund/agents/graph.py` (2,736 lines) keeps the run
  drivers, `FundRun`, `build_fund_agent` and every other run-graph function that
  calls a name the tests patch on `fund.agents.graph`, so those patches still
  reach the run graph (the profiler calls its own `build_backend` binding); it
  also keeps the four role builders and their descriptions, because a skills
  test reads `_ALLOCATOR_DESCRIPTION` from its source. The subagent factory, the
  run instruction and the journal, HITL, failed-round and stream helpers moved to
  `graph_assembly`, `graph_prompts`, `graph_journal`, `graph_hitl`,
  `graph_failures` and `graph_stream`. Each definition moved byte-for-byte, the
  old modules re-export every name they defined, the moved code logs under the
  old logger names, and the complexity ratchet's baselines follow the moved
  functions with unchanged values. Behaviour is unchanged.
- **fund**: the agent layer no longer imports the TUI. `StepStatus` and
  `StreamEvent` moved from `fund/tui/stream.py` to `fund/stream_events.py`;
  `MAX_MESSAGE_CHARS`, `strip_control`, `clip`, `defang_markdown` and
  `describe_error` moved from `fund/tui/safe_text.py` to `fund/text_safety.py`;
  and `build_context` moved from `fund/tui/ask.py` to `fund/ask_context.py`.
  `fund/agents/graph.py`, `graph_stream.py` and `ask_agent.py` import them from
  their new modules. The old modules re-export those names, so imports from the
  old paths still work. Replacing a module attribute reaches only code that reads
  the name from that module, and the Ask agent's `fund_snapshot` tool now reads
  `build_context` from `fund.ask_context`, so `test_ask_tools_contract.py` and
  `test_ask_agent.py` now patch `fund.ask_context` instead of `fund.tui.ask`. A
  new guard, `test_agents_do_not_import_tui.py`, fails on an `import` or
  `from ... import` statement under `fund/agents` that names `fund.tui` or one of
  its submodules, function-local, type-only and relative statements included,
  and on an `import_module`/`__import__` call whose first positional argument is
  a string literal naming one; computed names, keyword arguments, relative string
  names and the `fromlist` form are not detected. It also checks that the three
  new modules load no `fund.tui` module and none of `deepagents`, `langchain`,
  `langchain_core`, `langgraph`, `optimizer`, `skfolio` or `textual`. Behaviour
  is unchanged.
- **fund**: the reads `fund` ran through `YFinanceRepository`, `FxRatesRepository` and
  `UniverseRepository` go through the `portopt_db.ports` protocols. In
  `fund/readers.py`, `price_reader`, `fx_rate_reader` and `universe_reader` return those
  repositories typed as `PriceReader`, `FxRateReader` and `UniverseReader`.
  `fund.tools.orders`, `fund.tools.prices`, `fund.tools.mandate`, `fund.tools.universe`,
  `fund.tools.eligibility`, `fund.actions` and `fund.performance` take their readers
  from it and no longer import the three repositories, and the five of them that read
  instrument, profile, price or FX rows now read the DTOs instead of ORM rows. Tool and
  public function signatures are unchanged. Every numeric column those five read from
  the rows was already passed through `float()`, so the `float` fields of the DTOs give
  the same numbers. `price_bars_for` and `rates_for` call `get_price_history` and `get_rates`, so
  tests that replace either on the repository class still see each read. The complexity
  ratchet's baselines for `_check_tradeable` and `_load_price_frame`, both shorter now,
  are lowered. Behaviour is unchanged.
- **fund**: statements that eight `fund` modules built and ran inline now run in
  repository methods that return builtins or frozen dataclasses: `fund.worker` calls
  `FundJobRepository.fail_active_slots_of`, `fund.scheduler` calls
  `FundJobRepository.jobs_started_since` (`StartedJob`), `fund.actions` calls
  `HumanAuditLogRepository.latest_idempotency_key`, `fund.tools.orders` calls
  `OrderRepository.restamp_created_at`, `fund.performance` calls
  `OrderRepository.filled_tickets` and `latest_filled_ticket` (`FilledTicket`) and
  `NavHistoryRepository.newest_nav_dates`, `fund.agents.graph_failures` calls
  `AgentRunRepository.insert_run_if_absent`, `fund.observe` calls
  `YFinanceRepository.newest_close_dates` through the `PriceReader` port, and
  `fund.tools.eligibility` calls `UniverseRepository.instrument_classes_for`
  (`InstrumentClass`) through the `UniverseReader` port and
  `ETFMetadataRepository.newest_stock_pct`. Each method runs the statement the module
  ran before, on the session the caller passes, with the same parameters except the time
  stamp noted below. None of the methods commits; the callers that committed after the
  inline statement still do. Code that read columns of ORM rows or entities now reads
  them from the returned values instead: frozen dataclasses with the same field names,
  or, for the fund stock shares, a mapping of instrument id to share.
  `fund.performance`'s private ticket helpers, for example, hand `FilledTicket` values
  instead of `PaperOrder` rows to the code that reads a ticket's id, fill date, notional
  and lines. The one change in a stored value is that `fail_active_slots_of` takes the
  `finished_at` and `updated_at` stamp of the slots it fails just after the session
  opens instead of just before. `performance.py` may now import the `fund.audit`
  repositories: the static import scan in `test_performance_nav.py` bars only the
  `deepagents`, `langgraph`, `langchain` and `fund.agents` prefixes for it (it still
  bars `fund.audit` too for `attribution.py`). A new fresh-interpreter probe imports
  `fund.performance` and every module it names in an import statement, and fails if any
  of them loads a module with one of those four prefixes. A new guard,
  `fund/tests/unit/hygiene/test_no_orm_outside_repositories.py`, counts `sqlalchemy`
  `select`/`insert`/`update`/`delete`/`text` calls bound by import, `.execute`, `.query`
  and `.add_all` calls, `.scalar`/`.scalars` calls with a positional argument, and
  `.get(Model, ...)`/`.add(Model(...))` calls with a model imported by name from
  `portopt_db.models`, in every module under `fund/src/fund` except
  `fund/audit/*repository.py`. It fails on a call its allowlist `orm_baseline.txt` does
  not hold and on an entry the tree no longer matches; its docstring lists the forms it
  does not detect. The guard found 36 calls under 34 keys; the allowlist now holds 11
  calls under 9 keys: `fund.tools.universe` (`list_universe`, `_investable`,
  `uninvestable_reasons`), the Ask tools' `SET TRANSACTION READ ONLY` in
  `fund.agents.ask_agent`, and the psycopg schema bootstrap in `fund.audit.persistence`.
  The complexity ratchet's long-function baselines are lowered for
  `run_scheduled_nav_mark` (56 to 47 lines) and `build_nav_series` (118 to 107), and
  `_fail_in_flight_slots`, now within the limit, is dropped from them.
- **db**: `YFinanceRepository` is split into seven repositories in
  `portopt_db/repositories/market_data/`: `PricesRepository`, `FundamentalsRepository`,
  `AnalystRepository`, `OwnershipRepository`, `NewsRepository`, `OptionsRepository` and
  `StalenessRepository`. `YFinanceRepository` subclasses the seven and keeps its 53
  public methods with the signatures they had; `test_yfinance_repository_facade.py`
  compares them with a snapshot taken before the split. The lookups by Yahoo symbol
  (`get_instruments_with_yfinance_ticker`, `get_instrument_by_yfinance_ticker`,
  `get_benchmark_coverage`) move to `UniverseRepository`, and `YFinanceRepository`
  forwards those three calls to it; `DORMANT_AFTER_DAYS` moves to the
  `universe_repository` module. The four `PriceReader` methods are still defined in
  `yfinance_repository.py`. `_safe_val`, `_safe_int`, `_safe_float`,
  `_safe_percent_as_fraction`, `_safe_str`, `_safe_date`, `_strip_nul`,
  `_refreshable_columns` and `_last_wins` move to `_mapping.py`, whose new
  `truncate(model, column, value)` reads a string column's width from the model.
  `truncate` replaces 12 of the 48 literal widths passed to `_safe_str`. Of the other
  36, 2 keep their literal because no model width matches it (`SecFiling.url` has no
  length, and the insider-purchase label has no column). The remaining 34 sit in eight
  functions (`upsert_profile`, `upsert_profile_extras`, `upsert_analyst_actions`,
  `upsert_sec_filings`, `upsert_financial_statements`, `upsert_insider_transactions`,
  `upsert_insider_roster`, `upsert_news`): converting every such call in one of them,
  measured on a copy, made ruff format wrap enough lines to push that function past
  its complexity ratchet baseline or the 30-line limit, so those 34 keep their
  literals. Apart from the 12 `truncate` calls, the moved code is the code that was
  there before, except that the `get_benchmark_coverage` docstring no longer names its
  callers and `get_instrument_by_yfinance_ticker` takes its sort keys from a
  module-level tuple it shares with `instrument_classes_for`. The constants, helpers
  and function that ingestion and this package's tests import from
  `yfinance_repository` are re-exported from it.
  `upsert_financial_statements` and `upsert_dividends` now log under the
  `fundamentals_repository` and `prices_repository` logger names. The ratchet baseline
  entries of the moved functions now use the new keys, with the values they had.
- **fund**: `place_orders` places its ticket through an execution venue, by default
  `PaperVenue`, which prices the ticket through `fund.pricing.price_ticket`, and
  `build_nav_series` is now `load_nav_inputs` followed by `compute_nav`. The six
  `1e4` literals of `fund/tools/orders.py` are `BPS_PER_UNIT` in the code that moved
  to `fund/pricing.py` (five) and `fund/execution.py` (one), and `_held` and
  `_DUST_WEIGHT` are now `fund.pricing.held_weights` and `DUST_WEIGHT`. `PaperVenue`
  runs what `place_orders` ran inline after checking its arguments: the stored-ticket
  lookup by weights hash, the cost-param guard, the re-stamp of a returned ticket, the
  dust check, the pricing and the `paper_orders` insert. `_cost_params_match`,
  `_ticket_from_row` and `_make_newest` move to `fund/execution.py`. `place_orders`
  keeps its existing parameters, its error texts and the order of its early exits, and
  the bound `place_orders` tool passes `RunContext.venue` on. `_next_close` and
  `_base_close` stay in `fund/tools/orders.py`, beside a new `_fill_quotes` that calls
  them, and `place_orders` builds the paper venue with `_fill_quotes` on each call
  that passes no venue, so tests that replace `orders._next_close` still reach the
  fill. `load_nav_inputs` reads the tickets, closes and FX rates, applies the currency
  and FX-coverage guards, converts the closes into the base-currency mark frame and
  picks the benchmark column, with a warning when the benchmark is left out; it
  returns `None` when there is nothing to mark. `compute_nav` takes that `NavInputs`
  and no session and marks it: shares from the ticket lines, the chain links and the
  per-bar marks. `test_price_ticket.py` and `test_performance_nav_inputs.py` hold
  tickets and NAV series printed by the code before this change, and the new code
  reproduces them exactly. The complexity ratchet drops `place_orders` and
  `build_nav_series` from its baselines and lowers `_place_orders_impl` from 49 to
  41 lines.

### Removed

- **ci**: the `publish` job of `release.yml`. This project is no longer published to
  PyPI: the `pypi` environment, the `id-token: write` grant and the
  `pypa/gh-action-pypi-publish` step are gone, so a `v*` tag cannot upload anything.
  A tag is still checked (on `main`, equal to the `pyproject.toml` version), runs
  the full CI workflow and then builds the `portopt-core` sdist and wheel as the
  `dist` artifact. `test_release_workflow_never_uploads_to_pypi` fails if an upload
  step comes back, and the README, CONTRIBUTING and CLAUDE.md no longer describe one.
- **api**: the entire HTTP layer — every `/api/v1/*` route, FastAPI, uvicorn, the
  API-key auth middleware, and the `api_keys` table
- **api**: the portfolio, optimization, backtest, factor, risk, rebalancing,
  attribution, dashboard, reports, scenarios, and views domains, across routes,
  services, repositories, models, and schemas
- **api**: six yfinance sub-clients that wrote nothing to the database — `market`,
  `sectors`, `screener`, `calendars`, `streaming`, `funds`
- **db**: migration `d1e2f3a4b5c6` drops the 17 tables those domains owned. It is
  **destructive and one-way** — `downgrade()` raises rather than recreate empty
  tables whose rows and owning code are both gone. Restore from a pre-upgrade dump
- `scheduler/smoke.sh` and `smoke.yml`, which drove `/optimize` and `/backtest`
- `TRADING_ECONOMICS_API_KEY`, plus Azure OpenAI / `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` from `.env.example`. None were read by any code — Trading
  Economics and Il Sole 24 Ore are scraped from HTML and take no key
- The `examples/` directory and the MkDocs documentation site (#50, #51). Both were
  part of the 0.1.0 tree
- **ingestion**: code nothing called. `ArticleScraper.fetch_multiple` and its
  `ArticleScraperProtocol` declaration.
  `TradingEconomicsIndicatorsScraper.get_all_portfolio_countries`. The `__main__`
  blocks of the Trading Economics and Il Sole scrapers, which scraped live pages
  when the module was run as a script. Also 28 response schemas left from the
  HTTP API:
  - `_shared`: `AsyncJobCreateResponse`.
  - `macro`: `MacroFetchJobResponse`, `FredObservationResponse`,
    `EconomicIndicatorObservationResponse`, `TradingEconomicsObservationResponse`,
    `BondYieldObservationResponse` and `MacroNewsResponse`.
  - `market_data`: `YFinanceFetchJobResponse`, `YFinanceSingleFetchResponse`,
    `TickerProfileResponse`, `PriceHistoryResponse`, `FinancialStatementResponse`,
    `DividendResponse`, `StockSplitResponse`, `AnalystRecommendationResponse`,
    `AnalystPriceTargetResponse`, `InstitutionalHolderResponse`,
    `MutualFundHolderResponse`, `InsiderTransactionResponse` and
    `TickerNewsResponse`.
  - `universe`: `ExchangeResponse`, `InstrumentResponse`, `InstrumentListResponse`,
    `BuildResultResponse`, `BuildJobResponse`, `BuildProgressResponse`,
    `CacheStatsResponse` and `UniverseStatsResponse`.

### Security

- **deps**: `soupsieve` 2.9.2 -> 4.1 in `uv.lock`. 4.0.0 is the first release that
  fixes CVE-2026-126190 and CVE-2026-126192 (super-linear CSS selector matching on
  untrusted HTML, reachable through the news and macro scrapers); the lock takes
  the current 4.1. `pip-audit --strict` failed on both, which turned the `lint`
  job red on every branch; `tests/test_lockfile_security_floors.py` now pins the
  4.0.0 floor.
- **deps**: `wcmatch` 11.0.1 -> 11.2 and its helper `bracex` 3.0.1 -> 3.2.4 in
  `uv.lock`. 11.2 is the first release that fixes CVE-2026-132276 (a glob pattern with
  several `*` wildcards takes polynomial time to reject a crafted file name);
  `deepagents` pulls it in. `pip-audit --strict` flagged it, which turned the `lint`
  job red on every branch and on `development` itself;
  `tests/test_lockfile_security_floors.py` now pins the 11.2 floor. The lock stays on
  the 11 line, since 12.0 is a major release.

### Fixed

- **launcher, fund**: `optimizer status`, or any argument that is not a portfolio
  UUID, no longer rebuilds the images and asks for the master passphrase before
  crashing the cockpit with a `ValueError: badly formed hexadecimal UUID string`
  traceback. The launcher now rejects the argument first (exit 2) and points at the
  `portopt` stack commands, and `fund-tui` reports a bad portfolio id as a usage
  error instead of a traceback. The launcher accepts a UUID written with hyphens
  (8-4-4-4-12) or as 32 hex digits, in any case, and rejects everything else: an
  argument that starts with `-`, more than one argument, and the `{...}` and
  `urn:uuid:` spellings. So `optimizer --help` now exits 2 with the same message
  instead of starting the stack.
- **fund, ingestion, portopt-db**: API keys, the broker secret, database
  passwords and the notification webhook URL no longer appear in `repr()` or
  `str()` of `FundConfig`, `DbConfig` or the ingestion `Settings`. A traceback that renders locals or a stray
  `logger.info(config)` used to print every provider key and the DB URL.
- **fund**: the risk gate and the weighting tools no longer judge a book on its
  youngest holding. The drawdown ceiling, `optimize_portfolio` and
  `estimate_moments` share one sample-depth rule (a fit needs at least half the
  median own history); an unverifiable drawdown now blocks instead of warning;
  `universe_filter` drops only assets that are thin on their own and no longer
  loses any asset that starts a bar after the oldest. A risk verdict is
  re-checked against the current constraint set before orders are placed, and the
  ESG look-through fails closed on non-finite weights, counts unreported weight
  as possible exposure at a strict tolerance, and ignores snapshots dated after
  the decision bar.
- **fund**: the risk gate compares the client's drawdown ceilings with the worst
  loss inside any one-year window of the history, because the stated tolerance is a
  one-year figure (the full-history drawdown is still reported). A common history
  shorter than one year blocks with `drawdown_unverified`, and a stated tolerance of
  zero is reported as `zero_loss_tolerance` instead of an opaque breach. Bars per
  year are derived from each panel's calendar (about 261 on a weekday union of
  venues, not a fixed 252) for moments, the optimizer, the risk-free rate and the
  Performance strip, so annualised figures move by up to about 3%. The profile's
  `moments_estimator` now reaches the optimizer's prior, so the covariance the PM
  reads is the covariance `optimize_portfolio` solves under. A legacy percent-style
  drift threshold that was clamped to "never fires" is shown as a warning on save,
  in `fund mandate show` and on the Posture view.
- **fund**: NAV starts at capital when the first ticket leaves cash undeployed
  (a later ticket used to lever the returns), a re-approved identical ticket
  becomes the active book again, holding contributions sum to the NAV return, a
  held currency with no FX rate fails loudly, and the NAV upsert drops rows the
  rebuilt series no longer emits.
- **fund**: an approve, edit or Ask approve resolves the gate the adviser was shown:
  the Decisions view remembers the id of the gate it rendered and the decision is
  refused (`GateMovedError`) when the run has since moved to another gate.
  `tradeable_only` keeps an allowlisted name the Trading 212 overlay could not judge
  (no ISIN) but names it in the PM seed and in the mandate-save warnings instead of
  letting it through silently.
- **fund**: approving a second gate on one run works (the idempotency key now
  names the gate, not the run), a database error inside a tool no longer rolls
  back the round's audit rows, a failed round is recorded after its rollback and
  its checkpoint thread deleted, mandate saves detect a concurrent change
  (`MandateConflictError`), and the first-round flag no longer blocks a freeze.
- **fund**: `OLLAMA_API_KEY` is sent only to `ollama.com` over https; a keyed
  gateway elsewhere needs `OLLAMA_BASE_URL_SEND_KEY=true`. The LangGraph
  bootstrap lock polls `pg_try_advisory_lock`, so two processes starting on a
  fresh database can no longer deadlock behind `CREATE INDEX CONCURRENTLY`.
- **ingestion**: `t212_annotate` judges mandate-pinned instruments, so
  `tradeable_only` bars a pinned name the broker does not list; the broker scope
  ignores delisted twin rows; a mandate save no longer revives delisted sibling
  rows or erases `delisting_return`; an empty Trading 212 catalogue no longer
  clears stored tickers; "Gold Explorers" and natural-gas equity funds are no
  longer labelled commodity (they skip the ESG look-through) and minimum-volatility
  equity ETFs are no longer dropped as volatility products.
- **ingestion**: a failed Yahoo fetch is no longer cached as "no data" for 72
  hours; the news search fallback goes through the shared rate limiter and
  circuit breaker; the news sweep reads only the ids it needs; an options sweep
  through an outage fails instead of reporting zero errors; a full-history fetch
  stops before today's unfinished bar; a delisted instrument with no bar for a
  year leaves the daily sweep. Migration `b8d0f2a4c6e1` turns JSON `null` values
  in `background_jobs` and `portfolio_journal` into SQL `NULL`.
- **ingestion**: `ticker_news` is pruned weekly (`NEWS_RETENTION_DAYS`, default
  365, 0 disables) in bounded batches as its own job slot; a prune error is
  recorded on that slot and never fails the refetch. The directly imported
  `certifi`, `urllib3`, `prompt-toolkit` and `python-dateutil` are declared in
  `ingestion/pyproject.toml`, with a guard test.
- **ops**: `install.sh` installs the CLI against constraints exported from the
  checkout's `uv.lock` (a failed pinned install is retried without them), and
  `install.ps1` now updates an existing checkout (fast-forward only, refusing a
  checkout of another repo) and stops when it cannot, as `install.sh` does. The CI
  boot test also starts and stops the fund container on a placeholder provider key,
  and a `windows-install` job runs `install.ps1` against stub git, uv and portopt on
  a Windows runner under Windows PowerShell 5.1 and PowerShell 7.
- **ops**: on a Linux host the fund container runs as the invoking user
  (`PORTOPT_FUND_UID` / `PORTOPT_FUND_GID`, handed to every compose call and
  defaulting to the image's 1000), so TUI exports to `./exports` are writable and
  readable by that user; macOS, a uid-1000 host and root keep the image's user. A
  leftover root-owned `./exports` is reported with the command that fixes it.
- **ops**: `portopt start` rebuilds an image whose source-hash label no longer
  matches the checkout (the first start after upgrading rebuilds once, because
  older images carry no label), so old containers no longer run against a newly
  migrated database. The ingestion entrypoint handles SIGTERM during boot,
  `portopt status` reports a deliberately disabled fund and times out docker
  calls, `./exports` is created for the invoking user, `install.sh` stops when
  the checkout cannot be updated, and the LLM placeholder keys in `.env.example`
  are commented out.
- **docker**: `docker-compose.yml` now pins the project name with a top-level
  `name: optimizer`. Without it Compose took the name from the checkout
  directory, so re-cloning or unzipping under another basename (for example
  `optimizer-development`) created a new, empty `<basename>_postgres_data`
  volume: the database looked wiped and a full refetch started. The pin matches
  the `optimizer_postgres_data` volume of installs cloned as `optimizer` (what
  the README says to do), which need no action. **Upgrade note — checkouts
  under any other name:** their data lives in `<basename>_postgres_data`, and
  the old containers still hold the fixed `container_name`s (`optimizer_db`,
  …), so the next `portopt start` would either fail with a name conflict or
  come up on an empty database. Before upgrading, either
  - keep the old project: set `COMPOSE_PROJECT_NAME=<basename>` (it overrides
    `name:`, as does `-p`) and nothing moves; or
  - move to the pinned name: run `docker compose -p <basename> --profile
    ingestion --profile fund --profile tools down` (it frees the container names
    and keeps the volume; never use `down -v`, which deletes the database), then
    copy the data across with `docker volume create optimizer_postgres_data &&
    docker run --rm -v <basename>_postgres_data:/from -v
    optimizer_postgres_data:/to postgres:16-alpine cp -a /from/. /to/`. Compose
    may warn that it did not create `optimizer_postgres_data`; it still uses it
- **api**: synchronous scheduler steps now run a heartbeat companion. Only the
  heartbeat stamps `last_heartbeat_at`, so a step running in the scheduler thread
  went silent to the orphan reaper and was failed mid-run once it outlived the
  300s timeout. The reference-index seed (~12 minutes) was reaped at 5/12 tickers
  while still fetching successfully — the row said `failed`, the data said fine,
  and nothing reconciled the two
- **api**: the Il Sole scraper now pins `Accept-Encoding: gzip, deflate` instead of
  inheriting requests' default, which silently becomes `gzip, deflate, br` as soon
  as brotli is importable anywhere on `sys.path`. That is the same undecodable-body
  failure that froze the Trading Economics table for 16 days; TE was pinned
  afterwards, Il Sole never was
- **build**: `mypy` was configured for Python 3.10, so it aborted on numpy's PEP 695
  stubs before reaching any project file — the typecheck job had been passing while
  checking nothing. It now checks all 98 source files
- **core**: `benjamini_hochberg` delegates to `correct_pvalues`, so a NaN p-value is
  left out of the family size instead of shrinking every threshold. With
  `[0.02, 0.04, NaN]` it used to mark nothing significant; it now marks the first
  two, as `correct_pvalues` always did
- **ingestion**: a Trading 212 `Retry-After` is capped at the same ceiling as
  `x-ratelimit-reset` (120 s) and a negative value falls back to the backoff, so a
  server-sent hour-scale value can no longer stall the universe build or make
  `time.sleep` raise
- **ingestion**: `retry_with_backoff` logs a WARNING with the last exception (or
  "result rejected by is_valid") when it gives up. Its callers only report "failed
  after retries", so the cause used to be lost. The text is passed through
  `redact_secrets` and no traceback is attached, because FRED sends its key as
  `?api_key=` and `requests` echoes the URL in its errors

## [0.1.0] - 2026-02-21

### Added

- **preprocessing**: Data validation, outlier treatment, sector imputation, regression imputation, delisting adjustment
- **pre_selection**: Pipeline assembly with skfolio selectors (SelectComplete, DropZeroVariance, DropCorrelated, SelectKExtremes, SelectNonDominated, SelectNonExpiring)
- **moments**: Expected return and covariance estimation (empirical, shrinkage, denoised), HMM regime blending, DMM via Pyro, lognormal scaling
- **views**: Black-Litterman, Entropy Pooling, Opinion Pooling integration frameworks
- **optimization**: Mean-Risk, Risk Budgeting, HRP, HERC, NCO, Maximum Diversification, Benchmark Tracking, Equal-Weighted, Inverse-Volatility, Stacking; robust and distributionally robust variants; regime-conditional risk
- **synthetic**: Vine copula models, synthetic data generation, conditional stress testing
- **validation**: Walk-Forward, Combinatorial Purged CV, Multiple Randomized CV
- **scoring**: Performance scoring for model selection
- **tuning**: Grid search and randomized search with temporal cross-validation
- **rebalancing**: Calendar-based, threshold-based, and hybrid rebalancing; turnover and cost computation
- **pipeline**: End-to-end orchestration from prices to validated weights
- **universe**: Investability screening with hysteresis-based entry/exit thresholds
- **factors**: Factor construction, standardization, composite scoring, stock selection, regime tilts, validation, mimicking portfolios, integration with optimization

[Unreleased]: https://github.com/SilvioBaratto/optimizer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/SilvioBaratto/optimizer/releases/tag/v0.2.0
[0.1.0]: https://github.com/SilvioBaratto/optimizer/releases/tag/v0.1.0
