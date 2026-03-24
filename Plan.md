# Antigravity Execution Plan — Production-Ready CodeForge/Loveable Clone

This document is written so **Antigravity** can execute implementation with minimal ambiguity.
It is a concrete engineering plan for integrating:

- FastAPI production backend
- High-quality interactive frontend
- OAuth2 login (Google + GitHub)
- Multi-session support per user
- Real-time streaming UX
- Human-in-the-loop (HITL) approval after architect node
- Episodic + long-term memory
- Input/output safety validation
- Cost/usage monitoring and budget controls
- Clean project structure and required `.env` keys

---

## 0) Repository Baseline (Current State)

Current code already contains useful foundations:

- Dependency baseline includes FastAPI, LangChain, LangGraph, ChromaDB, SQLAlchemy and Uvicorn.
- Graph flow exists and currently is:
  `planner -> architect -> coder (loop) -> reviewer -> file_collector -> downloader`.
- Structured outputs and review loop patterns already exist.

Antigravity should **build on this**, not replace it blindly.

---

## 1) HITL Flow Requirement (Exact Behavior)

### Goal
Insert a human approval checkpoint **immediately after architecture generation**.

### Required behavior
1. `planner` runs.
2. `architect` returns `task_plan`.
3. Graph enters `await_human_approval` node.
4. Backend marks state: `status=AWAITING_APPROVAL`.
5. Frontend shows architecture plan and two actions:
   - **Accept** -> continue directly to `coder`.
   - **Reject & Edit** -> open editable task list UI.
6. If rejected:
   - user edits `implementation_steps` manually,
   - user clicks **Done Editing**,
   - edited `task_plan` is saved,
   - graph resumes at `coder` with edited plan.

### Graph changes (in this repo)
- Update `Agents/State.py` to add:
  - `approval_status: str` (`PENDING | ACCEPTED | REJECTED | EDITED`)
  - `awaiting_user_input: bool`
  - `edited_task_plan: TaskPlan | None`
  - `session_id: str`
  - `user_id: str`
- Update `Agents/Workflow.py`:
  - Add new node: `hitl_gate`
  - New edges:
    - `architect -> hitl_gate`
    - `hitl_gate -> coder` when accepted or edited
    - `hitl_gate -> END_WAIT` equivalent pause behavior when pending
- Implement resumable execution API in backend:
  - `POST /api/v1/runs/{run_id}/approve`
  - `POST /api/v1/runs/{run_id}/reject`
  - `POST /api/v1/runs/{run_id}/task-plan` (save edited steps)
  - `POST /api/v1/runs/{run_id}/resume`

### HITL frontend requirements
- Architecture Review Panel:
  - task list table
  - dependency visualization
  - estimated effort / token cost
- Buttons:
  - Accept
  - Reject & Edit
  - Done Editing
- Validation for edits:
  - step order must be deterministic
  - each step must include `filepath` and `task_description`

---

## 2) Memory System (CodeForge Model) — Integration into This Project

Use 3-tier memory exactly:

1. **Working memory**: GraphState per run (ephemeral + checkpointed)
2. **Episodic memory**: PostgreSQL factual run history
3. **Long-term memory**: semantic lessons via mem0 + pgvector

### 2.1 Working memory (GraphState additions)
Add fields in `Agents/State.py`:

- `user_prompt: str`
- `user_id: str`
- `session_id: str`
- `thread_id: str`
- `task_plan: TaskPlan`
- `past_review_lessons: str`
- `user_project_history: str`
- `review_result: ReviewResult`
- `retry_count: int`
- `status: str` (`IN_PROGRESS | AWAITING_APPROVAL | DONE | FAILED`)
- `error: str | None`

### 2.2 Episodic memory write point
Write episodic memory right after `file_collector` has complete artifacts.

Persist:
- project/run metadata
- review result summary and issues JSON
- generated file index + optionally content table
- retry count and timing metrics

### 2.3 Episodic read point
At generation start, fetch last N user projects and inject summary into planner context (`user_project_history`).

### 2.4 Long-term memory write point
After reviewer completes, convert each issue into a sentence lesson and store in mem0 with `user_id` scope.

### 2.5 Long-term memory read point
Before coder node starts, retrieve relevant lessons by semantic search using current prompt.
Inject into coder prompt as `past_review_lessons`.

### 2.6 Hard rules
- Never store mem0 lessons without `user_id`.
- Never raise fatal exception if memory write fails.
- Lessons from run N are only applied in run N+1.

---

## 3) Authentication + Authorization (Google/GitHub OAuth2)

### Required auth UX
- User can log in with:
  - Google
  - GitHub
- After login, user can create multiple sessions.
- Session list is visible in sidebar.
- User can create/select/archive session.

### Backend auth architecture
- FastAPI OAuth2 Authorization Code + PKCE.
- Identity provider options:
  - Auth0/Clerk/Cognito/Keycloak (pick one and keep abstraction layer).
- Persist local user table with:
  - provider (`google`/`github`)
  - provider_user_id
  - email
  - display_name
  - avatar_url
- JWT/session handling:
  - short-lived access token
  - refresh token rotation
  - secure HttpOnly cookies for web app

### RBAC scopes
- `runs:create`
- `runs:approve`
- `runs:edit_plan`
- `runs:view`
- `usage:view`
- `admin:billing`

---

## 4) Sessions Model (Per User Multi-Session)

### Data model
- `users`
- `sessions`
- `runs`
- `messages`
- `approvals`
- `usage_events`

### Rules
- One user -> many sessions.
- One session -> many runs.
- Session context is isolated.
- Run execution always references `(user_id, session_id, run_id)`.

### Frontend UX
- Left sidebar session switcher.
- “New Session” modal with name + optional template.
- Session activity badges (running, awaiting approval, completed, failed).

---

## 5) Real-Time Streaming

### Required behavior
- User sees live generation progress:
  - current node
  - tokens/cost so far
  - warnings/policy checks
  - file-by-file completion logs

### Transport choice
- Preferred: Server-Sent Events for one-way stream simplicity.
- Optional: WebSocket for bidirectional controls.

### Events contract
Emit structured events:
- `run.started`
- `node.started`
- `node.completed`
- `hitl.required`
- `usage.updated`
- `review.completed`
- `run.completed`
- `run.failed`

Each event should include:
- `run_id`
- `session_id`
- `timestamp`
- `payload`

---

## 6) Safety System (Input + Output Validation)

### Input safety
- Pydantic strict request models (length, regex, enums, bounds).
- Prompt injection pre-filter.
- Secret/PII detector on user input.
- Upload restrictions (MIME, extension, size).
- Per-user and per-session rate limits.

### Output safety
- Structured output enforcement with retries.
- Output moderation gate.
- Secret leakage scan before returning response/files.
- Dangerous action guard requires explicit HITL confirmation.

### Security extras
- CSRF protections for cookie-auth endpoints.
- CORS locked to frontend domains.
- Dependency and container scanning in CI.

---

## 7) Cost Monitoring + Usage

Track per model invocation:
- input tokens
- output tokens
- cached tokens (if applicable)
- latency
- estimated USD cost
- tool call counts

Budget controls:
- per-run caps
- daily per-user caps
- monthly org caps
- graceful downgrade model policy

Dashboard widgets:
- cost by day
- cost by model
- top expensive sessions
- approval impact on cost

---

## 8) Frontend Quality Bar ("very very good and interactive")

Use React + TypeScript + Next.js (recommended) with design system.

### Core pages
- Login
- Workspace (chat + generation timeline)
- Session manager
- Approval center
- Usage dashboard
- Project history & memory insights

### Interaction standards
- optimistic UI where safe
- skeleton loaders
- streamed token-by-token logs
- keyboard shortcuts
- undo/redo in plan editor
- accessible components (WCAG AA)

### Visual standards
- responsive layout
- dark/light mode
- clear status chips and trace timeline
- diff viewer for edited task plan

---

## 9) Suggested Project Structure

```text
project-root/
  backend/
    app/
      api/
        v1/
          auth.py
          sessions.py
          runs.py
          approvals.py
          usage.py
      core/
        config.py
        security.py
        langgraph/
          state.py
          workflow.py
          nodes/
            planner.py
            architect.py
            hitl_gate.py
            coder.py
            reviewer.py
            file_collector.py
            downloader.py
          memory/
            episodic.py
            long_term.py
      models/
        user.py
        session.py
        run.py
        review.py
        usage.py
      services/
        auth_service.py
        database_service.py
        usage_service.py
        memory_service.py
      main.py
    migrations/
  frontend/
    src/
      app/
      components/
      features/
        auth/
        sessions/
        runs/
        approvals/
        usage/
      lib/
        api-client.ts
        stream-client.ts
        validators.ts
```

---

## 10) Required `.env` Keys

Antigravity must output a `.env.example` with these keys:

```bash
# App
APP_ENV=development
APP_NAME=CodeForge
APP_URL=http://localhost:3000
API_URL=http://localhost:8000

# Security
SECRET_KEY=change_me
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=30
COOKIE_SECURE=false
COOKIE_DOMAIN=localhost

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=codeforge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/codeforge

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI / LLM
OPENAI_API_KEY=
DEFAULT_LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

# Mem0 / Vector
MEM0_PROVIDER=pgvector
MEM0_COLLECTION=codeforge_lessons
PGVECTOR_DIMENSIONS=1536

# OAuth - Google
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback

# OAuth - GitHub
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GITHUB_REDIRECT_URI=http://localhost:8000/api/v1/auth/github/callback

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=
SENTRY_DSN=
LOG_LEVEL=INFO

# Billing / Usage
DAILY_USER_BUDGET_USD=5
MONTHLY_ORG_BUDGET_USD=200
RUN_TOKEN_LIMIT=200000
```

---

## 11) Backend API Contract (Minimum)

- `POST /api/v1/auth/google/login`
- `GET /api/v1/auth/google/callback`
- `POST /api/v1/auth/github/login`
- `GET /api/v1/auth/github/callback`
- `POST /api/v1/auth/logout`
- `GET /api/v1/me`

- `POST /api/v1/sessions`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`

- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/stream` (SSE)

- `POST /api/v1/runs/{run_id}/approve`
- `POST /api/v1/runs/{run_id}/reject`
- `PUT /api/v1/runs/{run_id}/task-plan`
- `POST /api/v1/runs/{run_id}/resume`

- `GET /api/v1/usage/summary`
- `GET /api/v1/usage/events`

---

## 12) Delivery Phases for Antigravity

### Phase 1 — Foundation
- FastAPI app scaffold, DB, auth skeleton, session CRUD.

### Phase 2 — LangGraph Integration
- Move existing nodes into backend service layer.
- Add HITL pause/resume node after architect.

### Phase 3 — Memory
- Episodic writes and reads.
- mem0 long-term lesson write/read.

### Phase 4 — Frontend
- Workspace, session switcher, architecture editor, streaming timeline.

### Phase 5 — Safety + Cost + Hardening
- Validation gates, usage dashboards, quotas, SLO alerts, tests.

### Phase 6 — Production readiness
- CI/CD, migrations, backups, runbooks, incident playbooks.

---

## 13) Acceptance Criteria (Must Pass)

1. User can sign in with Google or GitHub.
2. User can create multiple sessions and switch among them.
3. Graph pauses after architect and waits for user decision.
4. Reject path allows full manual task-step editing, then resume.
5. Streaming updates are visible in realtime in frontend.
6. Episodic memory persists every completed run.
7. Long-term lessons are retrieved and injected into coder prompt on later runs.
8. Usage/cost dashboard shows per-run and aggregate metrics.
9. Input/output validation blocks unsafe traffic.
10. `.env.example` includes all required keys and boot instructions.

---

## 14) Direct Instruction to Antigravity

Implement exactly in this order:

1. Introduce backend project structure and auth/session models.
2. Integrate existing graph with new `hitl_gate` node after architect.
3. Build pause/resume APIs and architecture edit APIs.
4. Add SSE streaming for run events.
5. Implement episodic + long-term memory write/read hooks.
6. Add safety validators and usage metering middleware.
7. Build polished frontend UX (sessions + approvals + streaming + dashboards).
8. Provide `.env.example`, `docker-compose.yml`, and setup docs.

Do not ship partial memory wiring or partial HITL — both must be end-to-end functional.
