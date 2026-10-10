# Longitudinal Case Agent

An AI agent that follows one patient over several days. Each time a new event comes in (admission, lab results, culture results), it:

1. Remembers everything it already knows about the patient, from its own files on disk
2. Asks OpenAI (`gpt-5.5`) to update the care plan (v1 → v2 → v3 …)
3. Saves the new plan version without overwriting the old ones
4. Adds an entry to the **audit trail** recording what came in, what changed, and why

The web page shows the current plan, every past version, and the audit trail. It updates automatically.

> ⚠️ **The patient is fictional, and this is a demo only. It is not medical advice.**

## The story

| File | Event | Expected plan |
|---|---|---|
| `inbox/day1_admit.json` | Day 1: Sam (45) admitted with fever and cough | **v1**: suspected pneumonia; order tests, fluids, fever medicine |
| `inbox/day2_labs.json` | Day 2: high white cell count, CRP and procalcitonin; X-ray shows a right-lung spot | **v2**: bacterial pneumonia; start antibiotics |
| `inbox/day3_culture.json` | Day 3: culture finds *S. pneumoniae*; Sam is improving | **v3**: switch to targeted oral amoxicillin; plan discharge |

Each file contains **only the new information**. Day 2 doesn't repeat Sam's age or symptoms, so the agent has to remember them.

---

## Run it

You need **Python 3.10+** and two terminal windows.

### Terminal 1: start the server

```bash
cd /Users/tatsuyaarai/workspace/cal_hackathon_B

# First time only: create the virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Every time: activate and start
source .venv/bin/activate
uvicorn app.main:app --reload --port 8080
```

Leave this window running. You should see `Uvicorn running on http://127.0.0.1:8080`.

The OpenAI key is read automatically from `apikey.txt` in the project folder (or from `OPENAI_API_KEY` if you set it).

> **No API key or offline?** Run `PLANNER=mock uvicorn app.main:app --reload --port 8080` instead.
> The mock returns pre-written plans for the three demo events, so the full flow works without the API.
> If no key is found, the app uses the mock automatically.

### Browser: open the UI

Open **http://127.0.0.1:8080**

It shows "Waiting for the first event…" until you send one.

### Terminal 2: send the events one by one

Open a **new** terminal window and paste the commands one at a time. Watch the browser after each one.

```bash
cd /Users/tatsuyaarai/workspace/cal_hackathon_B
```

**Day 1: Admission** (creates plan v1)
```bash
curl -X POST http://127.0.0.1:8080/events -H "Content-Type: application/json" -d @inbox/day1_admit.json
```

**Day 2: New labs** (plan v1 → v2)
```bash
curl -X POST http://127.0.0.1:8080/events -H "Content-Type: application/json" -d @inbox/day2_labs.json
```

**Day 3: Culture results** (plan v2 → v3)
```bash
curl -X POST http://127.0.0.1:8080/events -H "Content-Type: application/json" -d @inbox/day3_culture.json
```

With OpenAI, each call takes about 10–20 seconds while the model writes the plan. The mock answers instantly.

### Useful extra commands (Terminal 2)

```bash
# See the audit trail as JSON
curl http://127.0.0.1:8080/api/patients/sam-001/audit

# See everything: current plan, all versions, audit, raw events
curl http://127.0.0.1:8080/api/patients/sam-001

# Which planner is running (OpenAI model name or "mock")
curl http://127.0.0.1:8080/api/status

# Start the story over (deletes Sam's data); the UI also has a Reset button
curl -X DELETE http://127.0.0.1:8080/api/patients/sam-001
```

Sending the same file twice returns **409 Conflict**, so an event is never processed twice.

### Show persistence

1. Send Day 1 and Day 2.
2. Stop the server in Terminal 1 with `Ctrl+C`, then start it again with the same `uvicorn` command.
3. Refresh the browser. Plan v2 and the audit trail are still there.
4. Send Day 3. The agent continues from v2 to v3, because its memory is on disk and not in the chat.

The files are in `data/patients/sam-001/`:

```
data/patients/sam-001/
├── events/        0001_admission.json, 0002_lab_results.json, ...   (raw events as received)
├── plans/         plan_v1.json, plan_v2.json, plan_v3.json         (never overwritten)
└── audit.jsonl    one line per change: what came in, from → to version, changes, why
```

---

## Run the tests

```bash
source .venv/bin/activate
pytest -v
```

The tests use the mock planner, so they run offline, cost nothing, and take under a second. They cover:

- the full Day 1 → 2 → 3 story producing v1 → v2 → v3 with matching audit entries
- old plan versions never changing
- **restart persistence** (a new app instance on the same data folder continues from where it left off)
- the agent getting Day 1 history from disk when the Day 2 POST only contains labs
- duplicate events (409), invalid events (422), planner failure (502, nothing recorded), unknown patient (404), reset

**Optional live test** against the real OpenAI API (uses `apikey.txt`, costs a few cents):

```bash
RUN_LIVE=1 pytest -v -k live
```

---

## How it works

```
curl POST /events  (only the new event)
        │
        ▼
app/main.py     FastAPI routes
app/agent.py    CaseAgent.handle_event():
                  1. reject duplicates
                  2. load event history + current plan from disk   ← memory
                  3. ask the planner for an updated plan
                  4. save event, save plan_vN, append audit entry
app/planner.py  OpenAIPlanner (OpenAI API, structured output) or MockPlanner
app/store.py    reads/writes data/patients/<id>/...
app/static/index.html   the UI (polls the API every 2 seconds)
```

The OpenAI call uses `gpt-5.5` with structured outputs, so the reply always matches the plan schema in `app/models.py`. The prompt includes the full event history, the current plan, and the new event.

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | read from `apikey.txt` | Your OpenAI API key (overrides the file) |
| `OPENAI_API_KEY_FILE` | `apikey.txt` | Where to read the key from |
| `PLANNER` | `openai` if a key is found, else `mock` | `openai` or `mock` |
| `OPENAI_MODEL` | `gpt-5.5` | Which OpenAI model plans (e.g. `gpt-5.4-mini` for faster/cheaper) |
| `DATA_DIR` | `data` | Where the agent stores its memory |

### API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/events` | Process one new event → returns the new plan + audit entry |
| `GET` | `/api/patients/{id}` | Current plan, all versions, audit trail, raw events |
| `GET` | `/api/patients/{id}/audit` | Audit trail only |
| `DELETE` | `/api/patients/{id}` | Reset a patient (demo helper) |
| `GET` | `/api/status` | Planner name and list of patients |
| `GET` | `/` | Web UI |

An event needs only `patient_id` and `event_type`. Every other field is passed to the planner as-is, so you can write new event files (for example, a Day 4 nurse note) without changing any code.
