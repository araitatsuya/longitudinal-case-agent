"""FastAPI app: POST events in, read plans and the audit trail out, serve the UI.

Run:  uvicorn app.main:app --reload
Env:  PLANNER=openai|mock (default: openai if a key is found, else mock)
      OPENAI_API_KEY or apikey.txt, OPENAI_MODEL (default gpt-5.5), DATA_DIR (default ./data)
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .agent import CaseAgent, DuplicateEventError
from .models import Event
from .planner import MockPlanner, OpenAIPlanner, Planner, PlannerError, load_openai_key
from .store import Store

STATIC_DIR = Path(__file__).parent / "static"


def planner_from_env() -> Planner:
    choice = os.environ.get("PLANNER") or ("openai" if load_openai_key() else "mock")
    if choice == "mock":
        return MockPlanner()
    if choice == "openai":
        return OpenAIPlanner(model=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    raise ValueError(f"Unknown PLANNER={choice!r}; use 'openai' or 'mock'")


def create_app(data_dir: Path | None = None, planner: Planner | None = None) -> FastAPI:
    store = Store(data_dir or Path(os.environ.get("DATA_DIR", "data")))
    agent = CaseAgent(store, planner or planner_from_env())
    app = FastAPI(title="Longitudinal Case Agent")

    def require_patient(patient_id: str) -> None:
        if not store.patient_exists(patient_id):
            raise HTTPException(404, f"No patient '{patient_id}'")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/events", status_code=201)
    def post_event(event: Event):
        try:
            plan, entry = agent.handle_event(event)
        except DuplicateEventError as e:
            raise HTTPException(409, str(e))
        except PlannerError as e:
            raise HTTPException(502, str(e))
        return {"plan_version": plan.version, "audit": entry, "plan": plan}

    @app.get("/api/status")
    def status():
        return {"planner": agent.planner.name, "patients": store.list_patients()}

    @app.get("/api/patients/{patient_id}")
    def patient(patient_id: str):
        require_patient(patient_id)
        plans = store.plans(patient_id)
        return {
            "patient_id": patient_id,
            "current_plan": plans[-1] if plans else None,
            "plans": plans,
            "audit": store.audit(patient_id),
            "events": store.events(patient_id),
        }

    @app.get("/api/patients/{patient_id}/audit")
    def audit(patient_id: str):
        require_patient(patient_id)
        return store.audit(patient_id)

    @app.delete("/api/patients/{patient_id}")
    def reset(patient_id: str):
        """Demo helper: wipe a patient so the story can be replayed."""
        require_patient(patient_id)
        store.delete_patient(patient_id)
        return {"deleted": patient_id}

    return app


app = create_app()
