import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import Plan, PlanDraft
from app.planner import MockPlanner, OpenAIPlanner, PlannerError, build_user_message, load_openai_key

PATIENT = "sam-001"


def post(client, event):
    return client.post("/events", json=event)


# ---- the three-day story -------------------------------------------------


def test_admission_creates_plan_v1(client, day1):
    r = post(client, day1)
    assert r.status_code == 201
    body = r.json()
    assert body["plan_version"] == 1
    assert body["audit"]["from_version"] is None
    assert body["audit"]["to_version"] == 1
    assert "pneumonia" in body["plan"]["working_diagnosis"].lower()


def test_full_story_v1_to_v3(client, day1, day2, day3):
    for event in (day1, day2, day3):
        assert post(client, event).status_code == 201

    data = client.get(f"/api/patients/{PATIENT}").json()
    assert [p["version"] for p in data["plans"]] == [1, 2, 3]
    assert data["current_plan"]["version"] == 3
    assert "amoxicillin" in " ".join(data["current_plan"]["medications"]).lower()

    audit = data["audit"]
    assert [(a["from_version"], a["to_version"]) for a in audit] == [(None, 1), (1, 2), (2, 3)]
    assert [a["event_type"] for a in audit] == ["admission", "lab_results", "culture_results"]
    assert all(a["rationale"] and a["changes"] for a in audit)
    assert [a["seq"] for a in audit] == [1, 2, 3]


def test_old_plan_versions_are_kept_unchanged(client, day1, day2):
    post(client, day1)
    v1_before = client.get(f"/api/patients/{PATIENT}").json()["plans"][0]
    post(client, day2)
    v1_after = client.get(f"/api/patients/{PATIENT}").json()["plans"][0]
    assert v1_before == v1_after


# ---- persistence: memory lives on disk -----------------------------------


def test_state_survives_restart(data_dir, day1, day2):
    first = TestClient(create_app(data_dir=data_dir, planner=MockPlanner()))
    post(first, day1)

    # Simulate a server restart: brand-new app object, same data folder.
    second = TestClient(create_app(data_dir=data_dir, planner=MockPlanner()))
    assert second.get(f"/api/patients/{PATIENT}").json()["current_plan"]["version"] == 1
    r = post(second, day2)
    assert r.json()["plan_version"] == 2
    assert r.json()["audit"]["from_version"] == 1


def test_files_written_to_disk(client, data_dir, day1, day2):
    post(client, day1)
    post(client, day2)
    folder = data_dir / "patients" / PATIENT
    assert sorted(p.name for p in (folder / "plans").iterdir()) == ["plan_v1.json", "plan_v2.json"]
    assert sorted(p.name for p in (folder / "events").iterdir()) == ["0001_admission.json", "0002_lab_results.json"]
    lines = (folder / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["to_version"] == 2


class SpyPlanner(MockPlanner):
    def __init__(self):
        self.calls = []

    def make_plan(self, history, current, new_event):
        self.calls.append((history, current, new_event))
        return super().make_plan(history, current, new_event)


def test_agent_remembers_history_from_disk(data_dir, day1, day2):
    spy = SpyPlanner()
    client = TestClient(create_app(data_dir=data_dir, planner=spy))
    post(client, day1)
    post(client, day2)

    history, current, new_event = spy.calls[1]
    # The Day 2 POST only carried labs; the agent supplied Day 1 and plan v1 itself.
    assert "patient" not in day2
    assert history == [day1]
    assert history[0]["patient"]["name"] == "Sam Rivera"
    assert current.version == 1
    assert new_event["event_type"] == "lab_results"


# ---- error handling ------------------------------------------------------


def test_duplicate_event_rejected(client, day1):
    assert post(client, day1).status_code == 201
    r = post(client, day1)
    assert r.status_code == 409
    assert len(client.get(f"/api/patients/{PATIENT}").json()["plans"]) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"event_type": "admission"},                       # missing patient_id
        {"patient_id": "sam-001"},                         # missing event_type
        {"patient_id": "../etc", "event_type": "x"},       # unsafe id
    ],
)
def test_invalid_event_rejected(client, body):
    assert post(client, body).status_code == 422


class FailingPlanner:
    name = "failing"

    def make_plan(self, history, current, new_event):
        raise PlannerError("boom")


def test_planner_failure_records_nothing(data_dir, day1):
    client = TestClient(create_app(data_dir=data_dir, planner=FailingPlanner()))
    r = post(client, day1)
    assert r.status_code == 502
    assert client.get(f"/api/patients/{PATIENT}").status_code == 404


def test_unknown_patient_404(client):
    assert client.get("/api/patients/nobody").status_code == 404


def test_reset_deletes_patient(client, day1):
    post(client, day1)
    assert client.delete(f"/api/patients/{PATIENT}").status_code == 200
    assert client.get(f"/api/patients/{PATIENT}").status_code == 404
    assert post(client, day1).json()["plan_version"] == 1


def test_unknown_event_type_carries_plan_forward(client, day1):
    post(client, day1)
    r = post(client, {"patient_id": PATIENT, "event_type": "nurse_note", "note": "slept well"})
    assert r.status_code == 201
    assert r.json()["plan_version"] == 2
    assert r.json()["plan"]["medications"] == client.get(f"/api/patients/{PATIENT}").json()["plans"][0]["medications"]


# ---- UI + status ---------------------------------------------------------


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Case Agent" in r.text


def test_status(client, day1):
    post(client, day1)
    assert client.get("/api/status").json() == {"planner": "mock", "patients": [PATIENT]}


# ---- prompt construction -------------------------------------------------


def test_prompt_contains_history_plan_and_new_event(client, day1, day2):
    post(client, day1)
    data = client.get(f"/api/patients/{PATIENT}").json()
    current = Plan(**data["current_plan"])
    msg = build_user_message([day1], current, day2)
    assert "Sam Rivera" in msg            # from history
    assert '"version": 1' in msg          # previous plan
    assert "Procalcitonin" in msg         # new event
    assert "first event" not in msg


def test_prompt_for_first_event(day1):
    assert "first event" in build_user_message([], None, day1)


def test_missing_openai_key_gives_clear_error(monkeypatch, tmp_path, day1):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing.txt"))
    with pytest.raises(PlannerError, match="No OpenAI key"):
        OpenAIPlanner().make_plan([], None, day1)


def test_key_read_from_file(monkeypatch, tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-test-123\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))
    assert load_openai_key() == "sk-test-123"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert load_openai_key() == "sk-env"


# ---- optional live test against the real OpenAI API ------------------------


@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1 to call the real OpenAI API (uses apikey.txt)")
def test_live_openai_planner(day1, day2):
    planner = OpenAIPlanner(model=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    v1 = planner.make_plan([], None, day1)
    assert isinstance(v1, PlanDraft) and v1.working_diagnosis
    current = Plan(**v1.model_dump(), version=1, patient_id=PATIENT, created_at="", trigger_event_id="0001_admission", planner=planner.name)
    v2 = planner.make_plan([day1], current, day2)
    assert v2.changes and v2.rationale
