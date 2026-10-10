"""The agent step: one new event in -> one new plan version + one audit entry out."""

import threading
from datetime import datetime, timezone

from .models import AuditEntry, Event, Plan
from .planner import Planner
from .store import Store


class DuplicateEventError(Exception):
    def __init__(self, event_id: str):
        super().__init__(f"This exact event was already processed as {event_id}")
        self.event_id = event_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseAgent:
    def __init__(self, store: Store, planner: Planner):
        self.store = store
        self.planner = planner
        # One event at a time, so plan versions never race.
        self._lock = threading.Lock()

    def handle_event(self, event: Event) -> tuple[Plan, AuditEntry]:
        payload = event.model_dump()
        patient_id = event.patient_id
        with self._lock:
            dup = self.store.find_event_by_fingerprint(patient_id, Store.fingerprint(payload))
            if dup:
                raise DuplicateEventError(dup["event_id"])

            # Memory comes from disk, not from the request: the POST carries only the new event.
            history = [r["event"] for r in self.store.events(patient_id)]
            current = self.store.current_plan(patient_id)

            # Ask the planner first, so a failed call leaves no half-recorded event.
            draft = self.planner.make_plan(history, current, payload)

            now = _now()
            record = self.store.save_event(patient_id, payload, received_at=now)
            plan = Plan(
                **draft.model_dump(),
                version=(current.version + 1) if current else 1,
                patient_id=patient_id,
                created_at=now,
                trigger_event_id=record["event_id"],
                planner=self.planner.name,
            )
            self.store.save_plan(plan)

            entry = AuditEntry(
                seq=len(self.store.audit(patient_id)) + 1,
                recorded_at=now,
                patient_id=patient_id,
                event_id=record["event_id"],
                event_type=event.event_type,
                event_time=event.timestamp,
                from_version=current.version if current else None,
                to_version=plan.version,
                changes=plan.changes,
                rationale=plan.rationale,
                planner=plan.planner,
            )
            self.store.append_audit(entry)
            return plan, entry
