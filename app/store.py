"""Everything the agent remembers lives on disk, one folder per patient.

data/patients/<patient_id>/
    events/0001_admission.json     raw events, in arrival order
    plans/plan_v1.json             every plan version, never overwritten
    audit.jsonl                    append-only audit trail

Because nothing is kept only in memory, the agent picks up exactly where it
left off after a restart.
"""

import hashlib
import json
import shutil
from pathlib import Path

from .models import AuditEntry, Plan


class Store:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "patients"
        self.root.mkdir(parents=True, exist_ok=True)

    def _patient_dir(self, patient_id: str) -> Path:
        return self.root / patient_id

    def patient_exists(self, patient_id: str) -> bool:
        return self._patient_dir(patient_id).is_dir()

    def list_patients(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # ---- events -------------------------------------------------------

    @staticmethod
    def fingerprint(event: dict) -> str:
        canonical = json.dumps(event, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def events(self, patient_id: str) -> list[dict]:
        folder = self._patient_dir(patient_id) / "events"
        if not folder.is_dir():
            return []
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]

    def find_event_by_fingerprint(self, patient_id: str, fingerprint: str) -> dict | None:
        for record in self.events(patient_id):
            if record["fingerprint"] == fingerprint:
                return record
        return None

    def save_event(self, patient_id: str, event: dict, received_at: str) -> dict:
        folder = self._patient_dir(patient_id) / "events"
        folder.mkdir(parents=True, exist_ok=True)
        seq = len(list(folder.glob("*.json"))) + 1
        event_id = f"{seq:04d}_{event['event_type']}"
        record = {
            "event_id": event_id,
            "received_at": received_at,
            "fingerprint": self.fingerprint(event),
            "event": event,
        }
        (folder / f"{event_id}.json").write_text(json.dumps(record, indent=2))
        return record

    # ---- plans --------------------------------------------------------

    def plans(self, patient_id: str) -> list[Plan]:
        folder = self._patient_dir(patient_id) / "plans"
        if not folder.is_dir():
            return []
        files = sorted(folder.glob("plan_v*.json"), key=lambda p: int(p.stem.split("_v")[1]))
        return [Plan.model_validate_json(p.read_text()) for p in files]

    def current_plan(self, patient_id: str) -> Plan | None:
        plans = self.plans(patient_id)
        return plans[-1] if plans else None

    def save_plan(self, plan: Plan) -> None:
        folder = self._patient_dir(plan.patient_id) / "plans"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"plan_v{plan.version}.json"
        if path.exists():
            raise FileExistsError(f"{path} already exists; plans are never overwritten")
        path.write_text(plan.model_dump_json(indent=2))

    # ---- audit --------------------------------------------------------

    def audit(self, patient_id: str) -> list[AuditEntry]:
        path = self._patient_dir(patient_id) / "audit.jsonl"
        if not path.exists():
            return []
        return [AuditEntry.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]

    def append_audit(self, entry: AuditEntry) -> None:
        path = self._patient_dir(entry.patient_id) / "audit.jsonl"
        with path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")

    # ---- demo helper --------------------------------------------------

    def delete_patient(self, patient_id: str) -> bool:
        folder = self._patient_dir(patient_id)
        if not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True
