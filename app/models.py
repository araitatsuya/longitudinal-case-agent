"""Data shapes shared by the store, the planner, and the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """One thing that happened to a patient (admission, lab result, ...).

    Only `patient_id` and `event_type` are required; everything else in the
    JSON body is kept as-is and shown to the planner.
    """

    model_config = ConfigDict(extra="allow")

    patient_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    event_type: str = Field(min_length=1)
    timestamp: str | None = None


class PlanDraft(BaseModel):
    """What the planner (OpenAI or the mock) returns for one event."""

    patient_summary: str = Field(description="One or two sentences on who the patient is and how they are doing now.")
    working_diagnosis: str = Field(description="The current best explanation for the illness.")
    confidence: Literal["low", "medium", "high"] = Field(description="How sure we are of the working diagnosis.")
    differential: list[str] = Field(description="Other diagnoses still being considered.")
    medications: list[str] = Field(description="Current medications with dose/route if known.")
    actions: list[str] = Field(description="Tests, procedures, and care steps to do now.")
    monitoring: list[str] = Field(description="What to watch and how often.")
    pending: list[str] = Field(description="Results or decisions we are still waiting on.")
    changes: list[str] = Field(description="Bullet list of what changed compared with the previous plan. For the first plan, describe it as created.")
    rationale: str = Field(description="Why the plan changed, citing the specific new findings.")


class Plan(PlanDraft):
    """A saved plan version."""

    version: int
    patient_id: str
    created_at: str
    trigger_event_id: str
    planner: str


class AuditEntry(BaseModel):
    """One line of the audit trail: what came in, what changed, and why."""

    seq: int
    recorded_at: str
    patient_id: str
    event_id: str
    event_type: str
    event_time: str | None
    from_version: int | None
    to_version: int
    changes: list[str]
    rationale: str
    planner: str
