"""Planners turn (patient history + current plan + new event) into a new plan.

OpenAIPlanner calls the OpenAI API. MockPlanner returns canned plans for the
three demo events so the app runs without an API key and tests stay offline.
"""

import json
import os
from pathlib import Path
from typing import Protocol

import openai

from .models import Plan, PlanDraft

SYSTEM_PROMPT = """\
You are a clinical case-tracking agent that follows one hospitalized patient over several days.
Each time a new event arrives (admission, lab results, culture results, nursing notes), you
receive the patient's full event history, the current care plan, and the new event. Produce the
updated care plan.

Guidelines:
- Base every change on the evidence in the events. Cite the specific values or findings that
  drove each change in `rationale`.
- Keep what still applies from the previous plan; change only what the new information
  justifies. List each concrete difference in `changes` (for the first plan, summarize what was
  created).
- Be specific and concise: drug names with dose/route/frequency where standard, named tests,
  and monitoring intervals.
- This is a fictional teaching demo. The plan is decision support for a clinician, not medical
  advice; do not include disclaimers in the output.
"""


DEFAULT_KEY_FILE = Path(__file__).resolve().parent.parent / "apikey.txt"


def load_openai_key() -> str | None:
    """OPENAI_API_KEY wins; otherwise read the key file (OPENAI_API_KEY_FILE or ./apikey.txt)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    path = Path(os.environ.get("OPENAI_API_KEY_FILE", DEFAULT_KEY_FILE))
    if path.is_file():
        return path.read_text().strip() or None
    return None


class Planner(Protocol):
    name: str

    def make_plan(self, history: list[dict], current: Plan | None, new_event: dict) -> PlanDraft: ...


class PlannerError(RuntimeError):
    pass


def build_user_message(history: list[dict], current: Plan | None, new_event: dict) -> str:
    previous = (
        current.model_dump_json(indent=2, exclude={"patient_id", "trigger_event_id", "planner"})
        if current
        else "None yet. This is the first event for this patient; create plan v1."
    )
    return (
        f"<event_history>\n{json.dumps(history, indent=2)}\n</event_history>\n\n"
        f"<current_plan>\n{previous}\n</current_plan>\n\n"
        f"<new_event>\n{json.dumps(new_event, indent=2)}\n</new_event>\n\n"
        "Write the updated care plan."
    )


class OpenAIPlanner:
    def __init__(self, model: str = "gpt-5.5", client: openai.OpenAI | None = None, api_key: str | None = None):
        self.model = model
        self.name = model
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> openai.OpenAI:
        # Created on first use, so a missing key becomes a clear error on POST instead of a startup crash.
        if self._client is None:
            api_key = self._api_key or load_openai_key()
            if not api_key:
                raise PlannerError(f"No OpenAI key found. Set OPENAI_API_KEY or put the key in {DEFAULT_KEY_FILE.name}.")
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def make_plan(self, history: list[dict], current: Plan | None, new_event: dict) -> PlanDraft:
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(history, current, new_event)},
                ],
                response_format=PlanDraft,
            )
        except openai.AuthenticationError as e:
            raise PlannerError("OpenAI rejected the API key. Check apikey.txt / OPENAI_API_KEY or use PLANNER=mock.") from e
        except openai.RateLimitError as e:
            raise PlannerError("OpenAI rate limit or quota hit. Wait a moment and POST the event again.") from e
        except openai.APIStatusError as e:
            raise PlannerError(f"OpenAI API error {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:
            raise PlannerError("Could not reach the OpenAI API. Check your network.") from e
        except openai.LengthFinishReasonError as e:
            raise PlannerError("OpenAI ran out of output tokens before finishing the plan.") from e

        message = completion.choices[0].message
        if message.refusal:
            raise PlannerError(f"OpenAI declined to produce a plan: {message.refusal}")
        if message.parsed is None:
            raise PlannerError("OpenAI did not return a plan.")
        self.name = completion.model
        return message.parsed


# Canned plans for the three demo events in inbox/.
_MOCK_PLANS: dict[str, dict] = {
    "admission": {
        "patient_summary": "45-year-old man with 3 days of fever, productive cough, and right-sided pleuritic chest pain. Febrile and tachycardic, SpO2 94% on room air.",
        "working_diagnosis": "Suspected community-acquired pneumonia (right lower lobe)",
        "confidence": "low",
        "differential": ["Viral respiratory infection (influenza, COVID-19)", "Acute bronchitis"],
        "medications": ["Acetaminophen 1 g PO every 6 hours as needed for fever", "IV normal saline maintenance fluids"],
        "actions": ["CBC with differential, CRP, procalcitonin, lactate, creatinine", "Chest X-ray", "Influenza / COVID-19 PCR", "Blood cultures x2 and sputum culture before any antibiotics"],
        "monitoring": ["Vitals every 4 hours", "Continuous pulse oximetry; supplemental O2 if SpO2 < 92%"],
        "pending": ["Blood tests", "Chest X-ray", "Viral PCR", "Cultures"],
        "changes": ["Created initial plan v1"],
        "rationale": "Fever, productive cough, focal right-sided crackles and pleuritic pain suggest pneumonia, but a viral cause is possible. Hemodynamically stable, so obtain labs, imaging, and cultures first and support with fluids and antipyretics.",
    },
    "lab_results": {
        "patient_summary": "45-year-old man admitted with fever and cough; labs and imaging now support bacterial pneumonia. Clinically stable.",
        "working_diagnosis": "Community-acquired bacterial pneumonia, right lower lobe",
        "confidence": "high",
        "differential": ["Atypical pneumonia (Legionella, Mycoplasma)"],
        "medications": ["Ceftriaxone 1 g IV every 24 hours", "Azithromycin 500 mg IV/PO daily", "Acetaminophen 1 g PO every 6 hours as needed for fever"],
        "actions": ["Start empiric antibiotics now", "Legionella urinary antigen", "Encourage oral intake; taper IV fluids"],
        "monitoring": ["Vitals every 4 hours", "Continuous pulse oximetry", "Repeat CBC and CRP in 48 hours"],
        "pending": ["Blood culture", "Sputum culture", "Legionella urinary antigen"],
        "changes": ["Diagnosis firmed up: suspected pneumonia -> bacterial pneumonia (RLL)", "Started ceftriaxone + azithromycin", "Ruled out influenza / COVID-19", "Added Legionella urinary antigen"],
        "rationale": "WBC 16.2 with 85% neutrophils, CRP 145, and procalcitonin 2.1 point to a bacterial infection; chest X-ray shows right lower lobe consolidation; viral PCR negative. Normal lactate and creatinine mean no sepsis or kidney injury, so standard ward treatment is appropriate.",
    },
    "culture_results": {
        "patient_summary": "45-year-old man with pneumococcal pneumonia, now afebrile and improving on antibiotics; eating well.",
        "working_diagnosis": "Streptococcus pneumoniae pneumonia, right lower lobe (penicillin-susceptible)",
        "confidence": "high",
        "differential": [],
        "medications": ["Amoxicillin 1 g PO three times daily to complete 5-7 days total", "Acetaminophen 1 g PO every 6 hours as needed"],
        "actions": ["Stop ceftriaxone and azithromycin", "Switch to oral amoxicillin (IV-to-PO switch)", "Begin discharge planning"],
        "monitoring": ["Vitals every 8 hours", "Confirm afebrile 24-48 hours before discharge"],
        "pending": [],
        "changes": ["Narrowed antibiotics: ceftriaxone -> oral amoxicillin", "Stopped azithromycin (organism resistant; Legionella negative)", "Started discharge planning"],
        "rationale": "Sputum culture grew S. pneumoniae susceptible to penicillin/amoxicillin and resistant to azithromycin. Blood cultures negative and Legionella antigen negative. Patient afebrile (37.4 C), HR 86, SpO2 97%, eating, so narrow to targeted oral therapy and plan discharge.",
    },
}


class MockPlanner:
    name = "mock"

    def make_plan(self, history: list[dict], current: Plan | None, new_event: dict) -> PlanDraft:
        canned = _MOCK_PLANS.get(new_event.get("event_type", ""))
        if canned:
            return PlanDraft(**canned)
        if current is None:
            raise PlannerError("Mock planner needs an admission event first.")
        # Unknown event type: keep the plan and record that nothing changed.
        base = current.model_dump(include=set(PlanDraft.model_fields))
        base.update(changes=[f"No change (mock planner has no rule for '{new_event.get('event_type')}')"], rationale="Event recorded; plan carried forward unchanged.")
        return PlanDraft(**base)
