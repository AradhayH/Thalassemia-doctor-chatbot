import os
import time
import uuid
import json
import re
from typing import List, Optional, Literal, Dict, Any, Tuple
from enum import Enum
from collections import Counter

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


from google import genai
from google.genai import types as gtypes
client = genai.Client()

MODEL_ID = "gemini-2.5-flash-lite"

THREADS: Dict[str, Dict[str, Any]] = {}   
MESSAGES: Dict[str, List[Dict[str, Any]]] = {}  

DEFAULT_FAQS: List[Dict[str, str]] = [
    {
        "role": "patient",
        "title": "Pre-transfusion basics",
        "content": "Before transfusion: ABO/Rh typing, antibody screen, and crossmatch are standard. Bring prior transfusion records, allergy info, and current meds. Follow your clinician’s instructions on food/med timing."
    },
    {
        "role": "patient",
        "title": "Post-transfusion — when to seek help",
        "content": "Mild effects: small itchy hives, mild headache, or low-grade fever. Urgent red flags: fever ≥38°C, rigors, back/chest pain, dark urine, jaundice, widespread rash. Emergency: breathing trouble, face/lip swelling, severe chest pain, confusion, fainting."
    },
    {
        "role": "patient",
        "title": "Targets & intervals",
        "content": "Many thalassemia regimens aim for transfusions every 2–4 weeks to maintain pre-transfusion hemoglobin around 9–10.5 g/dL (sometimes individualized)."
    },
    {
        "role": "both",
        "title": "Iron overload & chelation",
        "content": "Repeated transfusions can cause iron overload; chelation is prescribed by clinicians. Monitoring includes ferritin and sometimes MRI."
    },
    {
        "role": "donor",
        "title": "Who can donate?",
        "content": "Healthy adults meeting eligibility criteria (age/weight/health) can donate. Intervals and deferrals vary by policy. Always bring ID and disclose recent illnesses or travel."
    },
    {
        "role": "donor",
        "title": "After donation care",
        "content": "Hydrate well, avoid heavy lifting for a day, and keep the bandage on for a few hours. If you feel faint or unwell, rest and contact the blood center."
    }
]

def load_external_kb() -> List[Dict[str, str]]:
    kb_path = os.path.join(os.path.dirname(__file__), "../../../../kb/cards.json")
    kb_path = os.path.abspath(kb_path)
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return DEFAULT_FAQS

FAQS = load_external_kb()

class Severity(str, Enum):
    mild = "mild"
    urgent = "urgent"
    emergency = "emergency"
    na = "n/a"

class Plan(BaseModel):
    """Structured plan emitted by the LLM router."""
    intent: Literal[
        "pre_transfusion_assistance",
        "post_transfusion_assistance",
        "symptom_triage",
        "faq_patient",
        "faq_donor",
        "contact_provider",
        "education"
    ]
    severity: Severity = Severity.na
    actions: List[Literal[
        "answer_now",
        "show_self_care",
        "handoff_blood_bank",
        "call_emergency",
        "start_thread",
        "show_education"
    ]] = []
    topics: List[str] = []
    need_profile_fields: List[Literal["patient_id","provider_id","blood_bank_contact","allergies","age_group"]] = []
    rationale: Optional[str] = None  

class ChatProfile(BaseModel):
    user_id: Optional[str] = None
    role: Optional[Literal["patient","donor"]] = None
    name: Optional[str] = None
    provider_id: Optional[str] = None
    blood_bank_contact: Optional[str] = None
    allergies: Optional[List[str]] = None
    age_group: Optional[Literal["child","adult"]] = None
    clinician_ok_otc: Optional[bool] = False

class ChatRequest(BaseModel):
    message: str
    profile: Optional[ChatProfile] = None
    thread_id: Optional[str] = None  

class ChatResponse(BaseModel):
    severity: Severity
    text: str
    actions: List[str] = []
    call_link: Optional[str] = None
    whatsapp_link: Optional[str] = None
    educational_cards: Optional[List[Dict[str, str]]] = None
    thread_id: Optional[str] = None
    disclaimer: str = "This is general information, not medical advice."

class StartThreadRequest(BaseModel):
    patient_id: str
    provider_id: str

class SendMessageRequest(BaseModel):
    sender_role: Literal["patient","provider"]
    text: str

_word = re.compile(r"[a-z0-9]+")

def _tokens(s: str) -> List[str]:
    return _word.findall(s.lower())

def score_faq(query: str, role: Optional[str]) -> List[Tuple[float, Dict[str,str]]]:
    qtok = set(_tokens(query))
    results = []
    for item in FAQS:
        if role and item.get("role") not in (role, "both"):
            continue
        text = (item.get("title","") + " " + item.get("content","")).strip()
        itok = set(_tokens(text))
        if not itok:
            continue
        inter = len(qtok & itok)
        union = len(qtok | itok)
        jacc = inter / union if union else 0.0

        bonus = 0.1 if any(t in _tokens(item.get("title","")) for t in qtok) else 0.0
        score = jacc + bonus
        results.append((score, item))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:3]

def build_context(query: str, role: Optional[str]) -> str:
    top = score_faq(query, role)
    if not top:
        return ""
    ctx_lines = []
    for sc, it in top:
        ctx_lines.append(f"[{it.get('title','Card')}] {it.get('content','')}")
    return "\n".join(ctx_lines)

EMERGENCY_TRIGGERS = {
    "trouble breathing","shortness of breath","wheezing","swelling face","swelling lips",
    "swelling tongue","chest pain","pressure in chest","confusion","fainting",
    "very dark urine","black urine","rigors with fever","low blood pressure",
    "severe dizziness","anaphylaxis"
}
URGENT_TRIGGERS = {
    "fever 38","fever 38.5","fever ≥38","fever above 38","fever of 101",
    "rigors","shaking chills","back pain","loin pain","dark urine","jaundice","widespread rash","persistent vomiting"
}

def emergency_override(user_text: str) -> Optional[Severity]:
    low = user_text.lower()
    if any(t in low for t in EMERGENCY_TRIGGERS):
        return Severity.emergency
    if any(t in low for t in URGENT_TRIGGERS):
        return Severity.urgent
    return None

ROUTER_PROMPT = """You are “Care Chat”, a safety-first assistant for thalassemia patients and blood donors.
Your job: read the user's message and produce a PLAN in JSON.
Do not use any bold or italics words
- intent: classify as one of {pre_transfusion_assistance, post_transfusion_assistance, symptom_triage, faq_patient, faq_donor, contact_provider, education}
- severity: {mild, urgent, emergency, n/a}. If message implies request to talk to provider or schedule, severity may still be n/a.
- actions: choose from {answer_now, show_self_care, handoff_blood_bank, call_emergency, start_thread, show_education}
- topics: keywords like ["allergic_reaction","fever","hives","hb_target","chelation_basics"]
- need_profile_fields: which profile fields are useful to retrieve (e.g., blood_bank_contact, provider_id, patient_id)
Rules:
• If any emergency-type issues (breathing trouble, facial swelling, severe chest pain, confusion, fainting, very dark urine, rigors with fever) -> severity=emergency and actions=[call_emergency].
• If urgent red flags (fever ≥38°C 24–48h post transfusion, rigors, back/loin/chest pain, dark urine, jaundice, widespread rash) -> severity=urgent and actions include handoff_blood_bank (and possibly start_thread).
• For generic questions from patients -> faq_patient; from donors -> faq_donor; otherwise infer.
• Only use 'contact_provider' when the user explicitly asks to talk to their provider or the message clearly indicates they want help from provider now.
• rationale: short one-liner on why you chose this plan.
Return ONLY JSON, no extra text.
Few-shot examples:

User: "I feel itchy small hives after my transfusion yesterday, breathing fine."
-> {"intent":"symptom_triage","severity":"mild","actions":["answer_now","show_self_care"],"topics":["hives","allergic_reaction"],"need_profile_fields":[],"rationale":"mild allergic-type symptoms without breathing issues"}

User: "I have fever 38.6 and back pain today after transfusion."
-> {"intent":"symptom_triage","severity":"urgent","actions":["handoff_blood_bank","start_thread"],"topics":["fever","back_pain"],"need_profile_fields":["provider_id","blood_bank_contact"],"rationale":"urgent red flags post-transfusion"}

User: "Can I donate if I had a cold last week?"
-> {"intent":"faq_donor","severity":"n/a","actions":["answer_now"],"topics":["donor_eligibility"],"need_profile_fields":[],"rationale":"donor FAQ"}

User: "Please connect me with my hematologist."
-> {"intent":"contact_provider","severity":"n/a","actions":["start_thread"],"topics":[],"need_profile_fields":["provider_id","patient_id"],"rationale":"explicit provider contact request"}
"""

GENERATOR_SYSTEM = """You answer questions for thalassemia patients and blood donors.
Use the provided CONTEXT if helpful. Write in plain language, short steps, and include clear next steps.
Do NOT diagnose or prescribe. For mild symptoms you may mention generic OTC paracetamol or an antihistamine ONLY if the user already confirmed their clinician allowed OTCs; never give dosing.
Always end with: "This is general information, not medical advice."
If the question is outside your scope or risky, politely recommend contacting a clinician or blood bank.
"""

def llm_plan(user_text: str) -> Plan:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[ROUTER_PROMPT, f"User: {user_text}\nReturn ONLY the JSON plan."],
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Plan,  # SDK parses to Pydantic
        ),
    )
    try:
        return response.parsed
    except Exception as e:

        return Plan(intent="symptom_triage", severity=Severity.na, actions=["answer_now"], topics=[])

def llm_answer(user_text: str, context: str, profile: ChatProfile) -> str:

    profile_line = []
    if profile.role: profile_line.append(f"role={profile.role}")
    if profile.age_group: profile_line.append(f"age_group={profile.age_group}")
    if profile.allergies: profile_line.append(f"allergies={', '.join(profile.allergies[:3])}")
    if profile.clinician_ok_otc: profile_line.append("clinician_ok_otc=true")
    meta = " | ".join(profile_line)

    contents = [
        GENERATOR_SYSTEM,
        f"PROFILE: {meta}" if meta else "PROFILE: (none)",
        "CONTEXT:\n" + (context or "(no special context)"),
        "USER:\n" + user_text,
        "Please answer helpfully with clear next steps and finish with the standard disclaimer."
    ]
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=gtypes.GenerateContentConfig(max_output_tokens=512),
    )
    return (resp.text or "").strip()

app = FastAPI(title="Care Chat (LLM + RAG + Messaging)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    profile = req.profile or ChatProfile()


    plan = llm_plan(req.message)

    
    forced = emergency_override(req.message)
    if forced == Severity.emergency:
        return ChatResponse(
            severity=Severity.emergency,
            text="🚨 Emergency symptoms detected. Call 112/108 immediately or go to the nearest emergency department. Do not take extra medicines unless a clinician advised.",
            actions=["call_emergency"],
            call_link="tel:112"
        )
    elif forced == Severity.urgent and plan.severity != Severity.emergency:
        plan.severity = Severity.urgent
        if "handoff_blood_bank" not in plan.actions:
            plan.actions.append("handoff_blood_bank")

    #Contact/handoff handling
    thread_id = req.thread_id
    if plan.intent == "contact_provider" or "start_thread" in plan.actions or plan.severity == Severity.urgent:
        # Auto-start a thread if not provided and we have IDs
        if not thread_id and profile.user_id and profile.provider_id:
            thread_id = _start_thread(profile.user_id, profile.provider_id)
        # Prebuild call/WhatsApp links if contact stored
        call_link, wa_link = None, None
        if profile.blood_bank_contact:
            call_link = f"tel:{profile.blood_bank_contact}"
            import urllib.parse as _u
            text = f"Hello, I need assistance. Message: {req.message.replace(chr(10),' ')}"
            wa_link = f"https://wa.me/{profile.blood_bank_contact.lstrip('+').replace(' ','')}?text={_u.quote(text)}"
        # If urgent, keep response focused on contact
        if plan.severity == Severity.urgent:
            return ChatResponse(
                severity=Severity.urgent,
                text="Your symptoms may need prompt attention. Please contact your blood bank/doctor now.",
                actions=sorted(list(set(plan.actions + ["handoff_blood_bank"]))),
                call_link=call_link,
                whatsapp_link=wa_link,
                thread_id=thread_id
            )

    role = profile.role or ("patient" if plan.intent in ("pre_transfusion_assistance","post_transfusion_assistance","symptom_triage","faq_patient") else "donor")
    context = build_context(req.message, role=role)
    answer = llm_answer(req.message, context=context, profile=profile)

    cards = None
    if plan.intent in ("education","faq_patient","faq_donor") or "show_education" in plan.actions:
        top = score_faq(req.message, role=role)
        if top:
            cards = [{"title": it["title"], "body": it["content"]} for _, it in top]

    return ChatResponse(
        severity=plan.severity,
        text=answer,
        actions=sorted(list(set(plan.actions or ["answer_now"]))),
        thread_id=thread_id,
        educational_cards=cards
    )

@app.get("/faqs/search")
def faqs_search(q: str = Query(..., description="search query"), role: Optional[str] = Query(None)):
    top = score_faq(q, role)
    return {
        "query": q,
        "role": role,
        "results": [{"score": round(sc, 3), "title": it["title"], "content": it["content"], "role": it["role"]} for sc, it in top]
    }

def _start_thread(patient_id: str, provider_id: str) -> str:
    tid = uuid.uuid4().hex[:12]
    THREADS[tid] = {"thread_id": tid, "patient_id": patient_id, "provider_id": provider_id, "created_at": int(time.time())}
    MESSAGES[tid] = []
    return tid

@app.post("/thread/start")
def start_thread(req: StartThreadRequest):
    tid = _start_thread(req.patient_id, req.provider_id)
    return THREADS[tid]

@app.get("/thread/{thread_id}")
def get_thread(thread_id: str):
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {
        "thread": THREADS[thread_id],
        "messages": MESSAGES.get(thread_id, [])
    }

@app.post("/thread/{thread_id}/message")
def send_message(thread_id: str, req: SendMessageRequest):
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")
    msg = {
        "id": uuid.uuid4().hex[:10],
        "sender_role": req.sender_role,
        "text": req.text,
        "ts": int(time.time())
    }
    MESSAGES[thread_id].append(msg)
    return {"ok": True, "message": msg}
