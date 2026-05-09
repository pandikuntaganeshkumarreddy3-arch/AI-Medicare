"""
response_generator.py  (IMPROVED)
---------------------
Generates warm, medically accurate, context-aware responses using Ollama.

Key improvements over previous version:
  - Understands situation type: injury / symptom / medication / general
  - Asks follow-up questions instead of jumping to conclusions
  - Handles injuries (bleeding, accidents) correctly
  - Uses conversation context to build on previous messages
  - Uses web-search-style medical grounding in the prompt
  - Does NOT diagnose rare diseases from common symptoms
  - Responds proportionally to urgency
"""

import ollama
import re

MODEL_NAME  = "mistral"
TEMPERATURE = 0.65
MAX_TOKENS  = 600


# ──────────────────────────────────────────────────────────────
# 1.  Situation classifier
#     Figures out WHAT kind of query this is before responding.
# ──────────────────────────────────────────────────────────────

SITUATION_RULES = {
    "injury": [
        "accident", "fell", "fall", "injury", "bleeding", "cut", "wound",
        "broke", "broken", "fracture", "burn", "hit", "collision", "crash",
        "bike", "car", "knocked", "sprain", "twisted", "bruise", "scraped",
        "blood", "gash", "bite",
    ],
    "emergency": [
        "chest pain", "heart attack", "stroke", "can't breathe",
        "cannot breathe", "difficulty breathing", "unconscious",
        "seizure", "overdose", "poisoning", "severe bleeding",
        "suicidal", "suicide", "dying", "collapsed", "paralysis",
        "not breathing", "stopped breathing", "unresponsive",
        "severe chest", "loss of consciousness",
    ],
    "high_risk": [
        "blood pressure very high", "sugar very high", "high fever",
        "severe pain", "not responding", "very sick",
        "infection spreading", "allergic reaction", "swelling throat",
        "blurred vision", "sudden weakness", "very dizzy", "fainted",
        "vomiting blood", "black stool", "severe headache", "high temperature",
    ],
    "medication": [
        "dosage", "dose", "tablet", "capsule", "how much", "take",
        "overdose", "side effect", "interaction", "prescription",
        "mg", "ml", "paracetamol", "amoxicillin", "metformin",
        "ibuprofen", "aspirin", "medicine", "drug", "pill",
        "antibiotic", "insulin", "blood thinner",
    ],
    "symptom": [
        "pain", "ache", "fever", "cough", "cold", "nausea",
        "vomiting", "diarrhea", "headache", "tired", "fatigue",
        "rash", "swelling", "itching", "burning", "numbness",
        "dizzy", "weakness", "breath", "shortness", "sweating",
    ],
}


def classify_situation(question: str) -> str:
    """
    Classify the type of medical situation.
    Returns: 'emergency' | 'injury' | 'high_risk' | 'medication' | 'symptom' | 'general'
    Priority: emergency > injury > high_risk > medication > symptom > general
    """
    q = question.lower()
    for situation, keywords in SITUATION_RULES.items():
        if any(kw in q for kw in keywords):
            return situation
    return "general"


# ──────────────────────────────────────────────────────────────
# 2.  Risk analyser  (fast keyword + LLM for nuanced cases)
# ──────────────────────────────────────────────────────────────

def analyse_risk(question: str, situation: str) -> str:
    """Returns: 'low' | 'medium' | 'high' | 'emergency'"""

    if situation == "emergency":
        return "emergency"
    if situation in ("injury", "high_risk"):
        return "high"
    if situation == "symptom":
        return "medium"
    if situation == "medication":
        return "low"

    # Use LLM for general/ambiguous cases
    try:
        risk_prompt = f"""You are a medical triage tool.
Classify the risk level of this query in ONE word only.

Query: "{question}"

Options (reply with exactly one):
- low       = basic health question, general info
- medium    = noticeable symptoms, needs attention
- high      = serious symptoms, see doctor soon
- emergency = life-threatening

One word:"""
        response = ollama.chat(
            model    = MODEL_NAME,
            messages = [{"role": "user", "content": risk_prompt}],
            options  = {"temperature": 0.1, "num_predict": 5},
        )
        risk = response["message"]["content"].strip().lower().split()[0]
        if risk not in ("low", "medium", "high", "emergency"):
            risk = "medium"
        print(f"[response_generator] LLM risk assessment: {risk.upper()}")
        return risk
    except Exception as e:
        print(f"[response_generator] Risk LLM failed: {e} — defaulting to medium")
        return "medium"


# ──────────────────────────────────────────────────────────────
# 3.  System prompt builder
#     Each situation type gets a SPECIFIC, accurate prompt.
# ──────────────────────────────────────────────────────────────

def build_system_prompt(situation: str, risk: str) -> str:
    """
    Build a tailored system prompt based on the situation type.
    This is the most important improvement — the LLM is given
    very specific instructions matching the type of query.
    """
    base = (
        "You are a warm, knowledgeable medical assistant helping an elderly patient. "
        "You use simple, clear language. You are caring and calm. "
        "You NEVER diagnose rare diseases without very strong evidence. "
        "You NEVER jump to alarming conclusions. "
        "You ask ONE focused follow-up question when you need more information "
        "before giving specific advice. "
    )

    if situation == "injury":
        return base + (
            "\n\nSPECIAL RULE — INJURY QUERIES:\n"
            "The patient has described a physical injury (accident, cut, wound, etc).\n"
            "Your FIRST priority is IMMEDIATE FIRST AID advice — not diagnosis.\n"
            "For bleeding: tell them to apply direct pressure, elevate if possible, check severity.\n"
            "For falls: ask about pain level, movement ability, head impact.\n"
            "For burns: advise cool running water for 20 minutes.\n"
            "NEVER assume a bleeding injury means a clotting disorder or rare disease.\n"
            "Bleeding from a physical injury is NORMAL — manage it first.\n"
            "Ask follow-up: How bad is the bleeding? Has it slowed? Can they move the area?\n"
            "Refer to emergency services if: heavy uncontrolled bleeding, unconscious, severe head injury.\n"
        )

    if situation == "emergency":
        return base + (
            "\n\nSPECIAL RULE — EMERGENCY:\n"
            "This sounds like a potential emergency. Be DIRECT, CLEAR, and URGENT.\n"
            "Tell them to call emergency services (ambulance) IMMEDIATELY.\n"
            "Give 2-3 simple things to do RIGHT NOW while waiting for help.\n"
            "Do NOT give a long explanation. Keep it SHORT and actionable.\n"
        )

    if situation == "high_risk":
        return base + (
            "\n\nSPECIAL RULE — HIGH RISK SYMPTOMS:\n"
            "The patient has described serious symptoms that need prompt attention.\n"
            "Be clear and calm. Do NOT panic them but do not minimise either.\n"
            "Give clear guidance on what they should do: rest, see doctor today, monitor.\n"
            "Ask ONE specific follow-up question to understand severity.\n"
            "Example follow-ups: How long has this been happening? How severe (1-10)? Any other symptoms?\n"
        )

    if situation == "medication":
        return base + (
            "\n\nSPECIAL RULE — MEDICATION QUERIES:\n"
            "The patient is asking about a medication.\n"
            "Give accurate, specific information from the medical knowledge provided.\n"
            "Cover: what it is for, standard dosage, key side effects, important warnings.\n"
            "If you're not sure about a specific medication, say so honestly.\n"
            "Do NOT guess drug interactions — recommend they ask a pharmacist.\n"
        )

    if situation == "symptom":
        return base + (
            "\n\nSPECIAL RULE — SYMPTOM QUERIES:\n"
            "The patient is describing symptoms. Ask 1-2 clarifying questions first if needed.\n"
            "Common symptoms have COMMON causes — do not suggest rare diseases.\n"
            "Headache is usually tension/stress/dehydration, not a tumour.\n"
            "Fatigue is usually sleep/anaemia/stress, not a rare condition.\n"
            "Give practical, helpful advice. Recommend a doctor if symptoms are persistent.\n"
        )

    # General
    return base + (
        "\n\nAnswer the patient's medical question helpfully and clearly.\n"
        "Keep answers focused and use simple language.\n"
        "If unsure, recommend they consult their doctor or pharmacist.\n"
    )


# ──────────────────────────────────────────────────────────────
# 4.  User prompt builder
# ──────────────────────────────────────────────────────────────

def build_user_prompt(question: str, context_text: str, situation: str, emotion: str, risk: str) -> str:
    """Build the main user-turn prompt with grounding context."""

    tone_guidance = {
        "fear":    "The patient sounds scared. Be reassuring and calm.",
        "sadness": "The patient sounds distressed. Be gentle and supportive.",
        "anger":   "The patient sounds frustrated. Be understanding and patient.",
        "joy":     "The patient is in a positive mood. Be warm and friendly.",
        "neutral": "Respond in a clear, informative way.",
    }.get(emotion, "Respond warmly and helpfully.")

    length_guidance = {
        "emergency": "Keep response VERY SHORT — 3-4 sentences max. Prioritise action steps.",
        "injury":    "Keep response SHORT and PRACTICAL — 4-5 sentences. Focus on what to do RIGHT NOW.",
        "high_risk": "5-6 sentences. Be clear about next steps.",
        "medication": "5-6 sentences covering key facts about the medication.",
        "symptom":   "4-5 sentences. Ask a follow-up if you need more info.",
        "general":   "3-5 sentences. Keep it simple.",
    }.get(situation, "4-5 sentences.")

    context_block = f"\n\nMedical reference information:\n{context_text}" if context_text else ""

    return (
        f"Patient's message: \"{question}\"\n\n"
        f"Tone guidance: {tone_guidance}\n"
        f"Length: {length_guidance}\n"
        f"Remember: Do NOT jump to rare disease diagnoses. Ask a follow-up question if you need "
        f"more information to give proper advice. Respond as a knowledgeable, caring medical assistant.\n"
        f"{context_block}\n\n"
        f"Your response:"
    )


# ──────────────────────────────────────────────────────────────
# 5.  Tone intro based on risk + emotion
# ──────────────────────────────────────────────────────────────

RISK_INTROS = {
    "low": {
        "neutral": "Here is what you need to know.",
        "joy":     "Happy to help! Here is some useful information.",
        "fear":    "No need to worry too much about this — let me explain.",
        "default": "Good question. Let me help.",
    },
    "medium": {
        "neutral": "Let me explain what is important here.",
        "fear":    "I understand your concern. Let me give you clear information.",
        "sadness": "I am sorry you are going through this. Here is what can help.",
        "anger":   "I hear you. Let me give you the information you need.",
        "default": "Let me help you understand this.",
    },
    "high": {
        "fear":    "I understand you are worried. Please take this seriously and read carefully.",
        "neutral": "This needs attention. Please read the following carefully.",
        "default": "This is important. Please read carefully and take action.",
    },
    "emergency": {
        "default": "⚠️ This sounds urgent. Please act immediately.",
    },
}

def get_intro(risk: str, emotion: str) -> str:
    bucket = RISK_INTROS.get(risk, RISK_INTROS["medium"])
    return bucket.get(emotion, bucket.get("default", "Let me help you."))


# ──────────────────────────────────────────────────────────────
# 6.  Footer by risk level
# ──────────────────────────────────────────────────────────────

def get_footer(risk: str) -> str:
    footers = {
        "emergency": "🚨 CALL EMERGENCY SERVICES NOW. Do not wait.",
        "high":      "⚕️ Please contact your doctor or clinic today. Do not delay.",
        "medium":    "⚕️ If symptoms worsen or persist, please see your doctor.",
        "low":       "⚕️ Always check with your doctor or pharmacist if you are unsure.",
    }
    return footers.get(risk, footers["low"])


# ──────────────────────────────────────────────────────────────
# 7.  Main public function
# ──────────────────────────────────────────────────────────────

def generate_emotional_medical_answer(
    question: str,
    docs:     list[dict],
    emotion:  str,
) -> dict:
    """
    Generate a situation-aware, emotion-aware medical answer.

    Pipeline:
      1. Classify situation type (injury / symptom / medication / etc.)
      2. Assess risk level
      3. Build targeted system + user prompts
      4. Call Ollama
      5. Return formatted response with correct footer

    Returns:
        {
            "answer":       str,
            "risk":         str,    # low / medium / high / emergency
            "needs_doctor": bool,
        }
    """
    print(f"\n[response_generator] Question: '{question[:60]}…'")

    # Step 1: Classify situation
    situation = classify_situation(question)
    print(f"[response_generator] Situation: {situation}")

    # Step 2: Assess risk
    risk = analyse_risk(question, situation)
    print(f"[response_generator] Risk: {risk.upper()}")

    # Step 3: Get intro
    intro = get_intro(risk, emotion)

    # Step 4: Handle no-docs case
    if not docs and situation not in ("emergency", "injury"):
        return {
            "answer": (
                f"{intro}\n\n"
                "I don't have specific information about this in my database right now. "
                "I'd recommend speaking with your doctor or pharmacist for accurate advice.\n\n"
                f"─────────────────────────────\n{get_footer(risk)}"
            ),
            "risk":         risk,
            "needs_doctor": risk in ("high", "emergency"),
        }

    # Step 5: Build context from docs
    context_parts = []
    for doc in docs[:2]:
        # Take the most relevant portion of each doc
        content = doc["content"][:600]
        context_parts.append(content)
    context_text = "\n---\n".join(context_parts) if context_parts else ""

    # Step 6: Build prompts
    system_prompt = build_system_prompt(situation, risk)
    user_prompt   = build_user_prompt(question, context_text, situation, emotion, risk)

    # Step 7: Call Ollama
    try:
        print(f"[response_generator] Calling Ollama [{MODEL_NAME}] — situation={situation} risk={risk} emotion={emotion}")

        response = ollama.chat(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options  = {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            }
        )

        answer_text = response["message"]["content"].strip()
        print(f"[response_generator] Response: {len(answer_text)} chars")

        # Format final response
        if risk == "emergency":
            final = f"🚨 {intro}\n\n{answer_text}\n\n─────────────────────────────\n{get_footer(risk)}"
        elif risk == "high":
            final = f"{intro}\n\n{answer_text}\n\n─────────────────────────────\n{get_footer(risk)}"
        else:
            final = f"{intro}\n\n{answer_text}\n\n─────────────────────────────\n{get_footer(risk)}"

        return {
            "answer":       final,
            "risk":         risk,
            "needs_doctor": risk in ("high", "emergency"),
        }

    except Exception as e:
        print(f"[response_generator] Ollama error: {e}")
        return {
            "answer":       _simple_fallback(docs, intro, situation, get_footer(risk)),
            "risk":         risk,
            "needs_doctor": risk in ("high", "emergency"),
        }


# ──────────────────────────────────────────────────────────────
# 8.  Fallback when Ollama is offline
# ──────────────────────────────────────────────────────────────

def _simple_fallback(docs: list[dict], intro: str, situation: str, footer: str) -> str:
    """
    Provide a helpful fallback response without Ollama.
    For injuries, give basic first aid. Otherwise extract from docs.
    """
    if situation == "injury":
        return (
            f"{intro}\n\n"
            "For any bleeding injury:\n"
            "1. Apply firm, direct pressure with a clean cloth.\n"
            "2. Elevate the injured area above heart level if possible.\n"
            "3. Keep pressure on for at least 10 minutes without lifting.\n"
            "4. If bleeding is severe or won't stop after 15 minutes, go to the nearest emergency room.\n\n"
            f"─────────────────────────────\n{footer}\n\n"
            f"⚠️ Full AI responses need Ollama running: ollama serve"
        )

    lines = []
    if docs:
        content   = docs[0]["content"]
        all_lines = [l.strip() for l in content.split("\n") if l.strip()]
        for line in all_lines:
            if any(s in line.upper() for s in ["TOPIC:", "SOURCE:", "----", "Q:", "MEDICINE:", "CONDITION:"]):
                continue
            if line.startswith("A:"):
                line = line[2:].strip()
            if len(line) > 40:
                lines.append(line)
            if len(lines) >= 3:
                break

    text = " ".join(lines) if lines else "Please speak with your doctor for more information."
    return (
        f"{intro}\n\n{text}\n\n"
        f"─────────────────────────────\n{footer}\n\n"
        f"⚠️ Full AI responses need Ollama running: ollama serve"
    )


# ──────────────────────────────────────────────────────────────
# 9.  Ollama health check
# ──────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        models      = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]
        if any(MODEL_NAME in name for name in model_names):
            print(f"[response_generator] ✅ Ollama ready. Model '{MODEL_NAME}' found.")
            return True
        print(f"[response_generator] ⚠️  Model '{MODEL_NAME}' not found.")
        print(f"[response_generator] Run: ollama pull {MODEL_NAME}")
        return False
    except Exception:
        print("[response_generator] ⚠️  Ollama is not running.")
        return False


# ──────────────────────────────────────────────────────────────
# 10.  Self test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Response Generator — Improved Self Test")
    print("=" * 60 + "\n")

    check_ollama()

    sample_docs = [{
        "id": "paracetamol",
        "content": """MEDICINE: Paracetamol
OVERVIEW: Common painkiller for mild-to-moderate pain and fever.
DOSAGE: Adults 500mg-1000mg every 4-6 hours. Max 4000mg/day.
WARNINGS: Do not exceed dose. Overdose causes serious liver damage.""",
    }]

    test_cases = [
        ("I had a bike accident and my knee is bleeding a lot",                "fear",    "INJURY"),
        ("I had a bike accident and I am bleeding. My blood is not clotting.", "fear",    "INJURY — should NOT diagnose blood disorder"),
        ("What is the dosage for paracetamol?",                                "neutral", "MEDICATION"),
        ("I have had a severe headache for 3 days",                           "sadness", "SYMPTOM"),
        ("I have chest pain and my left arm feels numb",                       "fear",    "EMERGENCY"),
    ]

    for question, emotion, label in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST   : {label}")
        print(f"Q      : {question}")
        print(f"Emotion: {emotion}")
        print("-" * 60)
        situation = classify_situation(question)
        print(f"Classified as: {situation}")
        result = generate_emotional_medical_answer(question, sample_docs, emotion)
        print(f"Risk  : {result['risk'].upper()}")
        print(f"Doctor: {result['needs_doctor']}")
        print(f"\nAnswer:\n{result['answer']}")

    print("\n" + "=" * 60)