"""
doctor_simulator.py
-------------------
Simulated Doctor Chat System.

For demonstration purposes, simulates a list of certified doctors
that users can connect to when a HIGH or EMERGENCY risk query is detected.

In a real deployment:
  - Replace SIMULATED_DOCTORS with real doctor profiles from a database
  - Replace the Ollama simulation with a real messaging/video call system
  - Add authentication and consent flows

Currently:
  - Shows a list of 3 simulated doctors with specialties
  - User selects a doctor
  - Ollama LLM plays the role of that doctor
  - Shows a realistic "doctor is typing" delay
  - Responds in a professional but simple medical style
"""

import ollama
import time

MODEL_NAME = "mistral"

# ──────────────────────────────────────────────────────────────
# 1.  Simulated doctor profiles
#     In production: load these from a database with real doctors
#     who have given permission to be listed
# ──────────────────────────────────────────────────────────────

SIMULATED_DOCTORS = [
    {
        "id":         1,
        "name":       "Dr. Priya Nair",
        "specialty":  "General Physician",
        "experience": "12 years",
        "hospital":   "City Medical Centre",
        "available":  True,
        "avatar":     "👩‍⚕️",
        "persona": (
            "You are Dr. Priya Nair, a warm and experienced General Physician. "
            "You speak simply, gently, and clearly to elderly patients. "
            "You ask one follow-up question at a time. "
            "You always recommend in-person consultation for serious issues."
        ),
    },
    {
        "id":         2,
        "name":       "Dr. Arjun Mehta",
        "specialty":  "Cardiologist",
        "experience": "18 years",
        "hospital":   "Heart Care Hospital",
        "available":  True,
        "avatar":     "👨‍⚕️",
        "persona": (
            "You are Dr. Arjun Mehta, a senior Cardiologist. "
            "You are professional yet kind. You speak clearly in simple terms. "
            "You are especially careful about heart and blood pressure issues. "
            "You always ask about current medications and symptoms."
        ),
    },
    {
        "id":         3,
        "name":       "Dr. Suma Krishnan",
        "specialty":  "Diabetologist & Endocrinologist",
        "experience": "15 years",
        "hospital":   "Wellness Diabetes Clinic",
        "available":  True,
        "avatar":     "👩‍⚕️",
        "persona": (
            "You are Dr. Suma Krishnan, a specialist in diabetes and hormonal conditions. "
            "You are patient and thorough. You use very simple words. "
            "You always ask about diet, blood sugar levels, and current medications."
        ),
    },
]


# ──────────────────────────────────────────────────────────────
# 2.  Get list of available doctors
# ──────────────────────────────────────────────────────────────

def get_available_doctors() -> list[dict]:
    """
    Return list of available doctors for the frontend to display.
    Returns only safe-to-display fields (no persona prompt).
    """
    return [
        {
            "id":         d["id"],
            "name":       d["name"],
            "specialty":  d["specialty"],
            "experience": d["experience"],
            "hospital":   d["hospital"],
            "available":  d["available"],
            "avatar":     d["avatar"],
        }
        for d in SIMULATED_DOCTORS
        if d["available"]
    ]


def get_doctor_by_id(doctor_id: int) -> dict | None:
    """Return a full doctor profile by ID, or None if not found."""
    for doc in SIMULATED_DOCTORS:
        if doc["id"] == doctor_id:
            return doc
    return None


# ──────────────────────────────────────────────────────────────
# 3.  Simulate doctor response using Ollama
# ──────────────────────────────────────────────────────────────

def get_doctor_response(
    doctor_id:       int,
    patient_message: str,
    chat_history:    list[dict],
    original_query:  str = "",
) -> dict:
    """
    Generate a simulated doctor response using the Ollama LLM.

    The LLM is given the doctor's persona and the full conversation
    history so it can respond in character as that specific doctor.

    Parameters
    ----------
    doctor_id       : int         ID of the doctor to simulate.
    patient_message : str         Latest message from the patient.
    chat_history    : list[dict]  Previous messages in this chat session.
                                  Format: [{"role": "user"/"doctor", "content": "..."}]
    original_query  : str         The original high-risk query that triggered
                                  the doctor connection (for context).

    Returns
    -------
    dict with keys:
        success  : bool
        response : str   The doctor's reply
        doctor   : dict  Doctor profile info
        typing_delay : float  Seconds the frontend should show "typing" animation
    """
    doctor = get_doctor_by_id(doctor_id)
    if not doctor:
        return {
            "success":     False,
            "response":    "Doctor not found.",
            "doctor":      None,
            "typing_delay": 0,
        }

    # Build message history for Ollama
    messages = [
        {
            "role":    "system",
            "content": (
                f"{doctor['persona']}\n\n"
                f"IMPORTANT RULES:\n"
                f"1. You are in a TEXT CHAT with an elderly patient — keep responses SHORT (3-4 sentences max).\n"
                f"2. Ask only ONE question at a time.\n"
                f"3. Use VERY SIMPLE words — no complex medical terminology.\n"
                f"4. Be warm, calm, and reassuring.\n"
                f"5. If the situation sounds serious, advise them to come in person or call emergency services.\n"
                f"6. Always introduce yourself warmly at the start of the conversation.\n"
                f"7. Remember: this is a SIMULATION for demonstration purposes.\n\n"
                f"The patient originally asked about: '{original_query}'"
            ),
        }
    ]

    # Add conversation history
    for msg in chat_history[-10:]:   # keep last 10 messages for context
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": patient_message})

    try:
        print(f"[doctor_simulator] Dr. {doctor['name']} responding…")

        response = ollama.chat(
            model    = MODEL_NAME,
            messages = messages,
            options  = {
                "temperature": 0.8,
                "num_predict": 200,   # keep doctor responses concise
            }
        )

        reply = response["message"]["content"].strip()

        # Calculate realistic typing delay (1.5 to 3 seconds)
        # Based on response length — longer reply = longer "typing" time
        typing_delay = min(1.5 + len(reply) / 200, 3.0)

        print(f"[doctor_simulator] Response ready ({len(reply)} chars, delay={typing_delay:.1f}s)")

        return {
            "success":     True,
            "response":    reply,
            "doctor":      {
                "id":        doctor["id"],
                "name":      doctor["name"],
                "specialty": doctor["specialty"],
                "avatar":    doctor["avatar"],
            },
            "typing_delay": typing_delay,
        }

    except Exception as e:
        print(f"[doctor_simulator] Ollama error: {e}")
        return {
            "success":     False,
            "response":    (
                f"I apologise, I am having technical difficulties right now. "
                f"Please call our clinic directly or visit the nearest hospital "
                f"if this is urgent."
            ),
            "doctor":      {
                "id":        doctor["id"],
                "name":      doctor["name"],
                "specialty": doctor["specialty"],
                "avatar":    doctor["avatar"],
            },
            "typing_delay": 1.5,
        }


# ──────────────────────────────────────────────────────────────
# 4.  Self test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Doctor Simulator — Self Test")
    print("=" * 55 + "\n")

    print("--- Available Doctors ---")
    for doc in get_available_doctors():
        status = "✅ Available" if doc["available"] else "❌ Unavailable"
        print(f"  {doc['avatar']} {doc['name']} | {doc['specialty']} | {doc['experience']} | {status}")

    print("\n--- Simulating conversation with Dr. Priya Nair ---\n")

    history = []
    test_messages = [
        "Hello doctor, I have been having severe chest pain since this morning.",
        "The pain is on the left side and sometimes goes to my arm.",
    ]

    for msg in test_messages:
        print(f"Patient : {msg}")
        result = get_doctor_response(
            doctor_id       = 1,
            patient_message = msg,
            chat_history    = history,
            original_query  = "chest pain",
        )
        history.append({"role": "user",   "content": msg})
        history.append({"role": "doctor", "content": result["response"]})

        print(f"Dr. Nair: {result['response']}")
        print(f"(typing delay: {result['typing_delay']:.1f}s)")
        print()

    print("✅ Doctor simulator test complete!")
    print("=" * 55)