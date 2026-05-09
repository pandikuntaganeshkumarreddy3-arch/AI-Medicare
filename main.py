"""
main.py  (UPDATED)
-------
FastAPI Backend — AI Medicare Assistant.

New in this version:
  - 3-slot medication system with slot assignment
  - DELETE /medications/{name} endpoint to remove medications
  - GET /medications/slots endpoint to see available slots
  - Risk-aware chat responses (low/medium/high/emergency)
  - POST /doctor-chat endpoint for simulated doctor conversations
  - GET /doctors endpoint to list available doctors
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from emotion_module      import detect_emotion, map_emotion_to_tone
from rag_pipeline        import retrieve_relevant_docs
from response_generator  import generate_emotional_medical_answer, check_ollama
from doctor_simulator    import get_available_doctors, get_doctor_response
from database            import (
    create_tables,
    add_medication,
    remove_medication,
    get_medications,
    get_medication_count,
    get_available_slots,
    add_schedule,
    get_schedules,
    log_adherence,
    get_adherence_logs,
)
from scheduler           import start_scheduler
from hardware_controller import dispense, dispense_slot


# ──────────────────────────────────────────────────────────────
# 1.  Startup / Shutdown
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("  🏥 AI Medicare Assistant — Starting Up")
    print("=" * 60)

    print("[main] Setting up database…")
    create_tables()

    print("[main] Checking Ollama LLM…")
    ollama_ready = check_ollama()
    if ollama_ready:
        print("[main] ✅ Ollama ready — human-like responses enabled!")
    else:
        print("[main] ⚠️  Ollama not found — fallback responses will be used.")
        print("[main]    Run: ollama pull mistral  then restart the server.")

    print("[main] Starting medication scheduler…")
    app.state.scheduler = start_scheduler()

    print("[main] ✅ Server is ready!")
    print("[main] 📖 API docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield

    print("\n[main] Shutting down scheduler…")
    app.state.scheduler.shutdown()
    print("[main] Goodbye! 👋")


# ──────────────────────────────────────────────────────────────
# 2.  App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AI Medicare Assistant",
    description = "Emotion & Risk Aware Medical Chat with Smart Drug Dispenser",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ──────────────────────────────────────────────────────────────
# 3.  Request / Response models
# ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    class Config:
        json_schema_extra = {"example": {"question": "I have chest pain"}}

class ChatResponse(BaseModel):
    emotion:      str
    tone:         str
    risk:         str
    needs_doctor: bool
    answer:       str

class MedicationRequest(BaseModel):
    medicine_name: str
    dosage:        str
    frequency:     str
    slot:          int   # 1, 2, or 3
    notes:         str = ""
    class Config:
        json_schema_extra = {
            "example": {
                "medicine_name": "Paracetamol",
                "dosage":        "500mg",
                "frequency":     "twice daily",
                "slot":          1,
                "notes":         "take after food",
            }
        }

class ScheduleRequest(BaseModel):
    medicine_name:  str
    scheduled_time: str
    days_of_week:   str = "everyday"
    class Config:
        json_schema_extra = {
            "example": {
                "medicine_name":  "Paracetamol",
                "scheduled_time": "08:00",
                "days_of_week":   "everyday",
            }
        }

class DispenseSlotRequest(BaseModel):
    slot: int   # 1, 2, or 3
    class Config:
        json_schema_extra = {"example": {"slot": 1}}

class DispenseNameRequest(BaseModel):
    medicine_name: str
    class Config:
        json_schema_extra = {"example": {"medicine_name": "Paracetamol"}}

class DoctorChatRequest(BaseModel):
    doctor_id:       int
    patient_message: str
    chat_history:    list[dict] = []
    original_query:  str        = ""
    class Config:
        json_schema_extra = {
            "example": {
                "doctor_id":       1,
                "patient_message": "Doctor, I have had chest pain since morning.",
                "chat_history":    [],
                "original_query":  "chest pain",
            }
        }


# ──────────────────────────────────────────────────────────────
# 4.  Routes
# ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "🏥 AI Medicare Assistant v2.0 is running!",
        "docs":    "http://localhost:8000/docs",
    }

@app.get("/health")
def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Chat ──────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint with full pipeline:
    emotion detection → RAG retrieval → risk analysis → Ollama response.

    Returns risk level and needs_doctor flag so the frontend
    can show the Connect Doctor button when appropriate.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print(f"\n[main] /chat → '{question}'")

    # Detect emotion
    emotion = detect_emotion(question)
    tone    = map_emotion_to_tone(emotion)

    # Retrieve medical docs
    docs = retrieve_relevant_docs(question, k=3)

    # Generate risk-aware, human-like answer
    result = generate_emotional_medical_answer(
        question = question,
        docs     = docs,
        emotion  = emotion,
    )

    print(f"[main] Done — emotion={emotion} risk={result['risk']} needs_doctor={result['needs_doctor']}")

    return ChatResponse(
        emotion      = emotion,
        tone         = tone,
        risk         = result["risk"],
        needs_doctor = result["needs_doctor"],
        answer       = result["answer"],
    )


# ── Doctor Chat ───────────────────────────────────────────────

@app.get("/doctors")
def list_doctors():
    """List all available simulated doctors."""
    doctors = get_available_doctors()
    return {
        "count":   len(doctors),
        "doctors": doctors,
    }

@app.post("/doctor-chat")
def doctor_chat(request: DoctorChatRequest):
    """
    Chat with a simulated doctor.

    The frontend should:
      1. Show the list of doctors from GET /doctors
      2. Let the user select one
      3. Send messages here with the full chat history each time
      4. Show a typing animation for typing_delay seconds before
         displaying the doctor's response
    """
    result = get_doctor_response(
        doctor_id       = request.doctor_id,
        patient_message = request.patient_message,
        chat_history    = request.chat_history,
        original_query  = request.original_query,
    )
    return result


# ── Medications ───────────────────────────────────────────────

@app.get("/medications")
def list_medications():
    """List all medications (max 3, ordered by slot)."""
    medications = get_medications()
    return {
        "count":           len(medications),
        "max_slots":       3,
        "available_slots": get_available_slots(),
        "medications":     medications,
    }

@app.get("/medications/slots")
def available_slots():
    """Return which slot numbers (1/2/3) are still available."""
    return {
        "available_slots": get_available_slots(),
        "taken_slots":     [s for s in [1,2,3] if s not in get_available_slots()],
    }

@app.post("/medications")
def create_medication(request: MedicationRequest):
    """
    Add a new medication.
    Maximum 3 medications allowed.
    Slot must be 1, 2, or 3 and must not be already taken.
    """
    result = add_medication(
        medicine_name = request.medicine_name,
        dosage        = request.dosage,
        frequency     = request.frequency,
        slot          = request.slot,
        notes         = request.notes,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.delete("/medications/{medicine_name}")
def delete_medication(medicine_name: str):
    """
    Remove a medication by name.
    Also removes all its schedules.
    This frees up the slot for a new medicine.
    """
    result = remove_medication(medicine_name)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


# ── Schedules ─────────────────────────────────────────────────

@app.post("/medications/schedule")
def create_schedule(request: ScheduleRequest):
    """Add a dosage schedule for a medication."""
    result = add_schedule(
        medicine_name  = request.medicine_name,
        scheduled_time = request.scheduled_time,
        days_of_week   = request.days_of_week,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.get("/medications/schedules")
def list_schedules():
    """Get all medication schedules."""
    schedules = get_schedules()
    return {"count": len(schedules), "schedules": schedules}


# ── Adherence ─────────────────────────────────────────────────

@app.get("/adherence")
def list_adherence(medicine_name: Optional[str] = None):
    """Get adherence logs, optionally filtered by medicine name."""
    logs = get_adherence_logs(medicine_name)
    return {"count": len(logs), "logs": logs}

@app.post("/adherence/log")
def manual_adherence_log(medicine_name: str, scheduled_time: str, taken: bool):
    """Manually log whether a medication was taken."""
    try:
        log_adherence(medicine_name, scheduled_time, taken)
        return {"success": True, "message": f"Adherence logged for {medicine_name}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dispenser ─────────────────────────────────────────────────

@app.post("/dispense/slot")
def dispense_by_slot(request: DispenseSlotRequest):
    """
    Trigger the dispenser for a specific slot number (1, 2, or 3).

    Slot → Motor mapping (28BYJ-48 via ULN2003):
      Slot 1 → pins 4, 5, 6, 7   (IN1-IN4 of ULN2003 board 1)
      Slot 2 → pins 8, 9, 10, 11 (IN1-IN4 of ULN2003 board 2)
      Slot 3 → not wired (simulation only)
    """
    if request.slot not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Slot must be 1, 2, or 3.")
    result = dispense_slot(request.slot)
    # Normalise response so frontend always gets both pin and pins
    result.setdefault("pin",  result.get("pins", [None])[0] if result.get("pins") else None)
    result.setdefault("pins", [result["pin"]] if result.get("pin") else [])
    return result

@app.post("/dispense")
def dispense_by_name(request: DispenseNameRequest):
    """
    Trigger the dispenser for a medicine by name.
    Looks up which slot the medicine is assigned to, then fires that motor.
    """
    result = dispense(request.medicine_name)
    # Normalise keys
    result.setdefault("pin",  result.get("pins", [None])[0] if result.get("pins") else None)
    result.setdefault("pins", [result["pin"]] if result.get("pin") else [])
    if not result["success"] and result.get("mode") == "error":
        raise HTTPException(status_code=404, detail=result["response"])
    return result


# ──────────────────────────────────────────────────────────────
# 5.  Run
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)