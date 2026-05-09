================================================================================
  AI MEDICARE ASSISTANT — README
  Design and Development of MediRAG-Link: A Cyber-Physical System for
  Grounded Medical Assistance and Automated Dispensing
================================================================================

  Group No  : 058
  Group Name: AI Medicare
  Members   : Omkar S Patil · P J Pragadeesh · P Ganeshkumar · B Jeevan Reddy
  Guide     : Dr. Ravishankar D
  University: REVA University

================================================================================
  WHAT IS THIS PROJECT?
================================================================================

The AI Medicare Assistant is a fully offline, locally running healthcare
platform built for elderly patients who need help managing medications and
getting answers to medical questions.

It combines three things into one system:

  1. EMOTION-AWARE AI MEDICAL CHAT
     You type a health question in plain language. The system detects how you
     are feeling (scared, angry, sad, happy, neutral) and gives a response
     that matches both your emotion and the seriousness of your question.
     Simple questions get short, friendly answers. Serious symptoms get
     urgent, clear guidance. Emergencies get a direct instruction to call
     for help immediately.

  2. SMART DRUG DISPENSER
     An Arduino microcontroller controls 3 servo motors — one per medication
     slot. At the scheduled time every day, the system automatically dispenses
     the right medicine without any human intervention. Everything is logged.

  3. SIMULATED DOCTOR CONNECTIVITY
     For high-risk or emergency queries, a "Connect to Doctor" button appears.
     The user can chat with a simulated specialist (General Physician,
     Cardiologist, or Diabetologist) powered by the same local AI model.
     This is a demonstration feature — real doctors can be integrated later.

Everything runs 100% on your local machine. No paid APIs. No internet needed
after the first-time model download. No patient data ever leaves your computer.

================================================================================
  PROJECT FOLDER STRUCTURE
================================================================================

  ai-medicare-assistant/
  │
  ├── backend/                        ← All Python server code lives here
  │   ├── main.py                     ← FastAPI server — start this to run everything
  │   ├── emotion_module.py           ← Detects emotion from user text
  │   ├── rag_pipeline.py             ← Retrieves medical docs from ChromaDB
  │   ├── response_generator.py       ← Generates human-like answers via Ollama
  │   ├── doctor_simulator.py         ← Simulated doctor chat (3 specialist profiles)
  │   ├── database.py                 ← SQLite manager for medications/schedules/logs
  │   ├── scheduler.py                ← Auto-dispenses meds at scheduled times
  │   ├── hardware_controller.py      ← Sends commands to Arduino via USB serial
  │   ├── parse_medquad.py            ← Parses MedQuAD XML files into .txt documents
  │   ├── build_vectors.py            ← Embeds all docs into ChromaDB vector store
  │   │
  │   ├── medical_docs/               ← Medical knowledge base
  │   │   ├── paracetamol.txt         ← Base document: Paracetamol
  │   │   ├── amoxicillin.txt         ← Base document: Amoxicillin
  │   │   ├── diabetes.txt            ← Base document: Diabetes
  │   │   ├── hypertension.txt        ← Base document: Hypertension
  │   │   └── medquad/                ← 9,800+ parsed MedQuAD documents (auto-created)
  │   │
  │   ├── chroma_db/                  ← Vector database (auto-created by build_vectors.py)
  │   ├── medicare.db                 ← SQLite database (auto-created on first run)
  │   └── venv/                       ← Python virtual environment (you create this)
  │
  ├── frontend/
  │   └── index.html                  ← React web app (open in browser)
  │
  └── MedQuAD/                        ← Cloned from GitHub (47,457 Q&A pairs)
      ├── 1_CancerGov_QA/
      ├── 2_GARD_QA/
      └── ... (11 folders total)

================================================================================
  SYSTEM REQUIREMENTS
================================================================================

  Operating System : Windows 10/11, macOS, or Linux
  Python           : 3.10 or higher
  RAM              : Minimum 8 GB (16 GB recommended for Mistral 7B)
  Storage          : Minimum 10 GB free
                     (Mistral model ~4 GB + MedQuAD ~150 MB + embeddings ~500 MB)
  Git              : Required to clone MedQuAD
  Ollama           : Required for local LLM (free, from ollama.com)
  Arduino IDE      : Required only if connecting physical hardware

================================================================================
  FIRST-TIME SETUP — COMPLETE STEP BY STEP
================================================================================

Follow these steps in order. Do them once only.

────────────────────────────────────────────────────────────────────────────────
STEP 1 — Create the project folder
────────────────────────────────────────────────────────────────────────────────

  Create a folder called:   ai-medicare-assistant
  Inside it, create:        backend/   and   frontend/   and   medical_docs/ inside backend/

  Or run in terminal:
    mkdir ai-medicare-assistant
    cd ai-medicare-assistant
    mkdir backend
    mkdir backend\medical_docs        (Windows)
    mkdir backend/medical_docs        (Mac/Linux)
    mkdir frontend

────────────────────────────────────────────────────────────────────────────────
STEP 2 — Place all code files
────────────────────────────────────────────────────────────────────────────────

  Copy these files into the backend/ folder:
    main.py
    emotion_module.py
    rag_pipeline.py
    response_generator.py
    doctor_simulator.py
    database.py
    scheduler.py
    hardware_controller.py
    parse_medquad.py
    build_vectors.py

  Copy these files into backend/medical_docs/:
    paracetamol.txt
    amoxicillin.txt
    diabetes.txt
    hypertension.txt

  Copy index.html into the frontend/ folder.

────────────────────────────────────────────────────────────────────────────────
STEP 3 — Create a Python virtual environment
────────────────────────────────────────────────────────────────────────────────

  Open a terminal and navigate to the backend folder:
    cd ai-medicare-assistant/backend

  Create the virtual environment:
    python -m venv venv

  Activate it:
    Windows  →  venv\Scripts\activate
    Mac/Linux→  source venv/bin/activate

  You will see (venv) at the start of your terminal line when it is active.

────────────────────────────────────────────────────────────────────────────────
STEP 4 — Install all Python dependencies
────────────────────────────────────────────────────────────────────────────────

  With the virtual environment active, run:

    pip install fastapi uvicorn
    pip install transformers torch
    pip install sentence-transformers chromadb
    pip install apscheduler
    pip install pyserial
    pip install ollama
    pip install lxml tqdm

  NOTE: Installing torch may take 5-10 minutes as it is a large package (~2 GB).
  Let it complete fully before moving on.

────────────────────────────────────────────────────────────────────────────────
STEP 5 — Install Ollama and download the Mistral model
────────────────────────────────────────────────────────────────────────────────

  1. Go to:  https://ollama.com
  2. Click Download for Windows (or Mac/Linux)
  3. Run the installer
  4. Open a NEW terminal (separate from the one with venv) and run:

       ollama pull mistral

  This downloads the Mistral 7B model (~4 GB). It will take 5-15 minutes
  depending on your internet speed. You only do this once.

  5. Verify it works:
       ollama run mistral "Say hello in one sentence"

  Press Ctrl+D or type /bye to exit.

  NOTE: On Windows, Ollama installs as a background service and starts
  automatically with Windows. You do NOT need to run "ollama serve" manually.
  You can see it running in the system tray (bottom right of taskbar).

────────────────────────────────────────────────────────────────────────────────
STEP 6 — Clone the MedQuAD medical dataset
────────────────────────────────────────────────────────────────────────────────

  In terminal, go to the ROOT folder (ai-medicare-assistant, NOT inside backend):
    cd ..      (if you are inside backend)

  Then run:
    git clone https://github.com/abachaa/MedQuAD.git

  This downloads ~150 MB of medical Q&A data into a MedQuAD/ folder.
  It contains 47,457 verified Q&A pairs from NIH, NCI, and MedlinePlus.

  After cloning, your folder should look like:
    ai-medicare-assistant/
    ├── backend/
    ├── frontend/
    └── MedQuAD/          ← this was just created

────────────────────────────────────────────────────────────────────────────────
STEP 7 — Parse MedQuAD into documents
────────────────────────────────────────────────────────────────────────────────

  Go back into the backend folder and make sure venv is active:
    cd backend
    venv\Scripts\activate     (Windows)
    source venv/bin/activate  (Mac/Linux)

  Run the parser:
    python parse_medquad.py

  This reads all MedQuAD XML files and converts them into .txt documents
  saved in backend/medical_docs/medquad/

  Expected output:
    [parser] Found 489 XML files in 1_CancerGov_QA
    [parser] Found 1143 XML files in 2_GARD_QA
    ...
    Documents created : ~9,800

────────────────────────────────────────────────────────────────────────────────
STEP 8 — Build the vector store
────────────────────────────────────────────────────────────────────────────────

  Run:
    python build_vectors.py

  This embeds all 9,800+ documents into ChromaDB.
  A chroma_db/ folder will be created in the backend directory.

  IMPORTANT: This step takes 15-30 minutes on first run. Let it complete.
  You will see a progress bar like:
    [████████████░░░░░░░░]  60.0%  batch 60/98  (6,000/9,804 docs)

  You only need to run this once (or again if you add new medical documents).

────────────────────────────────────────────────────────────────────────────────
STEP 9 — Test each module (optional but recommended)
────────────────────────────────────────────────────────────────────────────────

  Test emotion detection:
    python emotion_module.py

  Test the database:
    python database.py

  Test the hardware controller (simulation mode):
    python hardware_controller.py

  Test the response generator (requires Ollama running):
    python response_generator.py

  Test the doctor simulator (requires Ollama running):
    python doctor_simulator.py

================================================================================
  RUNNING THE SYSTEM — EVERY TIME YOU USE IT
================================================================================

  You only need to do these steps each time you want to use the system.
  The long setup steps above (Steps 1-8) are done only once.

────────────────────────────────────────────────────────────────────────────────
STEP 1 — Make sure Ollama is running
────────────────────────────────────────────────────────────────────────────────

  On Windows: Check the system tray (bottom right) for the Ollama icon.
  If it is there, Ollama is already running — nothing to do.

  On Mac/Linux: Open a terminal and run:
    ollama serve
  Leave this terminal open.

  Verify at any time by opening in browser:   http://localhost:11434
  You should see:   Ollama is running

────────────────────────────────────────────────────────────────────────────────
STEP 2 — Start the backend server
────────────────────────────────────────────────────────────────────────────────

  Open a terminal, navigate to backend/, activate venv, and run main.py:

    cd ai-medicare-assistant/backend
    venv\Scripts\activate              (Windows)
    source venv/bin/activate           (Mac/Linux)
    python main.py

  You should see:
    ============================================================
      AI Medicare Assistant — Starting Up
    ============================================================
    [main] Setting up database...
    [main] Checking Ollama LLM...
    [main] Ollama is ready — human-like responses enabled!
    [main] Starting medication scheduler...
    [main] Server is ready!
    [main] API docs: http://localhost:8000/docs

  Leave this terminal open while using the system.

────────────────────────────────────────────────────────────────────────────────
STEP 3 — Open the frontend
────────────────────────────────────────────────────────────────────────────────

  Open the frontend/index.html file in your browser:

  Option A (simplest):
    Double-click index.html in your file explorer.
    It opens directly in the browser.

  Option B (recommended, with auto-refresh):
    In Cursor or VS Code, install the "Live Server" extension.
    Right-click index.html → "Open with Live Server"
    Browser opens at:  http://127.0.0.1:5500

  You should see the AI Medicare Assistant interface with 7 screens in the
  left sidebar:
    Medical Chat · Doctor Chat · Dashboard · Medications · Schedules ·
    Adherence Logs · Dispenser

  The green dot at the bottom left means the backend is connected.

================================================================================
  USING THE SYSTEM — FEATURE GUIDE
================================================================================

────────────────────────────────────────
MEDICAL CHAT
────────────────────────────────────────
  Type any health or medication question in plain language.
  Click Send or press Enter.

  The AI will:
  - Detect your emotion (shown as a badge: FEAR, SADNESS, NEUTRAL, etc.)
  - Assess the risk level of your query (LOW / MEDIUM / HIGH / EMERGENCY)
  - Retrieve relevant medical documents from the local knowledge base
  - Generate a simple, warm response via Mistral 7B

  Response style by risk level:
    LOW       → 2-3 sentences, friendly, no alarming warnings
    MEDIUM    → 3-4 sentences, informative, one caution
    HIGH      → 4-5 sentences, serious tone, urge doctor visit today
    EMERGENCY → 2 sentences, call ambulance immediately

  Quick-start chips at the top of the chat let you try example queries.

────────────────────────────────────────
DOCTOR CHAT
────────────────────────────────────────
  Available when risk level is HIGH or EMERGENCY.
  A "Connect to a Doctor" button appears automatically in the chat.

  Three specialist profiles are available:
    Dr. Priya Nair      — General Physician (12 years, City Medical Centre)
    Dr. Arjun Mehta     — Cardiologist (18 years, Heart Care Hospital)
    Dr. Suma Krishnan   — Diabetologist & Endocrinologist (15 years)

  Select a doctor and click Connect. A realistic typing animation plays
  before each response. The full conversation history is maintained.
  Click "End Session" to close the doctor chat.

  NOTE: This is a simulated feature for demonstration. The doctors are
  powered by the local Mistral 7B model, not real people.

────────────────────────────────────────
MEDICATIONS
────────────────────────────────────────
  Add up to 3 medications — one per dispenser slot.
  Each medication must be assigned to Slot 1, 2, or 3.
  The slot selection shows which pin the servo is on (Pin 9, 10, or 11).

  Fields:
    Medicine Name   — e.g. Paracetamol
    Dosage          — e.g. 500mg
    Frequency       — Once/Twice/Three times daily, As needed
    Dispenser Slot  — 1, 2, or 3 (each maps to an Arduino servo pin)
    Notes           — Optional, e.g. "take after food"

  To remove a medication, click the red "Remove" button next to it.
  This frees the slot for a new medicine.
  Removing a medication also deletes all its schedules.

────────────────────────────────────────
SCHEDULES
────────────────────────────────────────
  Set the time each medicine should be dispensed.
  Enter the medicine name (must match exactly), time in 24-hour format,
  and which days (Every Day, Weekdays Only, etc.)

  The scheduler checks every 60 seconds. When the current time matches
  a schedule, it:
    1. Prints a reminder to the server terminal
    2. Triggers the hardware dispenser (or simulation if no Arduino)
    3. Logs the dose as "Taken" in the adherence table

────────────────────────────────────────
ADHERENCE LOGS
────────────────────────────────────────
  Shows every dispensing event with:
    Medicine name · Scheduled time · Status (Taken/Missed) · Exact timestamp

  Click Refresh to update the view with the latest entries.
  Logs are stored permanently in the medicare.db SQLite database.

────────────────────────────────────────
SMART DISPENSER
────────────────────────────────────────
  Shows exactly 3 slot cards — filled or empty.
  Each filled slot shows the medicine name, dosage, frequency, and Arduino pin.

  Click the Dispense button to immediately trigger that slot's servo motor.
  A success/failure result is shown below the cards.

  Works in simulation mode (prints command to terminal) if Arduino is not
  connected. Switch to real hardware by connecting Arduino and editing
  SERIAL_PORT in hardware_controller.py.

================================================================================
  CONNECTING THE ARDUINO HARDWARE
================================================================================

────────────────────────────────────────────────────────────────────────────────
Hardware Required
────────────────────────────────────────────────────────────────────────────────
  - 1x Arduino Uno (or compatible board)
  - 3x Servo motors (standard 5V hobby servos)
  - USB A to USB B cable (to connect Arduino to PC)
  - 3D printed or physical dispenser housing with 3 compartments

────────────────────────────────────────────────────────────────────────────────
Wiring
────────────────────────────────────────────────────────────────────────────────
  Connect servos to the Arduino:
    Servo 1 (Slot 1)  →  Signal wire to Pin 9,  VCC to 5V,  GND to GND
    Servo 2 (Slot 2)  →  Signal wire to Pin 10, VCC to 5V,  GND to GND
    Servo 3 (Slot 3)  →  Signal wire to Pin 11, VCC to 5V,  GND to GND

────────────────────────────────────────────────────────────────────────────────
Upload Arduino Sketch
────────────────────────────────────────────────────────────────────────────────
  Open the Arduino IDE.
  Copy the sketch from the bottom of hardware_controller.py (Section 8).
  Select your board: Tools → Board → Arduino Uno
  Select your port: Tools → Port → COM3 (or whatever port your Arduino uses)
  Click Upload.

────────────────────────────────────────────────────────────────────────────────
Configure the COM Port in Python
────────────────────────────────────────────────────────────────────────────────
  Open hardware_controller.py and change line 28:

    SERIAL_PORT = "COM3"     ← change to your actual port (COM4, COM5, etc.)

  To find your port:
  - Windows: Open Device Manager → Ports (COM & LPT) → look for Arduino
  - Mac/Linux: Run:  ls /dev/tty.* or ls /dev/ttyACM*

  After changing the port, restart the server (python main.py).

────────────────────────────────────────────────────────────────────────────────
Test the connection
────────────────────────────────────────────────────────────────────────────────
  With the server running, go to the frontend Dispenser page and click
  a Dispense button. If wired correctly, the corresponding servo will rotate
  90 degrees, pause for 1 second, and return to 0 degrees.

  The server terminal will show:
    [hardware] Connected to Arduino on COM3
    [hardware] Sending command: DISPENSE_SLOT1
    [hardware] Arduino response: OK_SLOT1

================================================================================
  API ENDPOINTS REFERENCE
================================================================================

  All endpoints are served at:  http://localhost:8000
  Interactive docs (Swagger UI): http://localhost:8000/docs

  ┌─────────────────────────────┬────────────┬─────────────────────────────────┐
  │ Endpoint                    │ Method     │ Description                     │
  ├─────────────────────────────┼────────────┼─────────────────────────────────┤
  │ /                           │ GET        │ Welcome message                 │
  │ /health                     │ GET        │ Server health check             │
  │ /chat                       │ POST       │ Main AI medical chat            │
  │ /doctors                    │ GET        │ List available doctors          │
  │ /doctor-chat                │ POST       │ Chat with simulated doctor      │
  │ /medications                │ GET        │ List all medications            │
  │ /medications                │ POST       │ Add a new medication            │
  │ /medications/{name}         │ DELETE     │ Remove a medication             │
  │ /medications/slots          │ GET        │ Show available slot numbers     │
  │ /medications/schedule       │ POST       │ Add a dispensing schedule       │
  │ /medications/schedules      │ GET        │ List all schedules              │
  │ /adherence                  │ GET        │ Get adherence logs              │
  │ /adherence/log              │ POST       │ Manually log a dose             │
  │ /dispense                   │ POST       │ Dispense by medicine name       │
  │ /dispense/slot              │ POST       │ Dispense by slot number (1/2/3) │
  └─────────────────────────────┴────────────┴─────────────────────────────────┘

================================================================================
  COMMON ERRORS AND FIXES
================================================================================

  ┌────────────────────────────────────────────────────────────────────────────┐
  │ Error                          │ Fix                                       │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ "python" not recognized        │ Try "python3" instead of "python"         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ venv\Scripts\activate fails    │ Run in PowerShell:                        │
  │ (Windows)                      │ Set-ExecutionPolicy RemoteSigned          │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Ollama error: only one usage   │ Ollama is already running. Close the      │
  │ of each socket address         │ terminal where you ran "ollama serve"     │
  │                                │ — it runs automatically in the background │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Ollama not ready at startup    │ Check system tray for Ollama icon.        │
  │                                │ If not there, open Ollama app manually.   │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Model not found: mistral       │ Run: ollama pull mistral                  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ MedQuAD folder not found       │ Make sure you cloned MedQuAD inside       │
  │                                │ ai-medicare-assistant/ (not inside        │
  │                                │ backend/). Check path in parse_medquad.py │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ ImportError: cannot import     │ Your file was not saved completely.       │
  │ from rag_pipeline              │ Re-paste the full code and save again.    │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ "Server offline" in frontend   │ Backend is not running. Open terminal,    │
  │                                │ cd backend, activate venv, python main.py │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Cannot connect to Arduino      │ Check COM port in hardware_controller.py  │
  │                                │ and verify Arduino is plugged in.         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ Old database schema error      │ Delete medicare.db and restart the server.│
  │                                │ Tables will be recreated automatically.   │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ build_vectors.py only loads    │ You have the old version of the file.     │
  │ 4 documents                    │ Replace it with the updated version that  │
  │                                │ uses rglob() to scan all subfolders.      │
  └────────────────────────────────────────────────────────────────────────────┘

================================================================================
  TECHNOLOGY STACK SUMMARY
================================================================================

  COMPONENT              TECHNOLOGY
  ─────────────────────────────────────────────────────────────────────────────
  Backend framework      FastAPI + Uvicorn (Python)
  LLM (local)            Mistral 7B via Ollama
  Emotion detection      j-hartmann/emotion-english-distilroberta-base (HuggingFace)
  Embeddings             all-MiniLM-L6-v2 (sentence-transformers)
  Vector database        ChromaDB (local, persistent)
  Medical dataset        MedQuAD — 47,457 Q&A pairs (NIH, NCI, MedlinePlus)
  Relational database    SQLite (via Python sqlite3 — no install needed)
  Task scheduler         APScheduler BackgroundScheduler
  Serial communication   PySerial
  Microcontroller        Arduino Uno
  Hardware actuators     3x Servo motors (pins 9, 10, 11)
  Frontend               React (single HTML file — no build step needed)
  API documentation      FastAPI Swagger UI (auto-generated)
  ─────────────────────────────────────────────────────────────────────────────

================================================================================
  QUICK REFERENCE — STARTUP COMMANDS
================================================================================

  Every time you want to use the system, run these commands:

  # Terminal 1 — Start backend
  cd ai-medicare-assistant/backend
  venv\Scripts\activate             (Windows)
  source venv/bin/activate          (Mac/Linux)
  python main.py

  # Then open frontend in browser
  Open:  frontend/index.html

  # That's it. Ollama runs automatically in the background on Windows.
  # On Mac/Linux, start Ollama first: ollama serve (in a separate terminal)

================================================================================
  FIRST-TIME SETUP — COMMANDS ONLY (QUICK REFERENCE)
================================================================================

  cd ai-medicare-assistant/backend
  python -m venv venv
  venv\Scripts\activate
  pip install fastapi uvicorn transformers torch sentence-transformers chromadb apscheduler pyserial ollama lxml tqdm
  cd ..
  git clone https://github.com/abachaa/MedQuAD.git
  cd backend
  python parse_medquad.py
  python build_vectors.py
  python main.py

================================================================================
  NOTES FOR DEVELOPERS
================================================================================

  - All modules are independently testable by running them directly:
      python emotion_module.py       (tests emotion detection)
      python rag_pipeline.py         (tests document retrieval)
      python database.py             (tests SQLite operations)
      python scheduler.py            (tests auto-scheduling loop)
      python hardware_controller.py  (tests dispenser simulation)
      python response_generator.py   (tests Ollama response pipeline)
      python doctor_simulator.py     (tests simulated doctor chat)

  - To add more medical documents, place .txt files in medical_docs/ and
    run python build_vectors.py again. New documents will be upserted
    without recreating the entire vector store.

  - To add a new doctor profile, open doctor_simulator.py and add a new
    entry to the SIMULATED_DOCTORS list following the existing format.

  - To change the LLM model, edit MODEL_NAME in response_generator.py.
    Other supported Ollama models: llama3, gemma, phi3
    Run: ollama pull <model_name> first.

  - The sqlite database (medicare.db) can be viewed with any SQLite browser
    such as DB Browser for SQLite (free, from sqlitebrowser.org).

  - The chroma_db/ folder should NOT be manually edited. If it becomes
    corrupted, delete it and re-run build_vectors.py.

================================================================================
  LICENSE AND ACKNOWLEDGEMENTS
================================================================================

  This project was developed as an academic project at REVA University.

  Medical knowledge sourced from:
    MedQuAD Dataset — Asma Ben Abacha and Dina Demner-Fushman
    https://github.com/abachaa/MedQuAD
    Sources: NIH, NCI, NIDDK, MedlinePlus, Genetics Home Reference, CDC

  AI Models:
    Mistral 7B      — mistral.ai (open-weight, Apache 2.0 license)
    DistilRoBERTa   — j-hartmann/emotion-english-distilroberta-base (HuggingFace)
    MiniLM-L6-v2    — sentence-transformers (Apache 2.0 license)

  This system is intended for educational and demonstration purposes only.
  It is NOT a substitute for professional medical advice, diagnosis, or
  treatment. Always consult a qualified healthcare provider for medical
  decisions.

================================================================================
  END OF README
================================================================================
