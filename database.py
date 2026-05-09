"""
database.py
-----------
SQLite Medication Database Manager.
Updated to support:
  - Maximum 3 medications (matching 3 dispenser slots)
  - Slot assignment (1, 2, or 3) per medication
  - Slot uniqueness (no two medicines in the same slot)
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH  = Path(__file__).parent / "medicare.db"
MAX_MEDS = 3   # Hard limit matching the 3 dispenser slots


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────────────────────────────
# Table creation
# ──────────────────────────────────────────────────────────────

def create_tables() -> None:
    """
    Create all required tables.

    medications table now includes:
      slot  INTEGER  1, 2, or 3 — which dispenser slot this medicine uses
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medications (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name  TEXT    NOT NULL UNIQUE,
            dosage         TEXT    NOT NULL,
            frequency      TEXT    NOT NULL,
            notes          TEXT,
            slot           INTEGER UNIQUE,
            created_at     TEXT    DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name  TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            days_of_week   TEXT NOT NULL DEFAULT 'everyday',
            FOREIGN KEY (medicine_name) REFERENCES medications(medicine_name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS adherence_logs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name  TEXT    NOT NULL,
            scheduled_time TEXT    NOT NULL,
            taken          INTEGER NOT NULL DEFAULT 0,
            logged_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print("[database] Tables created (or already exist).")


# ──────────────────────────────────────────────────────────────
# Medication functions
# ──────────────────────────────────────────────────────────────

def get_medication_count() -> int:
    """Return how many medications are currently stored."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM medications")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def is_slot_taken(slot: int) -> bool:
    """Return True if the given slot (1/2/3) is already assigned."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM medications WHERE slot = ?", (slot,))
    taken = cursor.fetchone() is not None
    conn.close()
    return taken


def add_medication(
    medicine_name: str,
    dosage:        str,
    frequency:     str,
    slot:          int,
    notes:         str = "",
) -> dict:
    """
    Add a new medication to the database.

    Rules enforced:
      - Maximum 3 medications allowed (one per slot)
      - Slot must be 1, 2, or 3
      - Slot must not already be taken

    Parameters
    ----------
    medicine_name : str   Name of the medicine.
    dosage        : str   Dosage amount e.g. "500mg".
    frequency     : str   How often e.g. "twice daily".
    slot          : int   Dispenser slot number: 1, 2, or 3.
    notes         : str   Optional extra info.

    Returns
    -------
    dict  { success: bool, message: str }
    """
    # Validate slot number
    if slot not in [1, 2, 3]:
        return {"success": False, "message": "Slot must be 1, 2, or 3."}

    # Check total medication limit
    if get_medication_count() >= MAX_MEDS:
        return {
            "success": False,
            "message": (
                f"Maximum limit of {MAX_MEDS} medications reached. "
                "Please remove one before adding a new one."
            )
        }

    # Check if slot is already taken
    if is_slot_taken(slot):
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT medicine_name FROM medications WHERE slot = ?", (slot,))
        existing = cursor.fetchone()
        conn.close()
        return {
            "success": False,
            "message": (
                f"Slot {slot} is already assigned to "
                f"'{existing['medicine_name']}'. "
                "Please choose a different slot or remove that medicine first."
            )
        }

    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO medications (medicine_name, dosage, frequency, notes, slot)
            VALUES (?, ?, ?, ?, ?)
        """, (medicine_name, dosage, frequency, notes, slot))
        conn.commit()
        print(f"[database] Medication added: {medicine_name} → Slot {slot}")
        return {
            "success": True,
            "message": f"'{medicine_name}' added to Slot {slot} successfully."
        }
    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": f"'{medicine_name}' already exists in the database."
        }
    finally:
        conn.close()


def remove_medication(medicine_name: str) -> dict:
    """
    Remove a medication and all its schedules from the database.

    Parameters
    ----------
    medicine_name : str   Name of the medicine to remove.

    Returns
    -------
    dict  { success: bool, message: str }
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Check it exists
    cursor.execute(
        "SELECT id FROM medications WHERE medicine_name = ?",
        (medicine_name,)
    )
    if not cursor.fetchone():
        conn.close()
        return {
            "success": False,
            "message": f"'{medicine_name}' not found in database."
        }

    # Delete schedules first (foreign key)
    cursor.execute(
        "DELETE FROM schedules WHERE medicine_name = ?",
        (medicine_name,)
    )
    # Delete the medication
    cursor.execute(
        "DELETE FROM medications WHERE medicine_name = ?",
        (medicine_name,)
    )
    conn.commit()
    conn.close()
    print(f"[database] Medication removed: {medicine_name}")
    return {
        "success": True,
        "message": f"'{medicine_name}' and its schedules removed successfully."
    }


def get_medications() -> list[dict]:
    """Retrieve all medications ordered by slot number."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM medications ORDER BY slot")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_medication_by_name(medicine_name: str) -> dict | None:
    """Find a single medication by name. Returns dict or None."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM medications WHERE medicine_name = ?",
        (medicine_name,)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_available_slots() -> list[int]:
    """Return list of slot numbers (1/2/3) that are not yet assigned."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT slot FROM medications WHERE slot IS NOT NULL")
    taken = {row[0] for row in cursor.fetchall()}
    conn.close()
    return [s for s in [1, 2, 3] if s not in taken]


# ──────────────────────────────────────────────────────────────
# Schedule functions  (max 3 — one per medication)
# ──────────────────────────────────────────────────────────────

def add_schedule(
    medicine_name:  str,
    scheduled_time: str,
    days_of_week:   str = "everyday",
) -> dict:
    """
    Add a dosage schedule for a medication.
    Only medications already in the database can be scheduled.
    """
    # Verify medicine exists
    if not get_medication_by_name(medicine_name):
        return {
            "success": False,
            "message": (
                f"'{medicine_name}' not found. "
                "Please add the medication first."
            )
        }

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO schedules (medicine_name, scheduled_time, days_of_week)
        VALUES (?, ?, ?)
    """, (medicine_name, scheduled_time, days_of_week))
    conn.commit()
    conn.close()
    print(f"[database] Schedule added: {medicine_name} at {scheduled_time}")
    return {
        "success": True,
        "message": f"Schedule added: {medicine_name} at {scheduled_time} ({days_of_week})"
    }


def get_schedules() -> list[dict]:
    """Retrieve all medication schedules ordered by time."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedules ORDER BY scheduled_time")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ──────────────────────────────────────────────────────────────
# Adherence logging
# ──────────────────────────────────────────────────────────────

def log_adherence(
    medicine_name:  str,
    scheduled_time: str,
    taken:          bool,
) -> None:
    """Record whether a patient took their medication."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO adherence_logs (medicine_name, scheduled_time, taken, logged_at)
        VALUES (?, ?, ?, ?)
    """, (
        medicine_name,
        scheduled_time,
        1 if taken else 0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()
    status = "TAKEN" if taken else "MISSED"
    print(f"[database] Adherence: {medicine_name} at {scheduled_time} — {status}")


def get_adherence_logs(medicine_name: str = None) -> list[dict]:
    """Retrieve adherence logs, optionally filtered by medicine name."""
    conn   = get_connection()
    cursor = conn.cursor()
    if medicine_name:
        cursor.execute(
            "SELECT * FROM adherence_logs WHERE medicine_name = ? "
            "ORDER BY logged_at DESC",
            (medicine_name,)
        )
    else:
        cursor.execute("SELECT * FROM adherence_logs ORDER BY logged_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Database Module — Self Test (3-Slot System)")
    print("=" * 55 + "\n")

    create_tables()

    # Add 3 medications to slots
    print(add_medication("Paracetamol", "500mg",  "twice daily",      1, "take after food"))
    print(add_medication("Amoxicillin", "250mg",  "three times daily", 2, "complete full course"))
    print(add_medication("Metformin",   "500mg",  "twice daily",      3, "take with meals"))

    # Try adding a 4th — should be blocked
    print("\n--- Try adding 4th medication (should fail) ---")
    print(add_medication("Amlodipine", "5mg", "once daily", 1))

    # Show available slots
    print(f"\nAvailable slots: {get_available_slots()}")

    # Show all medications
    print("\n--- All Medications ---")
    for m in get_medications():
        print(f"  Slot {m['slot']} | {m['medicine_name']:15} | {m['dosage']} | {m['frequency']}")

    # Remove one and try again
    print("\n--- Remove Metformin ---")
    print(remove_medication("Metformin"))
    print(f"Available slots after removal: {get_available_slots()}")

    print("\n✅ Database test complete!")
    print("=" * 55)