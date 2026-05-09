"""
scheduler.py
------------
Medication Reminder Scheduler using APScheduler.

What this file does:
  1. Reads all medication schedules from the SQLite database
  2. Every minute, checks if any medication is due right now
  3. If a medication is due:
       - Prints a reminder to the terminal
       - Logs it in the adherence_logs table
       - (Later) triggers the hardware dispenser

How it works:
  - Uses APScheduler's BackgroundScheduler to run checks in the background
  - The check runs every 60 seconds automatically
  - Time matching is done by comparing HH:MM of current time with scheduled times

Usage:
  - Run standalone : python scheduler.py
  - Import into main.py to run alongside the FastAPI server

Dependencies:
  pip install apscheduler
"""

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import time

# Import our own modules
from database import get_schedules, log_adherence, create_tables


# ──────────────────────────────────────────────────────────────
# 1.  Core check function — runs every minute
# ──────────────────────────────────────────────────────────────

def check_medication_schedules() -> None:
    """
    Check if any medication is due at the current time.

    This function is called automatically every minute by the scheduler.

    Steps:
      1. Get the current time in HH:MM format  e.g. "08:00"
      2. Load all schedules from the database
      3. For each schedule, check if scheduled_time matches current time
      4. If it matches:
           - Print a reminder message to the terminal
           - Log it in the adherence_logs table
           - Call the hardware dispenser (when hardware is connected)

    Example output when a medication is due:
      ============================================================
      💊 MEDICATION REMINDER
      ============================================================
        Medicine  : Paracetamol
        Dose Time : 08:00
        Now       : 2026-03-12 08:00
        Action    : Please take your Paracetamol (500mg) now.
      ============================================================
    """
    # Get current time as HH:MM string  e.g. "08:00"
    now = datetime.now()
    current_time_str = now.strftime("%H:%M")

    print(f"[scheduler] Checking schedules at {current_time_str} …")

    # Load all schedules from the database
    schedules = get_schedules()

    if not schedules:
        print("[scheduler] No schedules found in database.")
        return

    # Check each schedule
    for schedule in schedules:
        medicine_name  = schedule["medicine_name"]
        scheduled_time = schedule["scheduled_time"]
        days_of_week   = schedule["days_of_week"]

        # Check if this schedule applies today
        if not _is_scheduled_today(days_of_week, now):
            continue

        # Check if the current time matches the scheduled time
        if current_time_str == scheduled_time:

            # ── Print reminder to terminal ──────────────────
            print("\n" + "=" * 60)
            print("  💊 MEDICATION REMINDER")
            print("=" * 60)
            print(f"  Medicine  : {medicine_name}")
            print(f"  Dose Time : {scheduled_time}")
            print(f"  Now       : {now.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Action    : Please take your {medicine_name} now.")
            print("=" * 60 + "\n")

            # ── Log in adherence table (marked as dispensed) ─
            log_adherence(
                medicine_name  = medicine_name,
                scheduled_time = scheduled_time,
                taken          = True,
            )

            # ── Trigger hardware dispenser ───────────────────
            try:
                from hardware_controller import dispense
                print(f"[scheduler] Triggering dispenser for {medicine_name}…")
                hw_result = dispense(medicine_name)
                if hw_result["success"]:
                    print(f"[scheduler] ✅ Dispenser OK — {hw_result['response']}")
                else:
                    print(f"[scheduler] ⚠️  Dispenser issue — {hw_result['response']}")
            except Exception as e:
                print(f"[scheduler] ❌ Hardware error: {e}")


# ──────────────────────────────────────────────────────────────
# 2.  Helper — check if schedule applies today
# ──────────────────────────────────────────────────────────────

def _is_scheduled_today(days_of_week: str, now: datetime) -> bool:
    """
    Return True if the schedule should run today.

    Parameters
    ----------
    days_of_week : str
        Either "everyday"  →  always runs
        Or comma-separated day abbreviations  e.g. "Mon,Wed,Fri"
    now : datetime
        Current datetime object.

    Examples
    --------
    >>> _is_scheduled_today("everyday", datetime.now())
    True
    >>> _is_scheduled_today("Mon,Wed,Fri", datetime.now())   # if today is Tuesday
    False
    """
    if days_of_week.strip().lower() == "everyday":
        return True

    # Get today's 3-letter abbreviation e.g. "Mon", "Tue", "Wed"
    today_abbr = now.strftime("%a")   # e.g. "Mon"

    # Split the days string and check if today is in the list
    scheduled_days = [d.strip() for d in days_of_week.split(",")]
    return today_abbr in scheduled_days


# ──────────────────────────────────────────────────────────────
# 3.  Scheduler setup and start
# ──────────────────────────────────────────────────────────────

def start_scheduler() -> BackgroundScheduler:
    """
    Create, configure, and start the background scheduler.

    The scheduler runs check_medication_schedules() every 60 seconds
    in a background thread — so it does not block the FastAPI server.

    Returns
    -------
    BackgroundScheduler
        The running scheduler instance.
        Call scheduler.shutdown() to stop it cleanly.

    Example
    -------
    >>> scheduler = start_scheduler()
    [scheduler] Starting medication reminder scheduler…
    [scheduler] Scheduler started. Checking every 60 seconds.
    """
    print("[scheduler] Starting medication reminder scheduler…")

    scheduler = BackgroundScheduler()

    # Add the job — runs check_medication_schedules every 60 seconds
    scheduler.add_job(
        func            = check_medication_schedules,
        trigger         = "interval",
        seconds         = 60,
        id              = "medication_check",
        name            = "Check Medication Schedules",
        replace_existing= True,
    )

    scheduler.start()
    print("[scheduler] Scheduler started. Checking every 60 seconds.")

    return scheduler


# ──────────────────────────────────────────────────────────────
# 4.  Quick self-test (run: python scheduler.py)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Scheduler Module — Self Test")
    print("=" * 55 + "\n")

    # Make sure tables exist before we try to read schedules
    create_tables()

    # ── Add a test schedule for RIGHT NOW so we can see it fire ──
    # We'll add a schedule for the current minute so the reminder
    # triggers immediately on the next check.
    from database import add_medication, add_schedule, get_schedules

    current_minute = datetime.now().strftime("%H:%M")
    print(f"[test] Current time is {current_minute}")
    print(f"[test] Adding a test schedule for Paracetamol at {current_minute}")

    # Add medication if it doesn't exist yet
    add_medication("Paracetamol", "500mg", "twice daily", "take after food")

    # Add schedule for right now
    add_schedule("Paracetamol", current_minute, "everyday")

    # Show all schedules
    print("\n--- Current Schedules in Database ---")
    for s in get_schedules():
        print(f"  {s['medicine_name']:15} | {s['scheduled_time']} | {s['days_of_week']}")

    # ── Run one immediate check ───────────────────────────────
    print("\n[test] Running immediate schedule check…")
    check_medication_schedules()

    # ── Start background scheduler and keep running ───────────
    print("\n[test] Starting background scheduler…")
    print("[test] You will see a check message every 60 seconds.")
    print("[test] Press Ctrl+C to stop.\n")

    scheduler = start_scheduler()

    try:
        # Keep the script alive so we can see the scheduler fire
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[scheduler] Stopping scheduler…")
        scheduler.shutdown()
        print("[scheduler] Scheduler stopped. Goodbye!")

    print("\n" + "=" * 55)