"""
hardware_controller.py
----------------------
Arduino Hardware Controller — 28BYJ-48 Stepper Motors via ULN2003.

Hardware wiring:
  Slot 1 — 28BYJ-48 motor #1 via ULN2003 board
            IN1 → pin 4 | IN2 → pin 5 | IN3 → pin 6 | IN4 → pin 7

  Slot 2 — 28BYJ-48 motor #2 via ULN2003 board
            IN1 → pin 8 | IN2 → pin 9 | IN3 → pin 10 | IN4 → pin 11

  Slot 3 — Not wired (simulation only)

Serial protocol:
  Python sends : DISPENSE_SLOT1 / DISPENSE_SLOT2 / DISPENSE_SLOT3\n
  Arduino sends: OK_SLOT1 / OK_SLOT2 / OK_SLOT3 when motor finishes
  Arduino sends: ARDUINO_READY once on boot

Key fixes in this version:
  - Serial port opened once, reset wait applied once, THEN command sent.
    Previously the port was opened inside a sub-function causing a second
    reset which lost the command entirely.
  - MOTOR_TIMEOUT_S applied to readline() correctly — the Arduino does
    not reply until the stepper finishes, which takes several seconds.
  - Return dict always has both "pin" (int) and "pins" (list) so the
    frontend and main.py both work without KeyError.
  - Scheduler dispense call uncommented in scheduler.py (see that file).

Dependencies:
  pip install pyserial
"""

import time

# ──────────────────────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────────────────────

# Change SERIAL_PORT to match your system:
#   Windows  → "COM3"  (check Device Manager → Ports (COM & LPT))
#   Linux    → "/dev/ttyUSB0"  or  "/dev/ttyACM0"
#   macOS    → "/dev/cu.usbmodem14101"
SERIAL_PORT = "COM5"

BAUD_RATE = 9600

# How long (seconds) to wait for the Arduino to reply after sending
# the dispense command.  The 28BYJ-48 at 1200 µs/step × 512 steps
# forward + 512 back ≈ 1.2 s of motor time.  We give 15 s of margin.
# If you increase DISPENSE_STEPS in the Arduino sketch, raise this too.
MOTOR_TIMEOUT_S = 15

# Seconds to wait after the serial port opens for the Arduino Uno
# to finish its reset cycle.  Do not reduce below 2.
ARDUINO_RESET_WAIT = 2

# Slot → motor wiring
SLOT_CONFIG = {
    1: {
        "pins":        [4, 5, 6, 7],
        "description": "28BYJ-48 Motor 1 via ULN2003 (pins 4-7)",
        "wired":       True,
    },
    2: {
        "pins":        [8, 9, 10, 11],
        "description": "28BYJ-48 Motor 2 via ULN2003 (pins 8-11)",
        "wired":       True,
    },
    3: {
        "pins":        [],
        "description": "Slot 3 — not wired (simulation only)",
        "wired":       False,
    },
}

MAX_SLOTS = 3


# ──────────────────────────────────────────────────────────────
# 2.  Import pyserial
# ──────────────────────────────────────────────────────────────

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
    print("[hardware] pyserial loaded successfully.")
except ImportError:
    SERIAL_AVAILABLE = False
    print("[hardware] WARNING: pyserial not installed — running in SIMULATION MODE.")
    print("[hardware] Install with: pip install pyserial")


# ──────────────────────────────────────────────────────────────
# 3.  List available serial ports
# ──────────────────────────────────────────────────────────────

def list_available_ports() -> list[str]:
    """Return a list of all available serial port device names."""
    if not SERIAL_AVAILABLE:
        print("[hardware] Cannot list ports — pyserial not installed.")
        return []
    ports      = serial.tools.list_ports.comports()
    port_names = [p.device for p in ports]
    print(f"[hardware] Available ports: {port_names}")
    return port_names


# ──────────────────────────────────────────────────────────────
# 4.  Core dispense by slot
# ──────────────────────────────────────────────────────────────

def dispense_slot(slot: int) -> dict:
    """
    Send a dispense command for the given slot and wait for the motor
    to finish before returning.

    Correct serial flow (fixes the silent-failure bug):
      1. serial.Serial() opens the port  →  Arduino resets
      2. time.sleep(ARDUINO_RESET_WAIT)  →  wait for boot to finish
      3. reset_input_buffer()            →  discard bootloader noise
      4. write(command)                  →  Arduino receives the command
      5. readline(timeout=MOTOR_TIMEOUT_S) →  wait for "OK_SLOTx"
      6. close()

    Parameters
    ----------
    slot : int   1, 2, or 3

    Returns
    -------
    dict with keys:
        success  : bool
        slot     : int
        pin      : int   — first IN pin (frontend compatibility)
        pins     : list  — all four IN pins
        command  : str
        response : str
        mode     : "hardware" | "simulation" | "error"
    """
    if slot not in SLOT_CONFIG:
        return _err(slot, [], None,
                    f"Invalid slot {slot}. Must be 1, 2, or 3.")

    cfg     = SLOT_CONFIG[slot]
    pins    = cfg["pins"]
    wired   = cfg["wired"]
    command = f"DISPENSE_SLOT{slot}"

    print(f"\n[hardware] {'='*50}")
    print(f"[hardware]  DISPENSE SLOT {slot}")
    print(f"[hardware]  {cfg['description']}")
    print(f"[hardware] {'='*50}")

    # ── Slot 3: not wired ─────────────────────────────────────
    if not wired:
        print("[hardware] Slot 3 not wired — simulating.")
        return _ok(slot, pins, command,
                   f"SIMULATION: Slot {slot} — no motor wired.", "simulation")

    # ── Slots 1 & 2: real hardware ───────────────────────────
    if SERIAL_AVAILABLE:
        arduino = None
        try:
            # Step 1: open port (this triggers Arduino reset)
            print(f"[hardware] Opening {SERIAL_PORT}…")
            arduino = serial.Serial(
                port     = SERIAL_PORT,
                baudrate = BAUD_RATE,
                timeout  = 1,          # short timeout for now
            )

            # Step 2: wait for Arduino to finish booting
            print(f"[hardware] Waiting {ARDUINO_RESET_WAIT}s for Arduino boot…")
            time.sleep(ARDUINO_RESET_WAIT)

            # Step 3: discard anything in the buffer (e.g. "ARDUINO_READY\n")
            arduino.reset_input_buffer()

            # Step 4: send the command
            print(f"[hardware] Sending: {command}")
            arduino.write((command + "\n").encode("utf-8"))
            arduino.flush()

            # Step 5: now extend timeout to wait for motor to finish
            arduino.timeout = MOTOR_TIMEOUT_S
            print(f"[hardware] Waiting up to {MOTOR_TIMEOUT_S}s for motor…")

            raw      = arduino.readline()
            response = raw.decode("utf-8", errors="ignore").strip()

            if not response:
                msg = (
                    f"No response within {MOTOR_TIMEOUT_S}s. "
                    "Check: correct port? Arduino sketch uploaded? Wiring OK?"
                )
                print(f"[hardware] ⚠️  Timeout — {msg}")
                return _err(slot, pins, command, msg)

            success = response.startswith("OK_")
            print(f"[hardware] Response : {response}")
            print(f"[hardware] Result   : {'✅ Success' if success else '❌ Failed'}")

            if success:
                return _ok(slot, pins, command, response, "hardware")
            else:
                return _err(slot, pins, command, response)

        except serial.SerialException as e:
            msg = f"Cannot open {SERIAL_PORT}: {e}. Ports available: {list_available_ports()}"
            print(f"[hardware] ❌ {msg}")
            return _err(slot, pins, command, msg)

        except Exception as e:
            print(f"[hardware] ❌ Unexpected error: {e}")
            return _err(slot, pins, command, str(e))

        finally:
            try:
                if arduino and arduino.is_open:
                    arduino.close()
                    print(f"[hardware] Port closed.")
            except Exception:
                pass

    # ── Fallback: no pyserial ─────────────────────────────────
    print(f"[hardware] SIMULATION (no pyserial): Slot {slot} pins {pins}.")
    return _ok(slot, pins, command,
               f"SIMULATION: Slot {slot} dispensed (no Arduino connected).",
               "simulation")


# ──────────────────────────────────────────────────────────────
# 5.  Dispense by medicine name
# ──────────────────────────────────────────────────────────────

def dispense(medicine_name: str) -> dict:
    """
    Look up a medicine's assigned slot in the DB and call dispense_slot().

    Parameters
    ----------
    medicine_name : str

    Returns
    -------
    dict — same as dispense_slot(), plus key 'medicine_name'.
    """
    from database import get_medication_by_name

    med = get_medication_by_name(medicine_name)
    if not med:
        print(f"[hardware] '{medicine_name}' not found in DB.")
        return {
            "success": False, "slot": None, "pin": None, "pins": [],
            "command": None,
            "response": f"Medicine '{medicine_name}' not found in database.",
            "mode": "error", "medicine_name": medicine_name,
        }

    slot = med.get("slot")
    if not slot:
        print(f"[hardware] No slot assigned to '{medicine_name}'.")
        return {
            "success": False, "slot": None, "pin": None, "pins": [],
            "command": None,
            "response": f"No slot assigned to '{medicine_name}'. Add it via the Medications page first.",
            "mode": "error", "medicine_name": medicine_name,
        }

    result = dispense_slot(int(slot))
    result["medicine_name"] = medicine_name
    return result


# ──────────────────────────────────────────────────────────────
# 6.  Internal dict builders
# ──────────────────────────────────────────────────────────────

def _ok(slot, pins, command, response, mode) -> dict:
    return {
        "success":  True,
        "slot":     slot,
        "pin":      pins[0] if pins else None,  # first pin for UI
        "pins":     pins,
        "command":  command,
        "response": response,
        "mode":     mode,
    }

def _err(slot, pins, command, response) -> dict:
    return {
        "success":  False,
        "slot":     slot,
        "pin":      pins[0] if pins else None,
        "pins":     pins,
        "command":  command,
        "response": response,
        "mode":     "error",
    }


# ──────────────────────────────────────────────────────────────
# 7.  Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Hardware Controller — Self Test")
    print("  28BYJ-48 + ULN2003 (Slots 1 & 2)")
    print("=" * 55 + "\n")

    print("Wiring summary:")
    for s, cfg in SLOT_CONFIG.items():
        w = f"pins {cfg['pins']}" if cfg["wired"] else "NOT WIRED"
        print(f"  Slot {s}: {cfg['description']}  [{w}]")

    print(f"\nSerial port configured : {SERIAL_PORT}")
    print("Detecting ports        :", list_available_ports())
    print()

    for slot in [1, 2, 3]:
        print(f"--- Testing Slot {slot} ---")
        result = dispense_slot(slot)
        mark = "✅" if result["success"] else "❌"
        print(f"  {mark} [{result['mode']}]  {result['response'][:70]}")
        print()
        time.sleep(1)

    print("Self-test complete.")
    print("=" * 55)