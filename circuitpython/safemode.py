# Runs when CircuitPython enters safe mode (CircuitPython 9+). Safe mode
# skips code.py until someone resets the board, so a power dip from the
# target device (e.g. an Android phone going to sleep) would leave the
# bridge dead. Reboot instead after a brownout.
import microcontroller
import supervisor

if supervisor.runtime.safe_mode_reason == supervisor.SafeModeReason.BROWNOUT:
    microcontroller.reset()
