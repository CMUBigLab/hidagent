import time
import board
import usb_hid
import busio
import json
import os
import supervisor

from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.keycode import Keycode
from adafruit_hid.mouse import Mouse

# UART connection to control computer (via USB-to-serial adapter).
# timeout=0 makes uart.read() return whatever has arrived instead of
# blocking up to 1 s (the default) waiting for a full 32-byte read, and the
# larger buffer holds a few queued commands without overflowing.
uart = busio.UART(board.GP0, board.GP1, baudrate=9600,
                  timeout=0, receiver_buffer_size=512)

buffer = bytearray()

# Reported by the "status" command so the control computer can tell when the
# target device may have changed: boot_id changes when the Pico restarts, and
# host_count goes up each time a USB host (phone, computer) connects.
boot_id = os.urandom(4).hex()
host_count = 0
host_connected = False

def poll_host():
    global host_count, host_connected
    connected = supervisor.runtime.usb_connected
    if connected and not host_connected:
        host_count += 1
    host_connected = connected

def read_uart_line():
    global buffer
    while True:
        poll_host()
        data = uart.read(32)
        if data:
            buffer.extend(data)
            if b'\n' in buffer:
                line, _, remainder = buffer.partition(b'\n')
                buffer = bytearray(remainder)
                # Line noise (e.g. wires moved while powered) can arrive
                # as invalid UTF-8; treat it as a bad command, not a crash.
                try:
                    return line.decode("utf-8").strip()
                except UnicodeError:
                    return ""
        time.sleep(0.01)

def try_parse_json(line):
    try:
        return json.loads(line)
    except Exception as e:
        print("JSON parse error:", e)
        return None

keyboard = keyboard_layout = mouse = None

def init_hid():
    # Keyboard()/Mouse() raise OSError until the host has enumerated the
    # Pico, e.g. while an Android phone is asleep or locked. Return False
    # instead of crashing so the UART loop keeps running.
    # With no host at all, Keyboard()/Mouse() wait for one forever by
    # default, which would stop the bridge answering, so check first and
    # pass a timeout.
    global keyboard, keyboard_layout, mouse
    if not supervisor.runtime.usb_connected:
        mouse = None
        return False
    try:
        keyboard = Keyboard(usb_hid.devices, timeout=1)
        keyboard_layout = KeyboardLayoutUS(keyboard)
        mouse = Mouse(usb_hid.devices, timeout=1)
        return True
    except OSError as e:
        print("HID not ready:", e)
        mouse = None
        return False

# Wait for the host to accept the HID devices, but not forever: commands
# retry init_hid() themselves.
for _ in range(10):
    time.sleep(1)
    if init_hid():
        break

print("HIDAgent ready")

def handle(command_json):
    type_param = command_json.get("type", "")
    x_param = command_json.get("x", 0)
    y_param = command_json.get("y", 0)
    text_param = command_json.get("text", "")

    def move_in_chunks(dx, dy, chunk=10, delay=0.01):
        # Move the mouse by (dx, dy) in small steps to avoid iOS
        # pointer acceleration kicking in on large single moves.
        rem_x, rem_y = dx, dy
        while rem_x != 0 or rem_y != 0:
            step_x = max(-chunk, min(chunk, rem_x))
            step_y = max(-chunk, min(chunk, rem_y))
            mouse.move(x=step_x, y=step_y)
            rem_x -= step_x
            rem_y -= step_y
            time.sleep(delay)

    if type_param == "click":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        move_in_chunks(x_param, y_param)
        time.sleep(0.2)
        mouse.click(Mouse.LEFT_BUTTON)

    elif type_param == "move":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        move_in_chunks(x_param, y_param)

    elif type_param == "nudge":
        # Relative move from the current position, no corner reset. Small,
        # slow steps keep iOS pointer acceleration out of it, so this is for
        # short visual-feedback corrections rather than long moves. "step"
        # (units per report) and "delay" trade speed for linearity.
        move_in_chunks(x_param, y_param, chunk=command_json.get("step", 1),
                       delay=command_json.get("delay", 0.02))

    elif type_param == "nudgeclick":
        move_in_chunks(x_param, y_param, chunk=command_json.get("step", 1),
                       delay=command_json.get("delay", 0.02))
        time.sleep(0.2)
        mouse.click(Mouse.LEFT_BUTTON)

    elif type_param == "type":
        keyboard_layout.write(text_param)

    elif type_param == "key":
        # Handle key names like "enter", "tab", "space"
        key_map = {
            "enter": Keycode.ENTER,
            "return": Keycode.ENTER,
            "tab": Keycode.TAB,
            "escape": Keycode.ESCAPE,
            "space": Keycode.SPACEBAR,
            "delete": Keycode.DELETE,
            "backspace": Keycode.BACKSPACE,
            "up": Keycode.UP_ARROW,
            "down": Keycode.DOWN_ARROW,
            "left": Keycode.LEFT_ARROW,
            "right": Keycode.RIGHT_ARROW,
        }
        key_code = key_map.get(text_param.lower(), None)
        if key_code:
            keyboard.press(key_code)
            keyboard.release_all()

    elif type_param == "run":
        # use command keyboard shortcut to run an app, works on windows/mac/ios/android
        keyboard.press(Keycode.LEFT_GUI, Keycode.SPACEBAR)
        keyboard.release_all()
        time.sleep(1)
        keyboard_layout.write(text_param)
        keyboard.press(Keycode.ENTER)
        keyboard.release_all()

    elif type_param == "downarrow":
        keyboard.press(Keycode.DOWN_ARROW)
        keyboard.release_all()

    elif type_param == "dragdown" or type_param == "dragup":
        step = 25 if type_param == "dragdown" else -25
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        move_in_chunks(x_param, y_param)
        time.sleep(0.1)
        mouse.press(Mouse.LEFT_BUTTON)
        for _ in range(5):
            time.sleep(0.05)
            mouse.move(y=step)
        mouse.release(Mouse.LEFT_BUTTON)


while True:
    line = read_uart_line()
    print("Received:", line)

    command_json = try_parse_json(line)
    if command_json is None:
        uart.write("Error\n".encode("utf-8"))
        continue

    # Doesn't need HID, so it answers even with no host attached.
    if isinstance(command_json, dict) and command_json.get("type") == "status":
        poll_host()
        uart.write("Status {} {} {}\n".format(
            boot_id, host_count, int(host_connected)).encode("utf-8"))
        continue

    # A sleeping or unplugged host makes HID sends raise OSError; reply
    # Error and set the devices up again on the next command.
    if mouse is None and not init_hid():
        uart.write("Error\n".encode("utf-8"))
        continue
    try:
        handle(command_json)
    except OSError as e:
        print("HID error:", e)
        mouse = None
        uart.write("Error\n".encode("utf-8"))
        continue
    except Exception as e:
        # A malformed command (e.g. text the layout can't type) must not
        # stop the bridge; it would go silent until power-cycled.
        print("Command error:", e)
        uart.write("Error\n".encode("utf-8"))
        continue

    uart.write("Success\n".encode("utf-8"))
