import time
import board
import usb_hid
import busio
import json

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

def read_uart_line():
    global buffer
    while True:
        data = uart.read(32)
        if data:
            buffer.extend(data)
            if b'\n' in buffer:
                line, _, remainder = buffer.partition(b'\n')
                buffer = bytearray(remainder)
                return line.decode("utf-8").strip()
        time.sleep(0.01)

def try_parse_json(line):
    try:
        return json.loads(line)
    except Exception as e:
        print("JSON parse error:", e)
        return None

# Initialize HID devices
time.sleep(1)
keyboard = Keyboard(usb_hid.devices)
keyboard_layout = KeyboardLayoutUS(keyboard)
mouse = Mouse(usb_hid.devices)
time.sleep(1)

print("HIDAgent ready")

while True:
    line = read_uart_line()
    print("Received:", line)

    command_json = try_parse_json(line)
    if command_json is None:
        uart.write("Error\n".encode("utf-8"))
        continue

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

    uart.write("Success\n".encode("utf-8"))
