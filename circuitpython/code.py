import time


import board
import digitalio
import usb_hid


from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.keycode import Keycode
from adafruit_hid.mouse import Mouse

from os import getenv
import ipaddress
import wifi
import socketpool

import busio
import json


from adafruit_connection_manager import get_radio_socketpool
from adafruit_httpserver import Request, Response, Server


uart = busio.UART(board.GP0, board.GP1, baudrate=9600)

buffer = bytearray()

# read a line from connected UART
def read_uart_line():
    global buffer
    while True:
        data = uart.read(32)  # Read up to 32 bytes at a time
        if data:
            buffer.extend(data)
            if b'\n' in buffer:
                line, _, remainder = buffer.partition(b'\n')
                buffer = bytearray(remainder)  # Save remaining bytes
                return line.decode("utf-8").strip()
        time.sleep(0.01)

# parse line of JSON
def try_parse_json(line):
    try:
        return json.loads(line)
    except Exception as e:
        print("JSON parse error:", e)
        return None



# The keyboard object!
time.sleep(1)  # Sleep for a bit to avoid a race condition on some systems
keyboard = Keyboard(usb_hid.devices)
keyboard_layout = KeyboardLayoutUS(keyboard)  # We're in the US :)
mouse = Mouse(usb_hid.devices)
time.sleep(1)

while True:
    # data = uart.read(32)  # read up to 32 bytes
    # print(data)  # this is a bytearray type

    line = read_uart_line()
    print("Received line:", line)

    command_json = try_parse_json(line)

    print(command_json["type"])


    command_key = Keycode.LEFT_GUI
    spacebar_key = Keycode.SPACEBAR
    enter_key = Keycode.ENTER
    downarrow_key = Keycode.DOWN_ARROW

    type_param = command_json["type"]
    x_param = command_json["x"]
    y_param = command_json["y"]
    text_param = command_json["text"]

    if type_param == "click":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        mouse.move(x=x_param, y=y_param)
        time.sleep(0.2)
        mouse.click(Mouse.LEFT_BUTTON)
    elif type_param == "move":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        mouse.move(x=x_param, y=y_param)
    elif type_param == "type":
        keyboard_layout.write(text_param)
    elif type_param == "run":
        keyboard.press(command_key, spacebar_key)
        #time.sleep(1)
        keyboard.release_all()
        time.sleep(1)
        keyboard_layout.write(text_param)
        #time.sleep(0.1)
        keyboard.press(enter_key)
        keyboard.release_all()
    elif type_param == "downarrow":
        keyboard.press(downarrow_key)
        keyboard.release_all()
    elif type_param == "dragdown":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        mouse.move(x=x_param, y=y_param)
        time.sleep(0.1)
        mouse.press(Mouse.LEFT_BUTTON)
        time.sleep(0.05)
        mouse.move(y=25)
        time.sleep(0.05)
        mouse.move(y=25)
        time.sleep(0.05)
        mouse.move(y=25)
        time.sleep(0.05)
        mouse.move(y=25)
        time.sleep(0.05)
        mouse.move(y=25)
        mouse.release(Mouse.LEFT_BUTTON)



    #if data is not None:
    #    # convert bytearray to string
    #    data_string = ''.join([chr(b) for b in data])
    #    print(data_string, end="")

    uart.write("Success\n".encode("utf-8"))






# Get WiFi details, ensure these are setup in settings.toml
ssid = getenv("CIRCUITPY_WIFI_SSID")
password = getenv("CIRCUITPY_WIFI_PASSWORD")

if None in [ssid, password]:
    raise RuntimeError(
        "WiFi settings are kept in settings.toml, "
        "please add them there. The settings file must contain "
        "'CIRCUITPY_WIFI_SSID', 'CIRCUITPY_WIFI_PASSWORD', "
        "at a minimum."
    )

print()
print("Connecting to WiFi")

#  connect to your SSID
try:
    wifi.radio.connect(ssid, password)
except TypeError:
    print("Could not find WiFi info. Check your settings.toml file!")
    raise

print("Connected to WiFi")

pool = socketpool.SocketPool(wifi.radio)

server = Server(pool, "/static", debug=True)


@server.route("/")
def base(request: Request):
    query_params = request.query_params

    type_param = query_params.get("type", "undefined")
    x_param = int(query_params.get("x", "undefined"))
    y_param = int(query_params.get("y", "undefined"))
    text_param = query_params.get("text", "undefined")

    command_key = Keycode.LEFT_GUI
    spacebar_key = Keycode.SPACEBAR
    enter_key = Keycode.ENTER
    downarrow_key = Keycode.DOWN_ARROW

    if type_param == "click":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        mouse.move(x=x_param, y=y_param)
        time.sleep(0.2)
        mouse.click(Mouse.LEFT_BUTTON)
    elif type_param == "move":
        mouse.move(x=-1000, y=-1000)
        time.sleep(0.1)
        mouse.move(x=x_param, y=y_param)
    elif type_param == "type":
        keyboard_layout.write(text_param)
    elif type_param == "run":
        keyboard.press(command_key, spacebar_key)
        #time.sleep(1)
        keyboard.release_all()
        time.sleep(1)
        keyboard_layout.write(text_param)
        #time.sleep(0.1)
        keyboard.press(enter_key)
        keyboard.release_all()
    elif type_param == "downarrow":
        keyboard.press(downarrow_key)
        keyboard.release_all()



    response_text = f"Received type: {type_param}\n x: {x_param}, y: {y_param}\n text: {text_param}"

    return Response(request, response_text, content_type="text/plain")

server.serve_forever(str(wifi.radio.ipv4_address))


