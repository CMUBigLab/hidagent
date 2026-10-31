# hidagent
HIDAgent: UI Agent control of any HID-Compatible Device

## Hardware Setup: Raspberry Pi Pico

The HIDAgent hardware bridge uses a Raspberry Pi Pico (RP2040) running
CircuitPython to act as a USB HID keyboard/mouse for the target device,
controlled by serial commands from your computer.

### What you need

- **Raspberry Pi Pico** (or any RP2040-based board)
- **USB cable** (Pico → target device, e.g., iPhone via Lightning/USB-C adapter)
- **USB-to-serial adapter** (FTDI, CP2102, CH340, etc.) — connects Pico's UART
  to your control computer
- **3 jumper wires** for the UART connection
- **HDMI capture card** for screen capture from the target device

### Step 1: Install CircuitPython on the Pico

1. Download the CircuitPython `.uf2` for your board from
   https://circuitpython.org/board/raspberry_pi_pico/
2. Hold the **BOOTSEL** button on the Pico, plug it into your Mac via USB
3. The Pico mounts as a USB drive named `RPI-RP2`
4. Drag the `.uf2` file onto the drive — the Pico reboots and remounts as
   `CIRCUITPY`

### Step 2: Install required CircuitPython libraries

1. Download the CircuitPython library bundle from
   https://circuitpython.org/libraries (match your CircuitPython version)
2. Copy these from the bundle's `lib/` folder to `CIRCUITPY/lib/`:
   - `adafruit_hid/` (folder)

### Step 3: Copy the HIDAgent code

Copy [circuitpython/code.py](circuitpython/code.py) to `CIRCUITPY/code.py`.
The Pico will auto-reload and start running it. The Pico is now ready to
act as a USB HID device.

Also copy [circuitpython/safemode.py](circuitpython/safemode.py) to
`CIRCUITPY/safemode.py`. Some phones (e.g. Android) cut or dip USB power
when they sleep; this reboots the Pico after a brownout instead of leaving
it in safe mode. The bridge itself waits for the phone to accept it and
replies `Error` to commands sent while the phone is asleep.

### Step 4: Wire the UART

Connect a USB-to-serial adapter to the Pico:

| Adapter | Pico (RP2040) |
|---------|---------------|
| TX      | GP1 (UART0 RX) |
| RX      | GP0 (UART0 TX) |
| GND     | GND           |
| 5V (optional) | VSYS (pin 39) |

The optional 5V wire powers the Pico from the adapter, so it keeps running
when a phone cuts USB power while asleep (common on Android). Use VSYS, not
VBUS (pin 40, which would feed power back into the phone) or 3V3. Connect
it with everything unplugged; connecting wires live can glitch the board.

The serial adapter should plug into your control computer's USB port and
appear as `/dev/tty.usbserial-*` (macOS/Linux) or a `COM` port (Windows).

### Step 5: Connect to the target device

1. Plug the Pico's USB-A side into the target device (e.g., iPhone via
   Lightning-to-USB or USB-C adapter)
2. The target device should recognize the Pico as a keyboard + mouse
3. On iPhone, a small gray cursor circle will appear on screen
4. On Mac/Windows, a regular mouse cursor appears

### Step 6: Connect the HDMI capture card

Plug the HDMI capture card into your control computer (USB) and into the
target device (HDMI). The capture card should appear as a webcam in macOS
camera settings.

### No calibration needed

`move_mouse`, `click_mouse`, `drag_down` and `drag_up` take coordinates in
`get_screenshot()` pixels. HIDAgent finds the pointer in the HDMI capture
and steers it onto the target (within about a pixel), so there's no
calibration step, and a new device works right away. Where the pointer
lands is remembered in `hidagent/cursor_map.json` (per screen size) so
later moves start close; deleting it only makes the first moves a little
slower. The screen's position in the capture frame is saved in
`hidagent/screen_rect.json`.

Moving the Pico to another device is detected automatically: the bridge
counts USB connections and picks a new ID each time it restarts, and
HIDAgent checks these before screenshots and pointer moves. On a change it
re-detects the screen's position and loads what it learned for that screen
size. Re-detection works best when the screen shows something bright out to
its edges (e.g. the home screen). After rotating a device, call
`get_screen_rect(refresh=True)` yourself. Wiring the Pico's power from the
serial adapter (5V to VSYS) keeps it running between devices; without it the
Pico restarts on each plug-in, which is detected the same way.

On iPhone, iOS hides the pointer in a few places (e.g. the AssistiveTouch
menu). `click_mouse` then clicks from what it has learned and marks the
result `blind`.

`calibrate()` (a fixed HID-to-pixel ratio) is still available for setups
without a video capture, but the pointer functions don't use it.

## Software Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables in `.env`:

```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Examples

See [examples/](examples/) for prototype applications:

- `claude_agent.py` — UI agent using Claude Computer Use
- `local_agent.py` — Local UI agent using Gemma + Omniparser
- `data_collector.py` — Automated UI dataset collection
- `screen_reader.py` — Web-based accessible interface for any device
- `helpful_observer.py` — Contextual help based on screen content
- `cross_device_interaction.py` — Tasks spanning multiple devices
- `social_media_scroller.py` — Automated social media feed recording

## Simulator Mode (No Hardware Required)

You can develop and test HIDAgent examples using the macOS Simulator
instead of physical hardware. See [hidagent/simulator.py](hidagent/simulator.py)
and [examples/sim_demo.py](examples/sim_demo.py).
