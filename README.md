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

### Step 4: Wire the UART

Connect a USB-to-serial adapter to the Pico:

| Adapter | Pico (RP2040) |
|---------|---------------|
| TX      | GP1 (UART0 RX) |
| RX      | GP0 (UART0 TX) |
| GND     | GND           |

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

### Step 7: Calibrate

Run the calibration procedure to learn the mapping between HID coordinates
and pixel positions:

```bash
python test_calibration.py
```

This takes ~5 minutes. The calibration result is saved to
`hidagent/calibration.json` and is automatically loaded on subsequent runs.
If you switch to a different target device, the calibration is keyed on
screen resolution so previous calibrations are reused automatically.

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
