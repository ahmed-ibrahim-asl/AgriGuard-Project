# Soil Moisture Sensor Test with Arduino Uno

This sketch tests a soil moisture sensor on an Arduino Uno. It reads analog values from `A0`, converts them to a percentage (0–100%), and displays them in the Serial Monitor.

## Purpose
Confirm that the soil moisture sensor is connected and working.

## Requirements
- **Arduino Uno**
- **Soil Moisture Sensor**

## Wiring
- Sensor **Analog Output** → `A0` on Arduino Uno
- Sensor **Power** → `5V` on Arduino Uno
- Sensor **Ground** → `GND` on Arduino Uno

## Usage
1. Upload the sketch.
2. Open Serial Monitor (9600 baud).
3. View soil moisture as a percentage, updating every 500 ms.

## Code Summary
- `setup()`: Initializes Serial Monitor.
- `loop()`: Reads sensor, converts to percentage, prints to Serial Monitor.

