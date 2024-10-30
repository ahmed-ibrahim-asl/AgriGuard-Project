# Soil Monitoring System with Arduino Uno, RS485 Soil Sensor, and Serial Output

This project uses an Arduino Uno connected to an RS485 soil sensor module to read various soil properties and nutrients. The sensor reads values for essential nutrients like Nitrogen, Phosphorous, Potassium, as well as pH, soil humidity, temperature, and conductivity. The collected data is printed to the Serial Monitor for real-time monitoring, making it a valuable tool for precision agriculture.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Library Requirements](#library-requirements)
- [Wiring](#wiring)
- [Usage](#usage)
- [Code Overview](#code-overview)
  - [Main Functions](#main-functions)
  - [Sensor Communication via RS485](#sensor-communication-via-rs485)

---

### Project Overview
This project sets up an Arduino Uno with a connected RS485 soil sensor module to continuously monitor soil properties. The sensor module communicates via RS485 and is connected to the Arduino Uno through a UART interface using the SoftwareSerial library, as the Uno has only one hardware UART.

### Features
- Real-time soil nutrient and environmental monitoring.
- Sensor data displayed via Serial Monitor.

### Hardware Requirements
1. **Arduino Uno** – Microcontroller for processing sensor data.
2. **RS485 Soil Sensor Module** – For measuring nitrogen, phosphorous, potassium, pH, soil humidity, temperature, and conductivity.
3. **RS485 to UART Module** – Connects the RS485 soil sensor to the Arduino Uno via a software-based UART interface.

### Library Requirements
Install the following libraries:
- [SoftwareSerial](https://www.arduino.cc/en/Reference/SoftwareSerial) – For creating a software-based UART interface for RS485 communication.

### Wiring
1. Connect the **RS485 sensor module** to an **RS485-to-UART converter** and then connect it to the Arduino Uno using SoftwareSerial.
2. **RS485 Communication Control Pins**: Connect `RE_PIN` to digital pin 3 and `DE_PIN` to digital pin 4 on the Arduino Uno for direction control during data transmission.
3. **Software Serial Pins**: Connect RX to pin 2 and TX to pin 5 on the Arduino Uno.

### Usage
1. Upload the code to the Arduino Uno.
2. Open the Serial Monitor (baud rate: 9600).
3. The Arduino will initialize the sensor and start displaying real-time readings for each parameter (Nitrogen, Phosphorous, Potassium, pH, soil humidity, temperature, and conductivity).

### Code Overview

#### Main Functions

- `setup()`: Initializes the Arduino, sets up the RS485 communication pins, and starts the SoftwareSerial interface.
- `loop()`: Reads values from the soil sensor module and prints them to the Serial Monitor at regular intervals.
- `readSensorData(SensorType sensor)`: Communicates with the RS485 soil sensor to retrieve readings based on the specified sensor type.

#### Sensor Communication via RS485

This sketch uses `SoftwareSerial` to communicate with the soil sensor module via RS485 commands. Each sensor type has a predefined command stored in flash memory, which is sent to the sensor module for each reading. The soil sensor communicates using Modbus protocol, and commands control the `DE_PIN` and `RE_PIN` to handle read and write modes. Supported sensors include:
- **Nitrogen**
- **Phosphorous**
- **Potassium**
- **pH**
- **Soil Humidity**
- **Temperature**
- **Conductivity**

Each reading command is sent over the RS485 line, and the resulting data is processed by the Arduino, with specific scaling and adjustments applied as needed for parameters like pH, soil humidity, and temperature.
