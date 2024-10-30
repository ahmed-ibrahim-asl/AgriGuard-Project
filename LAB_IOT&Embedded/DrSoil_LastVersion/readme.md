# Dr Soil: Soil Nutrient Monitoring with ESP8266, RS485 Sensors, and Firebase

This is the last stable version of our soil monitoring project, **Dr Soil**, completed before we began our graduation project in the summer. Using an ESP8266, this system collects and logs essential soil data (Nitrogen, Phosphorous, Potassium, pH, soil humidity, temperature, and conductivity) to Firebase for remote monitoring.

## Features
- **Wi-Fi Configuration Page**: ESP8266 hosts an HTML page for easy Wi-Fi setup.
- **Real-time Firebase Logging**: Logs soil data (NPK values, pH, humidity, temperature, and conductivity).
- **Debugging Mode**: Optional Serial Monitor output for diagnostics.

## Hardware Requirements
1. **ESP8266** – Main microcontroller for data collection and Wi-Fi connection.
2. **RS485 Soil Sensors** – Measures soil nutrients and environmental data.
3. **RS485 to UART Module** – Connects RS485 sensors to ESP8266.
4. **Firebase Account** – For cloud data storage.

## Wiring
- **ESP8266 Connections**:
  - `DE_PIN` → GPIO0, `RE_PIN` → GPIO2
  - **RX** → D5, **TX** → D2
- **RS485 Module**:
  - Connect sensor **A** and **B** lines to RS485 A and B.
  - Power the RS485 module with **3.3V** and **GND** from ESP8266.

## Setup and Usage
1. **Wi-Fi Setup**:
   - The ESP8266 creates an access point (`Dr_Soil`).
   - Connect and configure Wi-Fi via `192.168.4.1` in a browser.
2. **Firebase Setup**:
   - Replace `API_KEY` and `DATABASE_URL` in the code with Firebase credentials.
3. **Monitor Data**:
   - View real-time data in Firebase Realtime Database.
4. **Debugging**:
   - Enable `DEBUG_MODE` for Serial Monitor output (baud rate: 115200).

## Code Summary
- **Wi-Fi Setup**: HTML form allows SSID and password input for Wi-Fi.
- **Firebase Integration**: Uses `Firebase_ESP_Client` to upload sensor data.
- **Sensor Data Collection**:
  - RS485 commands in `PROGMEM` are sent to sensors to gather readings.
  - `readSensorData(SensorType sensor)`: Reads specific sensor values.
- **Data Logging**:
  - `sendSensorData()`: Sends data to Firebase at regular intervals.
- **Debugging**:
  - Serial output for diagnostics when enabled.

This version of **Dr Soil** was our final stable build, extensively tested for consistent data logging before moving on to the next phase of our graduation project.
