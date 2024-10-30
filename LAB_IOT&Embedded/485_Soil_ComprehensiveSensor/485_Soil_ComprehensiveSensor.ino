#include <SoftwareSerial.h>
#include <avr/pgmspace.h>

#define RE_PIN 3
#define DE_PIN 4

// Pre-defined commands stored in flash memory
const byte nitro[] PROGMEM = {0x01, 0x03, 0x00, 0x1e, 0x00, 0x01, 0xe4, 0x0c};  // Nitrogen
const byte phos[] PROGMEM = {0x01, 0x03, 0x00, 0x1f, 0x00, 0x01, 0xb5, 0xcc};  // Phosphorous
const byte pota[] PROGMEM = {0x01, 0x03, 0x00, 0x20, 0x00, 0x01, 0x85, 0xc0};  // Potassium
const byte phCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x06, 0x00, 0x01, 0x64, 0x0B};  // pH
const byte soilHumidityCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x12, 0x00, 0x01, 0x92, 0x92};  // Soil Humidity
const byte temperatureCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x13, 0x00, 0x01, 0xFF, 0x9B};  // Temperature
const byte conductivityCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x15, 0x00, 0x01, 0x95, 0xCE};  // Conductivity

enum SensorType {
  NITROGEN,
  PHOSPHOROUS,
  POTASSIUM,
  PH,
  SOIL_HUMIDITY,
  TEMPERATURE,
  CONDUCTIVITY
};

byte values[11];  // Buffer for sensor data
SoftwareSerial mod(2, 5);  // RX connected to pin 2, TX connected to pin 5

void setup() {
  Serial.begin(9600);
  mod.begin(9600);
  pinMode(RE_PIN, OUTPUT);
  pinMode(DE_PIN, OUTPUT);

  Serial.println("NPK Sensor Initializing...");
  delay(3000);
}

void loop() {
  // Read values from different sensors and print them
  float nitrogen = readSensorData(NITROGEN);
  delay(250);
  float phosphorous = readSensorData(PHOSPHOROUS);
  delay(250);
  float potassium = readSensorData(POTASSIUM);
  delay(250);
  float ph = readSensorData(PH) / 10.0;  // Assuming pH is scaled by a factor of 10
  delay(250);
  float soilHumidity = readSensorData(SOIL_HUMIDITY);
  delay(250);
  float temperature = readSensorData(TEMPERATURE);
  delay(250);
  float conductivity = readSensorData(CONDUCTIVITY);
  delay(250);

  // Output sensor readings
  printSensorData(nitrogen, phosphorous, potassium, ph, soilHumidity, temperature, conductivity);
  delay(2000);
}

float readSensorData(SensorType sensor) {
  const byte *command;
  switch(sensor) {
    case NITROGEN: command = nitro; break;
    case PHOSPHOROUS: command = phos; break;
    case POTASSIUM: command = pota; break;
    case PH: command = phCommand; break;
    case SOIL_HUMIDITY: command = soilHumidityCommand; break;
    case TEMPERATURE: command = temperatureCommand; break;
    case CONDUCTIVITY: command = conductivityCommand; break;
  }

  byte buffer[8];
  memcpy_P(buffer, command, 8);

  // Send Modbus request
  digitalWrite(DE_PIN, HIGH);
  digitalWrite(RE_PIN, HIGH);
  delay(100);

  if (mod.write(buffer, 8) == 8) {
    digitalWrite(DE_PIN, LOW);
    digitalWrite(RE_PIN, LOW);
    delay(200);

    // Read response from sensor
    for (byte i = 0; i < 7; i++) {
      if (mod.available()) {
        values[i] = mod.read();
      }
    }

    // Combine the two bytes for the result
    int result = (values[3] << 8) | values[4];

    // Handle scaling for specific sensors
    if (sensor == PH) {
      return result / 10.0;  // pH scaled by 10
    }
    
    if (sensor == SOIL_HUMIDITY && result > 100) {
      return 100.0;  // Cap soil humidity at 100%
    }

    if (sensor == TEMPERATURE && result > 100) {
      return 100.0;  // Cap temperature at 100°C
    }

    Serial.print("Parsed value: ");
    Serial.println(result);

    return result;  // Return the parsed result
  }

  Serial.println("Error: Failed to read data.");
  return -1;  // Return -1 in case of error
}

// Function to print sensor data
void printSensorData(float nitrogen, float phosphorous, float potassium, float ph, float soilHumidity, float temperature, float conductivity) {
  Serial.print("Nitrogen: ");
  Serial.print(nitrogen);
  Serial.println(" mg/kg");

  Serial.print("Phosphorous: ");
  Serial.print(phosphorous);
  Serial.println(" mg/kg");

  Serial.print("Potassium: ");
  Serial.print(potassium);
  Serial.println(" mg/kg");

  Serial.print("pH: ");
  Serial.println(ph);

  Serial.print("Soil Humidity: ");
  Serial.print(soilHumidity);
  Serial.println(" %");

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  Serial.print("Conductivity: ");
  Serial.print(conductivity);
  Serial.println(" µS/cm");
}