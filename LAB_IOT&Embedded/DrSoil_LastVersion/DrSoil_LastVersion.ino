#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <SoftwareSerial.h>
#include <avr/pgmspace.h>

#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

enum SensorType {
  NITROGEN,
  PHOSPHOROUS,
  POTASSIUM,
  PH,
  SOIL_HUMIDITY,
  TEMPERATURE,
  CONDUCTIVITY
};


// Function Prototypes
void handleConfigurationPage();
void handleWiFiConfig();
String scanNetworks();
void sendSensorData();
bool checkFirebaseConnection();
float readSensorData(SensorType sensor);
void printSensorData(float nitrogen, float phosphorous, float potassium, float ph, float soilHumidity, float temperature, float conductivity);

// Constants and Definitions
#define DEBUG_MODE                                      true

#if DEBUG_MODE
  #define DEBUG_PRINT(message_to_print) Serial.println(message_to_print)
#else
  #define DEBUG_PRINT(message_to_print)
#endif

#define ESP_BAUDRATE                                    115200
#define WIFI_SCAN_CACHE_DURATION_MS                     30000
#define SEND_DATA_INTERVAL_MS                           60000  // Data sending interval

#define CONFIG_NETWORK_SSID                             "Dr_Soil"
#define CONFIG_NETWORK_PASSWORD                         "12345678"

#define API_KEY                                         "AIzaSyDNg8jWhN7XnI7l8Oq0_A-I7HuupMChTXU"
#define DATABASE_URL                                    "https://dr-soil-db-default-rtdb.firebaseio.com/"

// HTML content stored as a constant string in flash memory to save RAM
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WiFi Configuration</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f2f2f2;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-top: 50px;
            font-size: 2em;
        }
        form {
            background-color: #fff;
            padding: 30px;
            margin: 20px auto;
            width: 400px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px 0px #aaa;
        }
        select, input[type="password"], input[type="submit"] {
            width: 90%;
            padding: 15px;
            margin-top: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-size: 18px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        p {
            color: red;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <h1>WiFi Configuration</h1>
    <form action="/wifi" method="post">
        <div id="networks">
            <!-- Wi-Fi networks will be populated here -->
        </div>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password"><br>
        <input type="submit" value="Submit">
    </form>
</body>
</html>
)rawliteral";

// Create a web server object on port 80
ESP8266WebServer server(80);

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

bool configuration_mode = true;
bool islogged_to_fireBase = false;

String cachedNetworks = "";
unsigned long lastScanTime = 0;
unsigned long sendDataPrevMillis = 0;

// Sensor Pin Assignments
#define RE_PIN 2  // RO pin connected to D4 on ESP8266
#define DE_PIN 0  // DE pin connected to D3 on ESP8266

// Pre-defined commands stored in flash memory for sensors
const byte nitro[] PROGMEM = {0x01, 0x03, 0x00, 0x1e, 0x00, 0x01, 0xe4, 0x0c};  // Nitrogen
const byte phos[] PROGMEM = {0x01, 0x03, 0x00, 0x1f, 0x00, 0x01, 0xb5, 0xcc};  // Phosphorous
const byte pota[] PROGMEM = {0x01, 0x03, 0x00, 0x20, 0x00, 0x01, 0x85, 0xc0};  // Potassium
const byte phCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x06, 0x00, 0x01, 0x64, 0x0B};  // pH
const byte soilHumidityCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x12, 0x00, 0x01, 0x92, 0x92};  // Soil Humidity
const byte temperatureCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x13, 0x00, 0x01, 0xFF, 0x9B};  // Temperature
const byte conductivityCommand[] PROGMEM = {0x01, 0x03, 0x00, 0x15, 0x00, 0x01, 0x95, 0xCE};  // Conductivity

// Software Serial for communication with sensors
SoftwareSerial mod(14, 4);  // RX connected to D5, TX connected to D2


byte Rx_buffer[11];  // Buffer for sensor data

void setup() {
    #if DEBUG_MODE
        Serial.begin(ESP_BAUDRATE);
    #endif

    mod.begin(9600);
    pinMode(RE_PIN, OUTPUT);
    pinMode(DE_PIN, OUTPUT);

    WiFi.softAP(CONFIG_NETWORK_SSID, CONFIG_NETWORK_PASSWORD);
    DEBUG_PRINT("Access Point Started");

    server.on("/", handleConfigurationPage);
    server.on("/wifi", handleWiFiConfig);
    server.onNotFound([]() {
        server.send(404, "text/plain", "404: Not Found");
    });

    server.begin();
    DEBUG_PRINT("HTTP server started");

    Serial.println("NPK Sensor Initializing...");
    delay(3000);
}

void loop() {
    if (configuration_mode) {
        server.handleClient();

    } else if (!islogged_to_fireBase) {
        
      config.api_key = API_KEY;
      config.database_url = DATABASE_URL;

      if(Firebase.signUp(&config, &auth, "", "")) {
        DEBUG_PRINT("SignUp Successful");
        config.token_status_callback = tokenStatusCallback;
        Firebase.begin(&config, &auth);
        Firebase.reconnectWiFi(true);
      
        islogged_to_fireBase = 1;
      } else {
        Serial.printf("%s\n", config.signer.signupError.message.c_str());
      }
    
    } else {
        
        unsigned long currentMillis = millis();
        if (currentMillis - sendDataPrevMillis >= SEND_DATA_INTERVAL_MS) {
            sendDataPrevMillis = currentMillis;
            sendSensorData();
        }

    }
}

void handleConfigurationPage() {
    unsigned long currentTime = millis();

    if (currentTime - lastScanTime > WIFI_SCAN_CACHE_DURATION_MS || cachedNetworks == "") {
        cachedNetworks = scanNetworks();
        lastScanTime = currentTime;
    }

    String html = FPSTR(index_html);
    html.replace("<!-- Wi-Fi networks will be populated here -->", cachedNetworks);

    server.send(200, "text/html", html);

    if (currentTime - lastScanTime > WIFI_SCAN_CACHE_DURATION_MS) {
        cachedNetworks = "";
    }
}

void handleWiFiConfig() {
    if (server.hasArg("ssid") && server.hasArg("password")) {
        String ssid = server.arg("ssid");
        String password = server.arg("password");

        DEBUG_PRINT("SSID: " + ssid);
        DEBUG_PRINT("Password: " + password);

        WiFi.begin(ssid.c_str(), password.c_str());

        if (WiFi.waitForConnectResult() == WL_CONNECTED) {
            DEBUG_PRINT("Connected to Wi-Fi");

            WiFi.softAPdisconnect(true);
            DEBUG_PRINT("AP mode disabled.");
            server.stop();
            configuration_mode = false;
            DEBUG_PRINT("Web server stopped.");

            server.send_P(200, "text/html", PSTR("<!DOCTYPE html><html><body><h1>Connected successfully!</h1><p>ESP8266 will now begin its main operation.</p></body></html>"));
        } else {
            DEBUG_PRINT("Failed to connect. Check password and try again.");
            server.send_P(200, "text/html", PSTR("<!DOCTYPE html><html><body><h1>Connection failed!</h1><p>Please check your SSID and password, and try again.</p></body></html>"));
        }
    }
}

String scanNetworks() {
    char ssid[32];
    char networkOptions[1024];

    strcpy(networkOptions, "<select name=\"ssid\">");

    int numberNetworks_Scanned = WiFi.scanNetworks();
    if (numberNetworks_Scanned == 0) {
        strcat(networkOptions, "<option value=\"none\">No networks found</option>");
    } else {
        for (int i = 0; i < numberNetworks_Scanned; i++) {
            WiFi.SSID(i).toCharArray(ssid, sizeof(ssid));
            const char* securityType = (WiFi.encryptionType(i) == AUTH_OPEN) ? " (Open)" : " (Secured)";

            char networkOption[128];
            snprintf(networkOption, sizeof(networkOption), "<option value=\"%s\">%s (%d dBm)%s</option>", ssid, ssid, WiFi.RSSI(i), securityType);
            strcat(networkOptions, networkOption);
        }
    }
    strcat(networkOptions, "</select>");
    return String(networkOptions);
}

void sendSensorData() {
    float nitrogen = readSensorData(NITROGEN);
    float phosphorous = readSensorData(PHOSPHOROUS);
    float potassium = readSensorData(POTASSIUM);
    float ph = readSensorData(PH) / 10.0;
    float soilHumidity = readSensorData(SOIL_HUMIDITY);
    float temperature = readSensorData(TEMPERATURE);
    float conductivity = readSensorData(CONDUCTIVITY);

    if (Firebase.RTDB.setFloat(&fbdo, "Sensor/Nitrogen", nitrogen) &&
        Firebase.RTDB.setFloat(&fbdo, "Sensor/Phosphorous", phosphorous) &&
        Firebase.RTDB.setFloat(&fbdo, "Sensor/Potassium", potassium) &&
        Firebase.RTDB.setFloat(&fbdo, "Sensor/pH", ph) &&
        Firebase.RTDB.setFloat(&fbdo, "Sensor/SoilHumidity", soilHumidity) &&
        Firebase.RTDB.setFloat(&fbdo, "Sensor/Temperature", temperature) &&
        Firebase.RTDB.setFloat(&fbdo, "Sensor/Conductivity", conductivity)) {
        DEBUG_PRINT("Data has been sent successfully");
    } else {
        DEBUG_PRINT("FAILED: " + fbdo.errorReason());
    }
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

    digitalWrite(DE_PIN, HIGH);
    digitalWrite(RE_PIN, HIGH);
    delay(100);

    if (mod.write(buffer, 8) == 8) {
        digitalWrite(DE_PIN, LOW);
        digitalWrite(RE_PIN, LOW);
        delay(200);

        for (byte i = 0; i < 7; i++) {
            if (mod.available()) {
                Rx_buffer[i] = mod.read();
            }
        }

        int result = (Rx_buffer[3] << 8) | Rx_buffer[4];

        if (sensor == PH) {
            return result / 10.0;
        }

        if (sensor == SOIL_HUMIDITY && result > 100) {
            return 100.0;
        }

        if (sensor == TEMPERATURE && result > 100) {
            return 100.0;
        }

        Serial.print("Parsed value: ");
        Serial.println(result);

        return result;
    }

    Serial.println("Error: Failed to read data.");
    return -1;
}

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