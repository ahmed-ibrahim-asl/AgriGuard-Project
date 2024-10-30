#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ESP8266httpUpdate.h> // For OTA updates
#include <ArduinoJson.h>       // For parsing JSON responses

#define BAUD_RATE             115200
char current_version[6] = "1.0.0";

/********************** config_wifi_credentials ***********************/
#define WIFI_SSID             "Test_Network"
#define WIFI_PASSWORD         "123456789"
/**********************************************************************/

/************************ remote_server_config ************************/
#define SERVER_URL            "http://192.168.1.3:5000/api/check_update"
#define DEVICE_ID             "device_001"
#define UPDATE_INTERVAL       10000           // 10000ms = 10sec 
/**********************************************************************/

/************************** timing_variables **************************/
unsigned long previous_millis = 0;
/**********************************************************************/

void setup() {
    Serial.begin(BAUD_RATE);
    connect_to_wifi(10, true); // Enable debug mode to see connection messages
}

void loop() {
    unsigned long current_millis = millis(); // Initialize current_millis here

    if (current_millis - previous_millis >= UPDATE_INTERVAL) {
        previous_millis = current_millis;

        check_for_updates(true); // Enable debug mode to see update messages
    }

    /****************** reset_application ******************/
    /*******************************************************/
}

/** 
 * Connect to Wi-Fi network
 * @param max_number_tries Maximum number of connection attempts
 * @param debug_mode Enable or disable debug messages
 */
void connect_to_wifi(int max_number_tries, bool debug_mode) {
    Serial.println("Connecting to WiFi...");

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int retry_counter = 0;

    while (WiFi.status() != WL_CONNECTED && retry_counter < max_number_tries) {
        delay(1000);

        if (debug_mode) {
            Serial.print(".");
        }
        retry_counter++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        if (debug_mode) {
            Serial.println("\nConnected to WiFi");
            Serial.print("IP Address: ");
            Serial.println(WiFi.localIP());
        }
    } else {
        if (debug_mode) {
            Serial.println("\nFailed to connect to WiFi");
        }
        // Handle the error as needed (e.g., restart, go to sleep, etc.)
    }
}

/** 
 * Create JSON payload for the update check request
 * @param debug_mode Enable or disable debug messages
 * @return Pointer to the JSON payload string
 */
const char* create_json_payload(bool debug_mode) {
    // Persist across function calls
    static char json_payload[256];

    StaticJsonDocument<200> doc;
    doc["device_id"] = DEVICE_ID;
    doc["current_version"] = current_version;

    // Serialize JSON document to string
    serializeJson(doc, json_payload, sizeof(json_payload));

    if (debug_mode) {
        Serial.print("Payload: ");
        Serial.println(json_payload);
    }

    return json_payload;
}

/** 
 * Check for firmware updates and perform OTA update if available
 * @param debug_mode Enable or disable debug messages
 */
void check_for_updates(bool debug_mode) {
    if (WiFi.status() == WL_CONNECTED) {
        WiFiClient client;
        HTTPClient http;

        int http_respond_code;

        Serial.println("Checking for updates...");

        if (http.begin(client, SERVER_URL)) {
            http.addHeader("Content-Type", "application/json");

            http_respond_code = http.POST(create_json_payload(debug_mode));

            if (http_respond_code > 0) {
                if (debug_mode) {
                    Serial.print("HTTP Response code: ");
                    Serial.println(http_respond_code);
                }

                String response = http.getString();
                if (debug_mode) {
                    Serial.println("Response: " + response);
                }

                // Handle the response
                handleUpdateAndPerformOTA(response.c_str(), client);
            } else {
                Serial.print("Error on HTTP request: ");
                Serial.println(http_respond_code);
            }

            http.end();
        } else {
            Serial.println("Failed to begin HTTP connection.");
        }

    } else {
        Serial.println("WiFi not connected, attempting to reconnect...");
        connect_to_wifi(10, true);
    }
}

/** 
 * Handle the server's response to the update check
 * @param response The JSON response from the server
 * @param client The WiFiClient used for the OTA update
 */
void handleUpdateAndPerformOTA(const char* response, WiFiClient& client) {
    // Create a buffer for storing the JSON data (512 bytes)
    StaticJsonDocument<512> doc;

    DeserializationError error = deserializeJson(doc, response);

    if (error) {
        Serial.print("Failed to parse JSON: ");
        Serial.println(error.c_str());
        return;
    }

    bool updateAvailable = doc["update_available"];
    if (updateAvailable) {
        const char* new_version = doc["version"];
        const char* firmwareUrl = doc["url"];

        Serial.println("Update available!");
        Serial.print("Version: ");
        Serial.println(new_version);
        Serial.print("Downloading firmware from: ");
        Serial.println(firmwareUrl);

        // Perform the OTA update with the firmware URL
        t_httpUpdate_return update_result = ESPhttpUpdate.update(client, firmwareUrl, current_version);

        switch (update_result) {
            case HTTP_UPDATE_FAILED:
                Serial.printf("UPDATE_FAILED Error (%d): %s\n", ESPhttpUpdate.getLastError(),
                              ESPhttpUpdate.getLastErrorString().c_str());
                break;

            case HTTP_UPDATE_NO_UPDATES:
                Serial.println("No Updates Available.");
                break;

            case HTTP_UPDATE_OK:
                Serial.println("Update successful! Device will restart now.");
                // Update was successful; device will restart automatically
                break;
        }

    } else {
        // No update was available according to the response
        Serial.println("No update available.");
    }
}
