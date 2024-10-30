#define BAUD_RATE       9600            
#define SENSOR_PIN      A0

float percentage;

void setup() {
  Serial.begin(BAUD_RATE); 
}

void loop() {

  percentage = ( analogRead(SENSOR_PIN) / 1023.0) * 100; 

  Serial.print("Soil Moisture: "); 

  Serial.print(percentage); 

  Serial.println("%");
  delay(500); 
}
