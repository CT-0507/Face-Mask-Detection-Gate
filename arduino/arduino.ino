#include <ArduinoJson.h>
#include <Servo.h>

Servo servoPin;
int blueLed = 2;
int redLed = 3;
int yellowLed = 4;
int buzzer = 6;
int leds[] = {blueLed, redLed, yellowLed};
int pir  = 7;
int incomingByte;
StaticJsonDocument<84> doc;

unsigned int sendDataDelay = 10000;
String content = "";
unsigned long lastSendData;
unsigned long lastResetLed;
bool isOpen = false;
bool isOpenLed = false;
unsigned long lastOpen;
unsigned long openDelay = 20000; // Mở 20s
unsigned long hornDelay = 5000;
unsigned long resetLedDelay = 7000; // 7s

bool detect = false;

void setup() {
  Serial.begin(9600);
  servoPin.attach(5);
  pinMode(buzzer, OUTPUT);
  pinMode(pir, INPUT);
  
  
  for(int i =0 ; i < 3; i++){
    pinMode(leds[i], OUTPUT);
  }
  // put your setup code here, to run once:
  servoPin.write(90);
  lastSendData = millis();
  lastOpen = millis();
  lastResetLed = millis();
  
  digitalWrite(yellowLed, HIGH);
  
}


void acceptSound(){
  tone(buzzer, 700, 2000);  
}
void denySound(){
//  int i = 0;
//  while (i < 3){
//    tone(buzzer, 500);
//    delay(500);
//    noTone(buzzer);
//    delay(300);  
//    i++;
//  }
//  noTone(buzzer);
  tone(buzzer, 1000, 3000);
}

int i = 0;

void loop() {
  while(Serial.available() > 0){
//    incomingByte = Serial.read();
//    content.concat(char(incomingByte));
      content = Serial.readStringUntil('\n');

      if (content != ""){
        DeserializationError error = deserializeJson(doc, content);
        if (error) return;
      }
        bool servo = doc["servo"];
        bool led1 = doc["leds"][0];
        bool led2 = doc["leds"][1];
        bool led3 = doc["leds"][2];
        bool horn = doc["horn"];
        
        if (servo){
            isOpen = true;
            servoPin.write(0); // Mở cửa, reset thời gian mở cửa
            lastOpen = millis(); 
        } else {
           isOpen = false;
           servoPin.write(90);
        }

        for(int i =0 ; i < 3; i++){
          bool led = doc["leds"][i];
          if (doc["leds"][i]){
            digitalWrite(leds[i], HIGH);
            lastResetLed = millis();
            isOpenLed = true;
          } else {
            digitalWrite(leds[i], LOW);
          }
        }

        if (horn){
          acceptSound();
        } else {
          denySound();
        }
       
      }
  }
  if(content != ""){
    content = "";
  }


  if (millis() - lastOpen > openDelay){
    if (isOpen){
      servoPin.write(90); 
    }
  }

  if (millis() - lastResetLed > resetLedDelay){
    if(isOpenLed){
      isOpenLed = false;
      turnOnYellowLed();
    }
  }
  
  
//  if (millis() - lastSendData > sendDataDelay){
//    String isOpenStr = "\"isOpen\":\"" +  String(isOpen)+"\"";
//    String detectPeople =  "\"people\":\"true\"";
//    String data = "{" + detectPeople + "," + isOpenStr + "}";
//    Serial.print(data);
//    lastSendData = millis();
//    Serial.flush();
//  }

  if (digitalRead(pir) == HIGH){
    i++;
    if (i > 5 ){
    String isOpenStr = "\"isOpen\":\"" +  String(isOpen)+"\"";
    String data = "{" + isOpenStr + "}";
    Serial.print(data);
    if (isOpen){
      servoPin.write(90);
    }
    
//    Serial.println(i);
    Serial.flush();
    }
    delay(200);
  } else {
    if (i>0){
      i--;  
    }
  }

  
}

void turnOnYellowLed(){
  digitalWrite(yellowLed, HIGH);
  digitalWrite(redLed, LOW);
  digitalWrite(blueLed, LOW);
}


/*
Reference:
https://create.arduino.cc/projecthub/ansh2919/serial-communication-between-python-and-arduino-e7cce0
https://pythonforundergradengineers.com/python-arduino-LED.html
https://stackoverflow.com/questions/5697047/convert-serial-read-into-a-useable-string-using-arduino

*/
