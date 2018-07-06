/* Arduino UNO with W5100 Ethernetshield or  W5100 Ethernet module, used as MQTT client */

#include <Ethernet.h>
#include "PubSubClient.h" 
#define CLIENT_ID       "ACCAR"
#define PUBLISH_DELAY   3000 // 3 seconds interval
#define DHTPIN          3
#define DHTTYPE         DHT11
#define ledPin 13
#define relayPin 8
String ip = "";
bool statusKD = HIGH;
bool statusBD = HIGH;
bool statusGD = HIGH;
bool relaystate = LOW;
bool pir = LOW;
bool startsend = HIGH;// flag for sending at startup
int lichtstatus;      //contains LDR reading
uint8_t mac[6] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x06};

EthernetClient ethClient;
PubSubClient mqttClient;

long previousMillis;

void callback(char* topic, byte* payload, unsigned int length) {
  char msgBuffer[20];

  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");//MQTT_BROKER
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
  Serial.println(payload[0]);
}

void setup() {

  // setup serial communication

  Serial.begin(9600);
  while (!Serial) {};
  Serial.println(F("MQTT Arduino Demo"));
  Serial.println();

  // setup ethernet communication using DHCP
  if (Ethernet.begin(mac) == 0) {
    //Serial.println(F("Unable to configure Ethernet using DHCP"));
    for (;;);
  }

  Serial.println(F("Ethernet configured via DHCP"));
  Serial.print("IP address: ");
  Serial.println(Ethernet.localIP());
  Serial.println();
 //convert ip Array into String
  ip = String (Ethernet.localIP()[0]);
  ip = ip + ".";
  ip = ip + String (Ethernet.localIP()[1]);
  ip = ip + ".";
  ip = ip + String (Ethernet.localIP()[2]);
  ip = ip + ".";
  ip = ip + String (Ethernet.localIP()[3]);
  //Serial.println(ip);

  // setup mqtt client
  mqttClient.setClient(ethClient);
  // mqttClient.setServer("test.mosquitto.org", 1883);//use public broker
  mqttClient.setServer( "192.168.1.102", 1883); // or local broker
  //Serial.println(F("MQTT client configured"));
  mqttClient.setCallback(callback);
 
  previousMillis = millis();
  mqttClient.publish("home/br/nb/ip", ip.c_str());
}

void loop() {
  mqttClient.loop();
}

void sendData() {
  char msgBuffer[20];
  Serial.println("send test msg to mqtt broker");
  mqttClient.publish("home/test", "this is test msg");
}
