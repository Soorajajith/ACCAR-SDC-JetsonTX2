/* Arduino UNO with W5100 Ethernetshield or W5100 Ethernet module, used as MQTT client */
// http://www.instructables.com/id/Arduino-Ethernet-Shield-Tutorial/
// MQTT with Arduino Ethernet Shield Youtube
// https://www.youtube.com/watch?v=CjG0JXCGye0
#include <Ethernet.h>
#include "PubSubClient.h" 

#define CLIENT_ID       "ACCAR"
#define PUBLISH_DELAY   3000 // 3 seconds interval
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


/* Enter a MAC address for your controller below.
   Newer Ethernet shield have a MAC address printed on a stiker on the shield
*/
uint8_t mac[6] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x06};

/* Initialize the Ethernet client library
   with the IP address and port of the server
   that you want to connect to (port 80 is default for HTTP):
*/
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

  // start the serial library:
  Serial.begin(9600);

  while (!Serial) {};
  Serial.println("MQTT Arduino Demo start!");

  // start the Ethernet connection:
  if (Ethernet.begin(mac) == 0) {
    Serial.println("Failed to configure Ethernet using DHCP");
	// no point in carrying on, so do nothing forevermore:
    for (;;);
  }

  // print your local IP address:
  Serial.println(Ethernet.localIP());

  // setup mqtt client
  mqttClient.setClient(ethClient);

  /* Creates an uninitialised client instance.
     Before it can be used, it must be configured with the property setters: */
  // mqttClient.setServer("test.mosquitto.org", 1883);//use public broker
  mqttClient.setServer( "192.168.1.102", 1883); // or local broker
  // client is now configured for use

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
