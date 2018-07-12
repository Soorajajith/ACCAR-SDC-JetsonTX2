//DC모터 & 초음파센서 전면 합친 코드(180704)

#include <NewPing.h>
#include <Servo.h>

const int L293N_IN1 = 2;
const int L293N_IN2 = 3;
const int L293N_ENA = 10;
const int SERVO = 11;
int loop_i = 0;
Servo myservo;

// trigger and echo pins for each sensor
#define SONAR1 6
#define SONAR2 7
#define SONAR3 8
#define MAX_DISTANCE 1000 // maximum distance for sensors
#define NUM_SONAR 3       // number of sonar sensors

#define GO    1
#define BACK  2
#define RIGHT 3
#define LEFT  4
 
// array of sonar sensor objects
NewPing sonar[NUM_SONAR] = { 
  NewPing(SONAR1, SONAR1, MAX_DISTANCE),
  NewPing(SONAR2, SONAR2, MAX_DISTANCE),
  NewPing(SONAR3, SONAR3, MAX_DISTANCE)

};

// array stores distances for each(cm)
int distance[NUM_SONAR]; 
int dir;

void setup() {

  Serial.begin(9600);
  pinMode(L293N_IN1, OUTPUT);     // 제어 1번핀 출력모드 설정
  pinMode(L293N_IN2, OUTPUT);     // 제어 2번핀 출력모드 설정
  pinMode(L293N_ENA, OUTPUT);     // PWM제어핀 출력모드 설정
  myservo.attach(SERVO);          // Servo PWM핀 출력설정

  // 임시적으로 GO
  dir = GO;
}

void car_go(){
  digitalWrite(L293N_IN1, LOW);         
  digitalWrite(L293N_IN2, HIGH);
  analogWrite(L293N_ENA, 150);
}

void car_stop(){
  digitalWrite(L293N_IN1, LOW);         
  digitalWrite(L293N_IN2, LOW);
  analogWrite(L293N_ENA, 0);
}

void loop() {

  // update the distance array
  updateSonar(); 

  // print all distances
  /*
  Serial.print("Sonar 1: ");
  Serial.print(distance[0]);
  
  Serial.print("  Sonar 2: ");
  Serial.print(distance[1]);

  Serial.print("  Sonar 3: ");
  Serial.println(distance[2]);
  */

  // 거리에 따른주행
 // if(dir == GO){
    
    if(distance[0] < 10)
      car_stop();
    else
      car_go();
 // }

  // servo 미완성
  /*
  loop_i = loop_i+10;
  myservo.write(loop_i%180);
  */

 /*
  data = Serial.read();
  Serial.write(data);
 */
}


/* takes a new reading from each sensor
   and updates the distance array */

void updateSonar() {

  for (int i = 0; i < NUM_SONAR; i++) {

    distance[i] = sonar[i].ping_cm(); // update distance

    // sonar sensors return 0 if no obstacle is detected
    // change distance to max value if no obstacle is detected

    if (distance[i] == 0)
      distance[i] = MAX_DISTANCE;

  }
}
