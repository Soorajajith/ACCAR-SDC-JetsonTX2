//DC모터 & 초음파센서 전면 합친 코드(180704)

#include <NewPing.h>
#include <Servo.h>
#include <SPI.h>

/*To Know Going*/
int flag = 0;

/*for SPI*/
char buf [100];  
volatile byte pos = 0;  
volatile boolean printIt = false;  
#define   spi_enable()   (SPCR |= _BV(SPE))  

/* trigger and echo pins for each sensor*/
#define SONAR1        7
#define SONAR2        8
#define SONAR3        9
#define MAX_DISTANCE  1000  // maximum distance for sensors
#define NUM_SONAR     3     // number of sonar sensors

#define L293N_ENA 6

#define GO    101
#define BACK  102
#define RIGHT 103
#define LEFT  104

int curve=90;
const int SERVO     = 3;
int loop_i          = 0;
int angle           = 0;
Servo myservo;

// array of sonar sensor objects
NewPing sonar[NUM_SONAR] = { 
  NewPing(SONAR1, SONAR1, MAX_DISTANCE),
  NewPing(SONAR2, SONAR2, MAX_DISTANCE),
  NewPing(SONAR3, SONAR3, MAX_DISTANCE)

};

// array stores distances for each(cm)
int distance[NUM_SONAR]; 
int dir;
int speed;

void setup() {

  Serial.begin(9600);
  pinMode(A1, OUTPUT);
  pinMode(A2, OUTPUT);
  
   //Master Input Slave Output 12번핀을 출력으로 설정  
  pinMode(MISO, OUTPUT);

  //slave 모드로 SPI 시작   
  spi_enable();  

   //인터럽트 시작  
  SPI.setClockDivider(SPI_CLOCK_DIV64); //250kHz   
  SPI.setDataMode(SPI_MODE0);  
  SPI.attachInterrupt(); 
  
  myservo.attach(SERVO);        

  // 임시적으로 GO
  speed = 100;
}
   
// SPI 인터럽트 루틴  
ISR (SPI_STC_vect)  
{  
  // SPI 데이터 레지스터로부터 한바이트 가져옴  
  byte c = SPDR;    
    
  //버퍼에 자리가 있다면...  
  if (pos < sizeof buf)  
  {  
    buf[pos++] = c;  
      
    // 출력을 진행한다.   
     if (c == '\0')  
      printIt = true;        
   }   
}    
   
void car_go(){
  Serial.println("GO");
  digitalWrite(A1, LOW);         
  digitalWrite(A2, HIGH); 
  analogWrite(L293N_ENA, speed);
}

void car_back(){
  Serial.println("BACK");
  digitalWrite(A1, HIGH);         
  digitalWrite(A2, LOW);
  analogWrite(L293N_ENA, speed);
}

void car_stop(){
  Serial.println("STOP");
  digitalWrite(A1, LOW);         
  digitalWrite(A2, LOW);
}

void loop() {

  String str2;
  
  if (printIt)  
    {  
      
        buf[pos] = 0;    
        Serial.println (buf);  
        pos = 0;  
        printIt = false;  
        str2 = buf;
    }
  /////// FOR TEST ////////
  //if(Serial.available() == 0)
  //  return;

 // dir = Serial.parseInt();
  /////////////////////////

  // update the distance array
  updateSonar(); 

  // print all distances  
  /*Serial.print("Sonar 1: ");
  Serial.println(distance[0]);

  Serial.print("Sonar 2: ");
  Serial.println(distance[1]);
  
  Serial.print("dir: ");
  Serial.println(buf);

  Serial.print("speed: ");
  Serial.println(speed);
  Serial.println("-------------------");*/


  if(distance[0] == 0 || distance[0] < 3){
 //   return;
  }
  else{
    // 거리에 따른주행
  
    if(distance[0] < 10)
      car_stop();
    else if(str2.compareTo("GO") == 0){
      car_go();
      flag = 1;
    }
    else if(str2.compareTo("BACK") == 0){
      car_back();
      flag = 2;
    }
    else if(str2.compareTo("STOP") == 0){
      car_stop();
      flag = 0;
    }
    else if(flag == 1)
      car_go();
    else if(flag == 2)
      car_back();
      
    if(str2.compareTo("LEFT") == 0){
      curve = curve - 30;
      myservo.write(curve);
      delay(50);
    }
    else if(str2.compareTo("RIGHT") == 0){
        curve = curve+30;
       //Serial.print("curve : ");
       //Serial.println(curve);
       myservo.write(curve);
       delay(50);
    }  
  }
}
   
void updateSonar() {

  for (int i = 0; i < NUM_SONAR; i++) {

    // update distance
    distance[i] = sonar[i].ping_cm(); 
  //  Serial.println(sonar[i].ping_cm());

  }
}
