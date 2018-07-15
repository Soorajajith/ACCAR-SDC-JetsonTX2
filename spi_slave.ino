#include <SPI.h>  
   
char buf [100];  
volatile byte pos = 0;  
volatile boolean printIt = false;  
#define   spi_enable()   (SPCR |= _BV(SPE))  
  
  
void setup (void)  
{  
  //시리얼 통신 초기화  
  Serial.begin (9600);  
   
  //Master Input Slave Output 12번핀을 출력으로 설정  
  pinMode(MISO, OUTPUT);  
    
  //slave 모드로 SPI 시작   
  spi_enable();  
   
  //인터럽트 시작  
  SPI.setClockDivider(SPI_CLOCK_DIV64); //250kHz   
  SPI.setDataMode(SPI_MODE0);  
  SPI.attachInterrupt();  
   
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
   
  
void loop (void)  
{  
  if (printIt)  
    {  
        buf[pos] = 0;    
        Serial.println (buf);  
        pos = 0;  
        printIt = false;  
    }    
      
}  

