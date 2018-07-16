#include <stdint.h>  
#include <unistd.h>  
#include <stdio.h>  
#include <stdlib.h>  
#include <getopt.h>  
#include <fcntl.h>  
#include <sys/ioctl.h>  
#include <linux/types.h>  
#include <linux/spi/spidev.h>  
  
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))  
  
static void pabort(const char *s)  
{  
    perror(s);  
    abort();  
}  
  
static const char *device = "/dev/spidev3.0";  
static uint8_t mode;  
static uint8_t bits = 8;  
static uint32_t speed = 250000;  
static uint16_t delay;  
  
static void transfer(int fd, char* tx, int len)  
{  
    int ret;  
    uint8_t rx[256];  
    struct spi_ioc_transfer tr = {  
        .tx_buf = (unsigned long)tx,  
        .rx_buf = (unsigned long)rx,  
        .len = len,  
        .delay_usecs = delay,  
        .speed_hz = speed,  
        .bits_per_word = bits,  
    };  
  
    ret = ioctl(fd, SPI_IOC_MESSAGE(1), &tr);  
    if (ret < 1)  
        pabort("can't send spi message");  
 
    for (ret = 0; ret < len; ret++)   
        printf("%c", rx[ret]);  
    puts("\n");  
}  
  
  
int main(int argc, char *argv[])  
{  
    int ret = 0;  
    int fd;  
  
  
    fd = open(device, O_RDWR);  
    if (fd < 0)  
        pabort("can't open device");  
  
    /* 
     * spi mode 
     */  
    ret = ioctl(fd, SPI_IOC_WR_MODE, &mode);  
    if (ret == -1)  
        pabort("can't set spi mode");  
  
    ret = ioctl(fd, SPI_IOC_RD_MODE, &mode);  
    if (ret == -1)  
        pabort("can't get spi mode");  
  
    /* 
     * bits per word 
     */  
    ret = ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits);  
    if (ret == -1)  
        pabort("can't set bits per word");  
  
    ret = ioctl(fd, SPI_IOC_RD_BITS_PER_WORD, &bits);  
    if (ret == -1)  
        pabort("can't get bits per word");  
  
    /* 
     * max speed hz 
     */  
    ret = ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);  
    if (ret == -1)  
        pabort("can't set max speed hz");  
  
    ret = ioctl(fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed);  
    if (ret == -1)  
        pabort("can't get max speed hz");  
  
    printf("spi mode: %d\n", mode);  
    printf("bits per word: %d\n", bits);  
    printf("max speed: %d Hz (%d KHz)\n", speed, speed/1000);  
 
    char buf[]="echo test";  
    transfer(fd, buf, ARRAY_SIZE(buf));  
  
    close(fd);  
  
    return ret;  
}
