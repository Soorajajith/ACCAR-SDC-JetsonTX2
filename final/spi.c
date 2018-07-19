#include "project.h"

void pabort(const char *s)
{
	perror(s);
	abort();
}

void transfer(int fd, char* tx, int len)  
{  
    uint8_t rx[256];  
    struct spi_ioc_transfer tr = {  
        .tx_buf = (unsigned long)tx,  
        .rx_buf = (unsigned long)rx,  
        .len = len,  
        .delay_usecs = delay,  
        .speed_hz = speed,  
        .bits_per_word = bits,  
    };  
    int ret;  
  
    ret = ioctl(fd, SPI_IOC_MESSAGE(1), &tr);  
    if (ret < 1)  
        pabort("can't send spi message");  
 
    for (ret = 0; ret < len; ret++)   
        printf("%c", rx[ret]);  
	printf("\n");
}  
  
void setSPI()
{  
    int ret = 0;  
  
 
    fd_spi = open(device, O_RDWR);  

    if (fd_spi < 0)  
        pabort("can't open device");  
  
    /* 
     * spi mode 
     */  
    ret = ioctl(fd_spi, SPI_IOC_WR_MODE, &mode);  
    if (ret == -1)  
        pabort("can't set spi mode");  
  
    ret = ioctl(fd_spi, SPI_IOC_RD_MODE, &mode);  
    if (ret == -1)  
        pabort("can't get spi mode");  
  
    /* 
     * bits per word 
     */  
    ret = ioctl(fd_spi, SPI_IOC_WR_BITS_PER_WORD, &bits);  
    if (ret == -1)  
        pabort("can't set bits per word");  
  
    ret = ioctl(fd_spi, SPI_IOC_RD_BITS_PER_WORD, &bits);  
    if (ret == -1)  
        pabort("can't get bits per word");  
  
    /* 
     * max speed hz 
     */  
    ret = ioctl(fd_spi, SPI_IOC_WR_MAX_SPEED_HZ, &speed);  
    if (ret == -1)  
        pabort("can't set max speed hz");  
  
    ret = ioctl(fd_spi, SPI_IOC_RD_MAX_SPEED_HZ, &speed);  
    if (ret == -1)  
        pabort("can't get max speed hz");  
  
    printf("spi mode: %d\n", mode);  
    printf("bits per word: %d\n", bits);  
    printf("max speed: %d Hz (%d KHz)\n", speed, speed/1000);  
 
}
