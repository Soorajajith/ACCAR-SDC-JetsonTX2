#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <getopt.h>
#include <fcntl.h>
#include <stdint.h>


#define BUFF_SIZE 128
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0])) 

// SOCKET
static const char *IP_ADDR   = "192.168.43.160";
static const char *PORT      = "8888";
void *do_socket(void *arg);


// SPI
static const char *device = "/dev/spidev3.0";
static uint8_t mode;
static uint8_t bits = 8;
static uint32_t speed = 250000;
static uint16_t delay;
int fd_spi;

void pabort(const char *s);
void transfer(int fd, char* tx, int len);
void setSPI();
