#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>

#define BUFF_SIZE 1024

char *IP_ADDR   = "192.168.56.1";
char *PORT      = "8888";


// socket
void *do_socket(void *arg);

