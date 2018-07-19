#include "project.h"

int main()
{
	pthread_t thread;

	setSPI();

	if(pthread_create(&thread, NULL, do_socket, NULL) < 0){
		printf("thread create error\n");
		exit(0);
	}

	for(;;);
}

