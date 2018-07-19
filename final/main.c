#include "project.h"

int main()
{
	pthread_t thread;

	if(pthread_create(&thread, NULL, do_socket, NULL) < 0){
		printf("thread create error\n");
		exit(0);
	}

	for(;;);
}


void *do_socket(void *arg)
{
	int server_socket;
	int client_socket;
	int client_addr_size;

	struct sockaddr_in server_addr;
	struct sockaddr_in client_addr;

	char buff_rcv[BUFF_SIZE + 5];
	char buff_snd[BUFF_SIZE + 5];

	server_socket = socket(PF_INET, SOCK_STREAM, 0);

	if(server_socket == -1){
		printf("server socket 생성 실패\n");
		exit(1);
	}

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;	// IPv4 인터넷 프로토콜
	server_addr.sin_port = htons(atoi(PORT));
	server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
//	server_addr.sin_addr.s_addr = inet_addr(IP_ADDR);


	if(-1 == bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr))){

		printf("bind() 실행 에러\n");
		exit(1);
	}

	if(-1 == listen(server_socket, 5)){
		printf("대기상태 모드 설정 실패\n");
		exit(1);
	}

	while(1)
	{
		client_addr_size = sizeof(client_addr);
		
		// 클라이언트와 통신 하기 위해 커널이 자동으로 소켓 생성->client_addr
		client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_size);

		if(-1 == client_socket){
			printf("클라이언트 연결 수락 실패\n");
			exit(1);
		}

		read(client_socket, buff_rcv, BUFF_SIZE);
		printf("receive: %s\n", buff_rcv);
	//	write(client_socket, buff_snd, strlen(buff_snd) + 1);
		close(client_socket);
	}

}
