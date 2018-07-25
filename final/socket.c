#include "project.h"

void *do_socket(void *arg)
{
	char buff_rcv[BUFF_SIZE + 5];
	char buff_snd[BUFF_SIZE + 5];

	int server_socket;
	int client_socket;
	int client_addr_size;

	struct sockaddr_in server_addr;
	struct sockaddr_in client_addr;

	server_socket = socket(PF_INET, SOCK_STREAM, 0);

	if(server_socket == -1){
		printf("server socket 생성 실패\n");
		exit(1);
	}

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;	
	server_addr.sin_port = htons(atoi(PORT));
//	server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	server_addr.sin_addr.s_addr = inet_addr(IP_ADDR);


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
		
		client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_size);

		if(-1 == client_socket){
			printf("클라이언트 연결 수락 실패\n");
			exit(1);
		}

		memset(buff_rcv, 0, sizeof(buff_rcv));
		read(client_socket, buff_rcv, BUFF_SIZE);

		buff_rcv[strlen(buff_rcv) - 1] = '\0';

		// DEBUG
		printf("SOCKET: %s\n", buff_rcv);

		// transfer data to Arduino
		transfer(fd_spi, buff_rcv, BUFF_SIZE);

		close(client_socket);
	}

}
/*
char *change2number(char *str)
{
	if(strcmp(str, "고\n") == 0)
		return "101";
	if(strcmp(str, "오른쪽\n") == 0)
		return "103";
	if(strcmp(str, "왼쪽\n") == 0)
		return "104";
}
*/
