ACCAR: main.o spi.o socket.o
	gcc -o ACCAR -pthread main.o spi.o socket.o
	gcc -shared -o spi.so spi.o
#	python line_stream.py

main.o: project.h main.c
	gcc -c main.c

spi.o: project.h spi.c
	gcc -fPIC -c spi.c
#	gcc -c spi.c

socket.o: project.h socket.c
	gcc -c socket.c

clean:
	rm socket.o
	rm main.o
	rm spi.o
	rm spi.so
	rm ACCAR
