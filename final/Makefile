ACCAR: main.o spi.o
	gcc -o ACCAR -pthread main.o spi.o

main.o: project.h main.c
	gcc -c main.c

spi.o: project.h spi.c
	gcc -c spi.c

clean:
	rm main.o
	rm spi.o
	rm ACCAR
