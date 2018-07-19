ACCAR: main.o spi_test.o
	gcc -o ACCAR -pthread main.o spi_test.o

main.o: project.h main.c
	gcc -c main.c

spi_test.o: project.h spi_test.c
	gcc -c spi_test.c

clean:
	rm main.o
	rm spi_test.o
	rm ACCAR
