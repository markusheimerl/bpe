CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra --cuda-gpu-arch=sm_86 -x cuda
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lm -flto

all: train.out

train.out: bpe.o data.o train.o
	$(CC) bpe.o data.o train.o $(LDFLAGS) -o $@

bpe.o: bpe.c bpe.h
	$(CC) $(CFLAGS) -c bpe.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c bpe.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@./train.out

clean:
	rm -f *.out *.o