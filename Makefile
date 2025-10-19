CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart

all: train.out

train.out: bpe.o data.o train.o
	$(CC) bpe.o data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

bpe.o: bpe.c bpe.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c bpe.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c data.c -o $@

train.o: train.c bpe.h data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o