#ifndef BPE_H
#define BPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    char** vocab;
    uint32_t vocab_size;
    uint32_t* merge_t1;  // First token of each merge
    uint32_t* merge_t2;  // Second token of each merge
    uint32_t num_merges;
} BPE;

BPE* init_bpe();
void free_bpe(BPE* bpe);
void train_bpe(BPE* bpe, const char* corpus, size_t corpus_size, uint32_t num_merges);
uint32_t* encode_bpe(BPE* bpe, const char* text, size_t text_len, uint32_t* num_tokens);
char* decode_bpe(BPE* bpe, const uint32_t* tokens, uint32_t num_tokens);
void save_bpe(BPE* bpe, const char* filename);
BPE* load_bpe(const char* filename);

#endif