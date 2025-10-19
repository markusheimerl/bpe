#ifndef BPE_H
#define BPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INITIAL_VOCAB_SIZE 256

typedef struct {
    uint32_t token1;
    uint32_t token2;
} Merge;

typedef struct {
    char** vocab;
    uint32_t* vocab_lens;
    uint32_t vocab_size;
    uint32_t num_merges;
    Merge* merges;
} BPE;

BPE* init_bpe(uint32_t target_vocab_size);
void free_bpe(BPE* bpe);
void train_bpe(BPE* bpe, const char* corpus, size_t corpus_size);
uint32_t* encode_bpe(BPE* bpe, const char* text, size_t text_len, uint32_t* num_tokens);
char* decode_bpe(BPE* bpe, const uint32_t* tokens, uint32_t num_tokens);

#endif