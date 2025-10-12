#ifndef BPE_H
#define BPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_VOCAB_SIZE 50000
#define INITIAL_VOCAB_SIZE 256

typedef struct {
    uint32_t token1;
    uint32_t token2;
} Merge;

typedef struct {
    uint32_t vocab_size;
    uint32_t num_merges;
    Merge* merges;
    char** vocab;
    uint32_t* vocab_lens;
} BPETokenizer;

BPETokenizer* bpe_create();
void bpe_free(BPETokenizer* tokenizer);
void bpe_train(BPETokenizer* tokenizer, const char* corpus, size_t corpus_size, uint32_t num_merges);
uint32_t* bpe_encode(BPETokenizer* tokenizer, const char* text, size_t text_len, uint32_t* num_tokens);
char* bpe_decode(BPETokenizer* tokenizer, const uint32_t* tokens, uint32_t num_tokens);
int bpe_save(BPETokenizer* tokenizer, const char* filename);
BPETokenizer* bpe_load(const char* filename);
void bpe_print_vocab(BPETokenizer* tokenizer, uint32_t max_entries);

#endif