#include <stdio.h>
#include "bpe.h"
#include "data.h"

void test_encode_decode(BPE* bpe, const char* test_text) {
    printf("\n--- Test ---\n");
    printf("Input:  \"%s\"\n", test_text);
    
    size_t text_len = strlen(test_text);
    uint32_t num_tokens;
    uint32_t* tokens = encode_bpe(bpe, test_text, text_len, &num_tokens);
    
    printf("Tokens: [");
    for (uint32_t i = 0; i < num_tokens && i < 20; i++) {
        printf("%u", tokens[i]);
        if (i < num_tokens - 1 && i < 19) printf(", ");
    }
    if (num_tokens > 20) printf(", ...");
    printf("]\n");
    
    printf("Count:  %zu bytes -> %u tokens\n", text_len, num_tokens);
    
    char* decoded = decode_bpe(bpe, tokens, num_tokens);
    printf("Output: \"%s\"\n", decoded);
    
    if (strcmp(test_text, decoded) == 0) {
        printf("Status: ✓ Match!\n");
    } else {
        printf("Status: ✗ Mismatch!\n");
    }
    
    free(tokens);
    free(decoded);
}

int main() {
    size_t corpus_size;
    char* corpus = load_corpus("corpus.txt", &corpus_size);
    if (!corpus) return 1;
    
    BPE* bpe = init_bpe();
    train_bpe(bpe, corpus, corpus_size, 1024);
    
    test_encode_decode(bpe, "Hello, world!");
    test_encode_decode(bpe, "<|bos|>BPE tokenization is awesome!");
    
    free(corpus);
    free_bpe(bpe);
    
    printf("\n✓ Done!\n");
    return 0;
}