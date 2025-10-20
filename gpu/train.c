#include <stdio.h>
#include <time.h>
#include "bpe.h"
#include "../data.h"

int main() {
    size_t corpus_size;
    char* corpus = load_corpus("../../corpus.txt", &corpus_size);
    if (!corpus) return 1;
    
    // Train
    BPE* bpe = init_bpe();
    train_bpe(bpe, corpus, corpus_size, 8192);
    
    // Get timestamp for filename
    char bpe_fname[64];
    time_t now = time(NULL);
    strftime(bpe_fname, sizeof(bpe_fname), "%Y%m%d_%H%M%S_bpe.bin", localtime(&now));
    
    // Save
    save_bpe(bpe, bpe_fname);
    free_bpe(bpe);
    
    // Load
    BPE* loaded_bpe = load_bpe(bpe_fname);
    if (!loaded_bpe) {
        free(corpus);
        return 1;
    }

    // Tokenize entire corpus
    printf("\n=== Corpus Compression ===\n");
    printf("Encoding %zu bytes...\n", corpus_size);
    uint32_t num_tokens;
    uint32_t* tokens = encode_bpe(loaded_bpe, corpus, corpus_size, &num_tokens);
    printf("Compression ratio: %.2f:1\n", (double)corpus_size / (double)num_tokens);

    // Verify decode
    printf("Verifying decode...\n");
    char* decoded = decode_bpe(loaded_bpe, tokens, num_tokens);
    if (memcmp(corpus, decoded, corpus_size) == 0) printf("✓ Perfect match!\n");
    else printf("✗ Mismatch!\n");
    
    // Cleanup
    free(tokens);
    free(decoded);
    free(corpus);
    free_bpe(loaded_bpe);
    
    return 0;
}