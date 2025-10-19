#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include "bpe.h"
#include "data.h"

BPE* bpe = NULL;

void handle_sigint(int signum) {
    if (bpe) {
        char tokenizer_filename[64];
        time_t now = time(NULL);
        strftime(tokenizer_filename, sizeof(tokenizer_filename), 
                 "%Y%m%d_%H%M%S_bpe.bin", localtime(&now));
        save_bpe(bpe, tokenizer_filename);
    }
    exit(128 + signum);
}

void test_encode_decode(BPE* bpe, const char* test_text) {
    printf("\n--- Test ---\n");
    printf("Input:  \"%s\"\n", test_text);
    
    size_t text_len = strlen(test_text);
    uint32_t num_tokens;
    uint32_t* tokens = encode_bpe(bpe, test_text, text_len, &num_tokens);
    
    // Print tokens
    printf("Tokens: [");
    uint32_t max_print = 20;
    for (uint32_t i = 0; i < num_tokens && i < max_print; i++) {
        printf("%u", tokens[i]);
        if (i < num_tokens - 1 && i < max_print - 1) printf(", ");
    }
    if (num_tokens > max_print) printf(", ... (%u total)", num_tokens);
    printf("]\n");
    
    printf("Count:  %zu bytes -> %u tokens (%.1f%% compression)\n", 
           text_len, num_tokens, 100.0 * (1.0 - (double)num_tokens / text_len));
    
    char* decoded = decode_bpe(bpe, tokens, num_tokens);
    printf("Output: \"%s\"\n", decoded);
    
    if (strcmp(test_text, decoded) == 0) {
        printf("Status: ✓ Perfect match!\n");
    } else {
        printf("Status: ✗ Mismatch detected!\n");
    }
    
    free(tokens);
    free(decoded);
}

int main(int argc, char** argv) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);
    
    const uint32_t target_vocab_size = 1024;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("corpus.txt", &corpus_size);
    if (!corpus) {
        return 1;
    }
    
    // Initialize or load BPE
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        bpe = load_bpe(argv[1]);
        if (!bpe) {
            printf("Failed to load checkpoint, initializing new tokenizer...\n");
            bpe = init_bpe(target_vocab_size);
        }
    } else {
        printf("Initializing new tokenizer...\n");
        bpe = init_bpe(target_vocab_size);
    }
    
    // Train if not fully trained
    if (bpe->vocab_size < bpe->target_vocab_size) {
        train_bpe(bpe, corpus, corpus_size);
    } else {
        printf("Tokenizer already fully trained (vocab size: %u)\n", bpe->vocab_size);
    }
    
    // Show vocabulary sample
    print_vocab_bpe(bpe, 20);
    
    // Save tokenizer with timestamped filename
    char tokenizer_fname[64];
    time_t now = time(NULL);
    strftime(tokenizer_fname, sizeof(tokenizer_fname), "%Y%m%d_%H%M%S_bpe.bin", localtime(&now));
    save_bpe(bpe, tokenizer_fname);
    
    // Testing phase
    test_encode_decode(bpe, "Hello, world!");
    test_encode_decode(bpe, "BPE tokenization is awesome!");
    test_encode_decode(bpe, "The quick brown fox jumps over the lazy dog.");
    
    // Corpus statistics
    print_stats_bpe(bpe, corpus, corpus_size);
    
    // Verification: Load the saved tokenizer and verify
    printf("\nVerifying saved tokenizer...\n");
    BPE* loaded_bpe = load_bpe(tokenizer_fname);
    
    if (loaded_bpe) {
        const char* verify_text = "Verification test string.";
        uint32_t num_tokens1, num_tokens2;
        uint32_t* tokens1 = encode_bpe(bpe, verify_text, strlen(verify_text), &num_tokens1);
        uint32_t* tokens2 = encode_bpe(loaded_bpe, verify_text, strlen(verify_text), &num_tokens2);
        
        bool match = (num_tokens1 == num_tokens2);
        if (match) {
            for (uint32_t i = 0; i < num_tokens1; i++) {
                if (tokens1[i] != tokens2[i]) {
                    match = false;
                    break;
                }
            }
        }
        
        if (match) {
            printf("Verification: ✓ Loaded tokenizer produces identical results\n");
        } else {
            printf("Verification: ✗ Loaded tokenizer differs from original\n");
        }
        
        free(tokens1);
        free(tokens2);
        free_bpe(loaded_bpe);
    }
    
    // Cleanup
    free(corpus);
    free_bpe(bpe);
    
    printf("\n✓ All done!\n\n");
    
    return 0;
}