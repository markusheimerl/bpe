#include "bpe.h"
#include "data.h"

void print_tokens(uint32_t* tokens, uint32_t num_tokens, uint32_t max_print) {
    printf("[");
    for (uint32_t i = 0; i < num_tokens && i < max_print; i++) {
        printf("%u", tokens[i]);
        if (i < num_tokens - 1 && i < max_print - 1) printf(", ");
    }
    if (num_tokens > max_print) printf(", ... (%u total)", num_tokens);
    printf("]\n");
}

void test_encode_decode(BPETokenizer* tokenizer, const char* test_text) {
    printf("\n--- Test ---\n");
    printf("Input:  \"%s\"\n", test_text);
    
    size_t text_len = strlen(test_text);
    uint32_t num_tokens;
    uint32_t* tokens = bpe_encode(tokenizer, test_text, text_len, &num_tokens);
    
    printf("Tokens: ");
    print_tokens(tokens, num_tokens, 20);
    printf("Count:  %zu bytes -> %u tokens (%.1f%% compression)\n", 
           text_len, num_tokens, 100.0 * (1.0 - (double)num_tokens / text_len));
    
    char* decoded = bpe_decode(tokenizer, tokens, num_tokens);
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
    const char* corpus_file = "corpus.txt";
    const char* tokenizer_file = "bpe_tokenizer.bin";
    uint32_t num_merges = 1000;
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║   BPE Tokenizer - Training Program    ║\n");
    printf("╚════════════════════════════════════════╝\n");
    
    if (argc > 1) {
        num_merges = atoi(argv[1]);
        printf("\nUsing %u merges (from command line)\n", num_merges);
    }
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus(corpus_file, &corpus_size);
    if (!corpus) {
        return 1;
    }
    
    // Create and train tokenizer
    BPETokenizer* tokenizer = bpe_create();
    bpe_train(tokenizer, corpus, corpus_size, num_merges);
    
    // Show some vocabulary entries
    bpe_print_vocab(tokenizer, 270);
    
    // Save tokenizer
    bpe_save(tokenizer, tokenizer_file);
    
    // Test examples
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║            Testing Phase               ║\n");
    printf("╚════════════════════════════════════════╝\n");
    
    test_encode_decode(tokenizer, "Hello, world!");
    test_encode_decode(tokenizer, "BPE tokenization is awesome!");
    test_encode_decode(tokenizer, "The quick brown fox jumps over the lazy dog.");
    
    // Full corpus statistics
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║         Corpus Statistics              ║\n");
    printf("╚════════════════════════════════════════╝\n");
    
    uint32_t total_tokens;
    uint32_t* corpus_tokens = bpe_encode(tokenizer, corpus, corpus_size, &total_tokens);
    
    printf("Original size:      %zu bytes\n", corpus_size);
    printf("Tokenized size:     %u tokens\n", total_tokens);
    printf("Compression ratio:  %.2f%%\n", 100.0 * (1.0 - (double)total_tokens / corpus_size));
    printf("Bytes per token:    %.2f\n", (double)corpus_size / total_tokens);
    printf("Vocabulary size:    %u\n", tokenizer->vocab_size);
    
    free(corpus_tokens);
    free(corpus);
    bpe_free(tokenizer);
    
    printf("\n✓ All done!\n\n");
    
    return 0;
}