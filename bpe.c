#include "bpe.h"

BPE* init_bpe() {
    BPE* bpe = (BPE*)malloc(sizeof(BPE));
    bpe->vocab_size = 256;
    bpe->vocab = (char**)malloc(256 * sizeof(char*));
    
    // Initialize base vocabulary (all bytes)
    for (uint32_t i = 0; i < 256; i++) {
        bpe->vocab[i] = (char*)malloc(2);
        bpe->vocab[i][0] = (char)i;
        bpe->vocab[i][1] = '\0';
    }
    
    return bpe;
}

void free_bpe(BPE* bpe) {
    if (!bpe) return;
    for (uint32_t i = 0; i < bpe->vocab_size; i++) free(bpe->vocab[i]);
    free(bpe->vocab);
    free(bpe);
}

void train_bpe(BPE* bpe, const char* corpus, size_t corpus_size, uint32_t max_vocab_size) {
    printf("\n=== Training BPE ===\n");
    
    // Initialize token sequence as bytes
    uint32_t* tokens = (uint32_t*)malloc(corpus_size * sizeof(uint32_t));
    uint32_t num_tokens = corpus_size;
    for (size_t i = 0; i < corpus_size; i++) {
        tokens[i] = (unsigned char)corpus[i];
    }
    
    // Pre-allocate pair counts array for maximum vocab size
    uint32_t num_merges = max_vocab_size - 256;
    size_t map_size = (size_t)max_vocab_size * max_vocab_size;
    uint32_t* pair_counts = (uint32_t*)malloc(map_size * sizeof(uint32_t));
    
    // Do num_merges iterations
    for (uint32_t merge_iter = 0; merge_iter < num_merges; merge_iter++) {
        // Reset pair counts
        memset(pair_counts, 0, map_size * sizeof(uint32_t));
        
        // Count pairs
        for (uint32_t i = 0; i < num_tokens - 1; i++) {
            uint32_t t1 = tokens[i];
            uint32_t t2 = tokens[i + 1];
            size_t idx = (size_t)t1 * max_vocab_size + t2;
            pair_counts[idx]++;
        }
        
        // Find most frequent pair
        uint32_t best_t1 = 0, best_t2 = 0, best_count = 0;
        for (uint32_t t1 = 0; t1 < bpe->vocab_size; t1++) {
            for (uint32_t t2 = 0; t2 < bpe->vocab_size; t2++) {
                size_t idx = (size_t)t1 * max_vocab_size + t2;
                uint32_t count = pair_counts[idx];
                if (count > best_count) {
                    best_count = count;
                    best_t1 = t1;
                    best_t2 = t2;
                }
            }
        }
        
        if (best_count == 0) break;
        
        // Grow vocab array
        bpe->vocab = (char**)realloc(bpe->vocab, (bpe->vocab_size + 1) * sizeof(char*));
        uint32_t new_token = bpe->vocab_size;
        
        // Create new vocab entry by concatenating
        size_t len1 = strlen(bpe->vocab[best_t1]);
        size_t len2 = strlen(bpe->vocab[best_t2]);
        bpe->vocab[new_token] = (char*)malloc(len1 + len2 + 1);
        strcpy(bpe->vocab[new_token], bpe->vocab[best_t1]);
        strcat(bpe->vocab[new_token], bpe->vocab[best_t2]);
        bpe->vocab_size++;
        
        // Replace all occurrences in token sequence
        uint32_t write_pos = 0;
        for (uint32_t i = 0; i < num_tokens; i++) {
            if (i < num_tokens - 1 && tokens[i] == best_t1 && tokens[i + 1] == best_t2) {
                tokens[write_pos++] = new_token;
                i++;  // Skip next token
            } else {
                tokens[write_pos++] = tokens[i];
            }
        }
        num_tokens = write_pos;
        
        printf("Merge %u: (%u, %u) -> %u | count: %u | tokens: %u\n", merge_iter + 1, best_t1, best_t2, new_token, best_count, num_tokens);
    }
    
    free(pair_counts);
    free(tokens);
    printf("\nDone! Vocab size: %u\n", bpe->vocab_size);
}

uint32_t* encode_bpe(BPE* bpe, const char* text, size_t text_len, uint32_t* num_tokens) {
    if (text_len == 0) {
        *num_tokens = 0;
        return NULL;
    }
    
    // Start with bytes
    uint32_t* tokens = (uint32_t*)malloc(text_len * sizeof(uint32_t));
    *num_tokens = text_len;
    for (size_t i = 0; i < text_len; i++) tokens[i] = (unsigned char)text[i];
    
    // Repeatedly find and merge pairs
    int changed = 1;
    while (changed) {
        changed = 0;
        
        // Try to find any pair that exists in vocab
        for (uint32_t i = 0; i < *num_tokens - 1; i++) {
            uint32_t t1 = tokens[i];
            uint32_t t2 = tokens[i + 1];
            
            // Search vocab for this pair
            for (uint32_t v = 256; v < bpe->vocab_size; v++) {
                // Check if vocab[v] = vocab[t1] + vocab[t2]
                size_t len1 = strlen(bpe->vocab[t1]);
                size_t len2 = strlen(bpe->vocab[t2]);
                size_t vlen = strlen(bpe->vocab[v]);
                
                if (len1 + len2 == vlen) {
                    if (memcmp(bpe->vocab[v], bpe->vocab[t1], len1) == 0 &&
                        memcmp(bpe->vocab[v] + len1, bpe->vocab[t2], len2) == 0) {
                        // Found it! Merge this one occurrence
                        tokens[i] = v;
                        // Shift rest left
                        for (uint32_t j = i + 1; j < *num_tokens - 1; j++) {
                            tokens[j] = tokens[j + 1];
                        }
                        (*num_tokens)--;
                        changed = 1;
                        break;
                    }
                }
            }
            if (changed) break;  // Start over from beginning
        }
    }
    
    return tokens;
}

char* decode_bpe(BPE* bpe, const uint32_t* tokens, uint32_t num_tokens) {
    if (num_tokens == 0) {
        char* empty = (char*)malloc(1);
        empty[0] = '\0';
        return empty;
    }
    
    // Calculate total length
    size_t total_len = 0;
    for (uint32_t i = 0; i < num_tokens; i++) {
        total_len += strlen(bpe->vocab[tokens[i]]);
    }
    
    // Concatenate all vocab entries
    char* text = (char*)malloc(total_len + 1);
    text[0] = '\0';
    for (uint32_t i = 0; i < num_tokens; i++) {
        strcat(text, bpe->vocab[tokens[i]]);
    }
    
    return text;
}