#include "bpe.h"

BPE* init_bpe() {
    BPE* bpe = (BPE*)malloc(sizeof(BPE));
    bpe->vocab_size = 256;
    bpe->vocab_capacity = 256;
    bpe->vocab = (char**)malloc(bpe->vocab_capacity * sizeof(char*));
    
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

void train_bpe(BPE* bpe, const char* corpus, size_t corpus_size, uint32_t num_merges) {
    printf("\n=== Training BPE ===\n");
    
    // Initialize token sequence as bytes
    uint32_t* tokens = (uint32_t*)malloc(corpus_size * sizeof(uint32_t));
    uint32_t num_tokens = corpus_size;
    for (size_t i = 0; i < corpus_size; i++) {
        tokens[i] = (unsigned char)corpus[i];
    }
    
    // Do num_merges iterations
    for (uint32_t merge_iter = 0; merge_iter < num_merges; merge_iter++) {
        // Count all pairs
        uint32_t best_t1 = 0, best_t2 = 0, best_count = 0;
        
        for (uint32_t i = 0; i < num_tokens - 1; i++) {
            uint32_t t1 = tokens[i];
            uint32_t t2 = tokens[i + 1];
            
            // Count how many times this pair appears
            uint32_t count = 0;
            for (uint32_t j = 0; j < num_tokens - 1; j++) {
                if (tokens[j] == t1 && tokens[j + 1] == t2) {
                    count++;
                }
            }
            
            if (count > best_count) {
                best_count = count;
                best_t1 = t1;
                best_t2 = t2;
            }
        }
        
        if (best_count == 0) break;
        
        // Expand vocab if needed
        if (bpe->vocab_size >= bpe->vocab_capacity) {
            bpe->vocab_capacity *= 2;
            bpe->vocab = (char**)realloc(bpe->vocab, bpe->vocab_capacity * sizeof(char*));
        }
        
        uint32_t new_token = bpe->vocab_size;
        
        // Create new vocab entry by concatenating
        size_t len1 = strlen(bpe->vocab[best_t1]);
        size_t len2 = strlen(bpe->vocab[best_t2]);
        bpe->vocab[new_token] = (char*)malloc(len1 + len2 + 1);
        strcpy(bpe->vocab[new_token], bpe->vocab[best_t1]);
        strcat(bpe->vocab[new_token], bpe->vocab[best_t2]);
        bpe->vocab_size++;
        
        // Replace all occurrences of the pair with new token
        uint32_t new_num_tokens = 0;
        for (uint32_t i = 0; i < num_tokens; i++) {
            if (i < num_tokens - 1 && tokens[i] == best_t1 && tokens[i + 1] == best_t2) {
                tokens[new_num_tokens++] = new_token;
                i++;  // Skip next token
            } else {
                tokens[new_num_tokens++] = tokens[i];
            }
        }
        num_tokens = new_num_tokens;
        
        printf("Merge %u: (%u, %u) -> %u | count: %u | tokens: %u\n", 
               merge_iter + 1, best_t1, best_t2, new_token, best_count, num_tokens);
    }
    
    free(tokens);
    printf("\nDone! Vocab size: %u\n", bpe->vocab_size);
}

uint32_t* encode_bpe(BPE* bpe, const char* text, size_t text_len, uint32_t* num_tokens) {
    if (text_len == 0) {
        *num_tokens = 0;
        return NULL;
    }
    
    uint32_t* tokens = (uint32_t*)malloc(text_len * sizeof(uint32_t));
    *num_tokens = text_len;
    
    // Start with bytes
    for (size_t i = 0; i < text_len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    
    // Keep merging until no more merges possible
    int merged = 1;
    while (merged) {
        merged = 0;
        
        // Try all vocab entries from highest to lowest (newest to oldest)
        for (uint32_t v = bpe->vocab_size - 1; v >= 256; v--) {
            // Try to find what pair creates this vocab entry
            for (uint32_t t1 = 0; t1 < v; t1++) {
                for (uint32_t t2 = 0; t2 < v; t2++) {
                    size_t len1 = strlen(bpe->vocab[t1]);
                    size_t len2 = strlen(bpe->vocab[t2]);
                    size_t vlen = strlen(bpe->vocab[v]);
                    
                    if (len1 + len2 != vlen) continue;
                    
                    // Check if concatenating t1 and t2 gives us v
                    char* combined = (char*)malloc(len1 + len2 + 1);
                    strcpy(combined, bpe->vocab[t1]);
                    strcat(combined, bpe->vocab[t2]);
                    
                    if (strcmp(combined, bpe->vocab[v]) == 0) {
                        // Found the pair! Now merge all occurrences
                        uint32_t new_num = 0;
                        for (uint32_t i = 0; i < *num_tokens; i++) {
                            if (i < *num_tokens - 1 && tokens[i] == t1 && tokens[i + 1] == t2) {
                                tokens[new_num++] = v;
                                i++;
                                merged = 1;
                            } else {
                                tokens[new_num++] = tokens[i];
                            }
                        }
                        *num_tokens = new_num;
                    }
                    free(combined);
                }
            }
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