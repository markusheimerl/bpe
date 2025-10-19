#include "bpe.h"

typedef struct {
    uint32_t token1;
    uint32_t token2;
    uint32_t count;
} PairCount;

// Count all adjacent pairs in the token sequence
static PairCount* count_pairs(uint32_t* tokens, uint32_t num_tokens, uint32_t* num_pairs) {
    // Allocate maximum possible pairs
    PairCount* pairs = (PairCount*)malloc(num_tokens * sizeof(PairCount));
    *num_pairs = 0;
    
    for (uint32_t i = 0; i < num_tokens - 1; i++) {
        uint32_t t1 = tokens[i];
        uint32_t t2 = tokens[i + 1];
        
        // Check if this pair already exists
        int found = 0;
        for (uint32_t j = 0; j < *num_pairs; j++) {
            if (pairs[j].token1 == t1 && pairs[j].token2 == t2) {
                pairs[j].count++;
                found = 1;
                break;
            }
        }
        
        // Add new pair
        if (!found) {
            pairs[*num_pairs].token1 = t1;
            pairs[*num_pairs].token2 = t2;
            pairs[*num_pairs].count = 1;
            (*num_pairs)++;
        }
    }
    
    return pairs;
}

// Find the most frequent pair
static PairCount find_best_pair(PairCount* pairs, uint32_t num_pairs) {
    PairCount best = {0, 0, 0};
    
    for (uint32_t i = 0; i < num_pairs; i++) {
        if (pairs[i].count > best.count) best = pairs[i];
    }
    
    return best;
}

// Merge a pair in the token sequence
static uint32_t merge_pair(uint32_t* tokens, uint32_t num_tokens, 
                           uint32_t token1, uint32_t token2, uint32_t new_token) {
    uint32_t write_idx = 0;
    uint32_t read_idx = 0;
    
    while (read_idx < num_tokens) {
        if (read_idx < num_tokens - 1 && 
            tokens[read_idx] == token1 && 
            tokens[read_idx + 1] == token2) {
            tokens[write_idx++] = new_token;
            read_idx += 2;
        } else {
            tokens[write_idx++] = tokens[read_idx++];
        }
    }
    
    return write_idx;
}

BPE* init_bpe(uint32_t target_vocab_size) {
    BPE* bpe = (BPE*)malloc(sizeof(BPE));
    
    bpe->vocab_size = INITIAL_VOCAB_SIZE;
    bpe->num_merges = 0;
    
    uint32_t max_size = target_vocab_size;
    bpe->merges = (Merge*)malloc(max_size * sizeof(Merge));
    bpe->vocab = (char**)malloc(max_size * sizeof(char*));
    bpe->vocab_lens = (uint32_t*)malloc(max_size * sizeof(uint32_t));
    
    // Initialize base vocabulary (all bytes)
    for (uint32_t i = 0; i < INITIAL_VOCAB_SIZE; i++) {
        bpe->vocab[i] = (char*)malloc(2);
        bpe->vocab[i][0] = (char)i;
        bpe->vocab[i][1] = '\0';
        bpe->vocab_lens[i] = 1;
    }
    
    return bpe;
}

void free_bpe(BPE* bpe) {
    if (!bpe) return;
    
    for (uint32_t i = 0; i < bpe->vocab_size; i++) free(bpe->vocab[i]);
    free(bpe->vocab);
    free(bpe->vocab_lens);
    free(bpe->merges);
    free(bpe);
}

void train_bpe(BPE* bpe, const char* corpus, size_t corpus_size) {
    printf("\n=== Training BPE ===\n");
    printf("Corpus: %zu bytes\n", corpus_size);
    
    // Initialize token sequence
    uint32_t* tokens = (uint32_t*)malloc(corpus_size * sizeof(uint32_t));
    uint32_t num_tokens = corpus_size;
    
    for (size_t i = 0; i < corpus_size; i++) {
        tokens[i] = (unsigned char)corpus[i];
    }
    
    // Keep merging until we can't find any more pairs
    while (1) {
        // Count all pairs
        uint32_t num_pairs;
        PairCount* pairs = count_pairs(tokens, num_tokens, &num_pairs);
        
        if (num_pairs == 0) {
            free(pairs);
            break;
        }
        
        // Find best pair
        PairCount best = find_best_pair(pairs, num_pairs);
        free(pairs);
        
        if (best.count == 0) break;
        
        uint32_t new_token = bpe->vocab_size;
        
        // Record merge
        bpe->merges[bpe->num_merges].token1 = best.token1;
        bpe->merges[bpe->num_merges].token2 = best.token2;
        bpe->num_merges++;
        
        // Create new vocab entry
        uint32_t new_len = bpe->vocab_lens[best.token1] + bpe->vocab_lens[best.token2];
        bpe->vocab[new_token] = (char*)malloc(new_len + 1);
        memcpy(bpe->vocab[new_token], bpe->vocab[best.token1], bpe->vocab_lens[best.token1]);
        memcpy(bpe->vocab[new_token] + bpe->vocab_lens[best.token1], bpe->vocab[best.token2], bpe->vocab_lens[best.token2]);
        bpe->vocab[new_token][new_len] = '\0';
        bpe->vocab_lens[new_token] = new_len;
        bpe->vocab_size++;
        
        // Apply merge
        num_tokens = merge_pair(tokens, num_tokens, best.token1, best.token2, new_token);
        
        printf("Step %4u: (%u, %u) -> %u | count: %u | tokens: %u\n", bpe->num_merges, best.token1, best.token2, new_token, best.count, num_tokens);
    }
    
    free(tokens);
    printf("\nDone! Vocab: %u | Merges: %u\n", bpe->vocab_size, bpe->num_merges);
}

uint32_t* encode_bpe(BPE* bpe, const char* text, size_t text_len, uint32_t* num_tokens) {
    if (text_len == 0) {
        *num_tokens = 0;
        return NULL;
    }
    
    uint32_t* tokens = (uint32_t*)malloc(text_len * sizeof(uint32_t));
    *num_tokens = text_len;
    
    // Initialize with bytes
    for (size_t i = 0; i < text_len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    
    // Apply all merge rules
    for (uint32_t i = 0; i < bpe->num_merges; i++) {
        uint32_t token1 = bpe->merges[i].token1;
        uint32_t token2 = bpe->merges[i].token2;
        uint32_t new_token = INITIAL_VOCAB_SIZE + i;
        
        *num_tokens = merge_pair(tokens, *num_tokens, token1, token2, new_token);
    }
    
    return tokens;
}

char* decode_bpe(BPE* bpe, const uint32_t* tokens, uint32_t num_tokens) {
    if (num_tokens == 0) {
        char* empty = (char*)malloc(1);
        empty[0] = '\0';
        return empty;
    }
    
    // Calculate length
    size_t total_len = 0;
    for (uint32_t i = 0; i < num_tokens; i++) {
        total_len += bpe->vocab_lens[tokens[i]];
    }
    
    // Build string
    char* text = (char*)malloc(total_len + 1);
    size_t pos = 0;
    
    for (uint32_t i = 0; i < num_tokens; i++) {
        memcpy(text + pos, bpe->vocab[tokens[i]], bpe->vocab_lens[tokens[i]]);
        pos += bpe->vocab_lens[tokens[i]];
    }
    text[pos] = '\0';
    
    return text;
}