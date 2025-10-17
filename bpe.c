#include "bpe.h"

#define HASH_SIZE 1048576

typedef struct HashNode {
    uint64_t pair;
    uint32_t count;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode** buckets;
    uint32_t size;
} HashTable;

// ============================================================================
// Hash Table Implementation
// ============================================================================

static HashTable* hash_create() {
    HashTable* ht = (HashTable*)malloc(sizeof(HashTable));
    ht->size = HASH_SIZE;
    ht->buckets = (HashNode **)calloc(HASH_SIZE, sizeof(HashNode*));
    return ht;
}

static void hash_free(HashTable* ht) {
    for (uint32_t i = 0; i < ht->size; i++) {
        HashNode* node = ht->buckets[i];
        while (node) {
            HashNode* next = node->next;
            free(node);
            node = next;
        }
    }
    free(ht->buckets);
    free(ht);
}

static uint32_t hash_function(uint64_t pair) {
    pair ^= pair >> 33;
    pair *= 0xff51afd7ed558ccdULL;
    pair ^= pair >> 33;
    pair *= 0xc4ceb9fe1a85ec53ULL;
    pair ^= pair >> 33;
    return pair % HASH_SIZE;
}

static void hash_increment(HashTable* ht, uint64_t pair) {
    uint32_t idx = hash_function(pair);
    HashNode* node = ht->buckets[idx];
    
    while (node) {
        if (node->pair == pair) {
            node->count++;
            return;
        }
        node = node->next;
    }
    
    HashNode* new_node = (HashNode*)malloc(sizeof(HashNode));
    new_node->pair = pair;
    new_node->count = 1;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
}

static uint64_t make_pair(uint32_t token1, uint32_t token2) {
    return ((uint64_t)token1 << 32) | token2;
}

static void count_pairs(uint32_t* tokens, uint32_t num_tokens, HashTable* ht) {
    for (uint32_t i = 0; i < num_tokens - 1; i++) {
        uint64_t pair = make_pair(tokens[i], tokens[i + 1]);
        hash_increment(ht, pair);
    }
}

static uint64_t find_most_frequent_pair(HashTable* ht, uint32_t* max_count) {
    uint64_t best_pair = 0;
    *max_count = 0;
    
    for (uint32_t i = 0; i < ht->size; i++) {
        HashNode* node = ht->buckets[i];
        while (node) {
            if (node->count > *max_count) {
                *max_count = node->count;
                best_pair = node->pair;
            }
            node = node->next;
        }
    }
    
    return best_pair;
}

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

// ============================================================================
// BPE Core Functions
// ============================================================================

BPE* init_bpe(uint32_t target_vocab_size) {
    BPE* bpe = (BPE*)malloc(sizeof(BPE));
    
    bpe->vocab_size = INITIAL_VOCAB_SIZE;
    bpe->num_merges = 0;
    bpe->target_vocab_size = target_vocab_size;
    bpe->training_step = 0;
    
    bpe->merges = (Merge *)malloc(MAX_VOCAB_SIZE * sizeof(Merge));
    bpe->vocab = (char **)malloc(MAX_VOCAB_SIZE * sizeof(char*));
    bpe->vocab_lens = (uint32_t *)malloc(MAX_VOCAB_SIZE * sizeof(uint32_t));
    
    // Initialize base vocabulary (all bytes)
    for (uint32_t i = 0; i < INITIAL_VOCAB_SIZE; i++) {
        bpe->vocab[i] = (char *)malloc(2);
        bpe->vocab[i][0] = (char)i;
        bpe->vocab[i][1] = '\0';
        bpe->vocab_lens[i] = 1;
    }
    
    return bpe;
}

void free_bpe(BPE* bpe) {
    if (!bpe) return;
    
    for (uint32_t i = 0; i < bpe->vocab_size; i++) {
        free(bpe->vocab[i]);
    }
    free(bpe->vocab);
    free(bpe->vocab_lens);
    free(bpe->merges);
    free(bpe);
}

void train_bpe(BPE* bpe, const char* corpus, size_t corpus_size) {
    printf("\n=== Training BPE Tokenizer ===\n");
    printf("Corpus size: %zu bytes\n", corpus_size);
    printf("Initial vocab size: %u\n", bpe->vocab_size);
    printf("Target vocab size: %u\n", bpe->target_vocab_size);
    printf("Merges to perform: %u\n\n", bpe->target_vocab_size - INITIAL_VOCAB_SIZE);
    
    // Initialize token sequence
    uint32_t* tokens = (uint32_t *)malloc(corpus_size * sizeof(uint32_t));
    uint32_t num_tokens = corpus_size;
    
    for (size_t i = 0; i < corpus_size; i++) {
        tokens[i] = (unsigned char)corpus[i];
    }
    
    uint32_t num_merges = bpe->target_vocab_size - INITIAL_VOCAB_SIZE;
    
    for (uint32_t merge_idx = 0; merge_idx < num_merges; merge_idx++) {
        // Count all pairs
        HashTable* ht = hash_create();
        count_pairs(tokens, num_tokens, ht);
        
        // Find most frequent pair
        uint32_t max_count;
        uint64_t best_pair = find_most_frequent_pair(ht, &max_count);
        hash_free(ht);
        
        if (max_count == 0) {
            printf("No more pairs to merge at step %u\n", merge_idx);
            break;
        }
        
        uint32_t token1 = (uint32_t)(best_pair >> 32);
        uint32_t token2 = (uint32_t)(best_pair & 0xFFFFFFFF);
        uint32_t new_token = bpe->vocab_size;
        
        // Record merge rule
        bpe->merges[bpe->num_merges].token1 = token1;
        bpe->merges[bpe->num_merges].token2 = token2;
        bpe->num_merges++;
        
        // Create new vocabulary entry
        uint32_t new_len = bpe->vocab_lens[token1] + bpe->vocab_lens[token2];
        bpe->vocab[new_token] = (char *)malloc(new_len + 1);
        memcpy(bpe->vocab[new_token], bpe->vocab[token1], bpe->vocab_lens[token1]);
        memcpy(bpe->vocab[new_token] + bpe->vocab_lens[token1], 
               bpe->vocab[token2], bpe->vocab_lens[token2]);
        bpe->vocab[new_token][new_len] = '\0';
        bpe->vocab_lens[new_token] = new_len;
        bpe->vocab_size++;
        
        // Apply merge
        num_tokens = merge_pair(tokens, num_tokens, token1, token2, new_token);
        
        bpe->training_step++;
        
        // Print progress
        printf("Step [%4u/%4u]: (%5u, %5u) -> %5u | count: %6u | tokens: %6u\n", 
                   merge_idx + 1, num_merges, token1, token2, new_token, max_count, num_tokens);
    }
    
    free(tokens);
    
    printf("\nTraining complete!\n");
    printf("Final vocabulary size: %u\n", bpe->vocab_size);
    printf("Total merge rules: %u\n", bpe->num_merges);
}

uint32_t* encode_bpe(BPE* bpe, const char* text, size_t text_len, uint32_t* num_tokens) {
    if (text_len == 0) {
        *num_tokens = 0;
        return NULL;
    }
    
    uint32_t* tokens = (uint32_t*)malloc(text_len * sizeof(uint32_t));
    *num_tokens = text_len;
    
    // Initialize with byte tokens
    for (size_t i = 0; i < text_len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    
    // Apply all merge rules in order
    for (uint32_t i = 0; i < bpe->num_merges; i++) {
        uint32_t token1 = bpe->merges[i].token1;
        uint32_t token2 = bpe->merges[i].token2;
        uint32_t new_token = INITIAL_VOCAB_SIZE + i;
        
        *num_tokens = merge_pair(tokens, *num_tokens, token1, token2, new_token);
        
        printf("  Merge [%5u/%5u]: (%5u, %5u) -> %5u | tokens: %6u\n", 
            i + 1, bpe->num_merges, token1, token2, new_token, *num_tokens);
    }
    
    return tokens;
}

char* decode_bpe(BPE* bpe, const uint32_t* tokens, uint32_t num_tokens) {
    if (num_tokens == 0) {
        char* empty = (char *)malloc(1);
        empty[0] = '\0';
        return empty;
    }
    
    // Calculate total length
    size_t total_len = 0;
    for (uint32_t i = 0; i < num_tokens; i++) {
        if (tokens[i] < bpe->vocab_size) {
            total_len += bpe->vocab_lens[tokens[i]];
        }
    }
    
    // Build output string
    char* text = (char*)malloc(total_len + 1);
    size_t pos = 0;
    
    for (uint32_t i = 0; i < num_tokens; i++) {
        if (tokens[i] < bpe->vocab_size) {
            memcpy(text + pos, bpe->vocab[tokens[i]], bpe->vocab_lens[tokens[i]]);
            pos += bpe->vocab_lens[tokens[i]];
        }
    }
    text[pos] = '\0';
    
    return text;
}

void save_bpe(BPE* bpe, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file for writing: %s\n", filename);
        return;
    }
    
    // Save metadata
    fwrite(&bpe->vocab_size, sizeof(uint32_t), 1, file);
    fwrite(&bpe->num_merges, sizeof(uint32_t), 1, file);
    fwrite(&bpe->target_vocab_size, sizeof(uint32_t), 1, file);
    fwrite(&bpe->training_step, sizeof(int), 1, file);
    
    // Save merge rules
    fwrite(bpe->merges, sizeof(Merge), bpe->num_merges, file);
    
    // Save vocabulary
    for (uint32_t i = 0; i < bpe->vocab_size; i++) {
        fwrite(&bpe->vocab_lens[i], sizeof(uint32_t), 1, file);
        fwrite(bpe->vocab[i], 1, bpe->vocab_lens[i], file);
    }
    
    fclose(file);
    printf("Tokenizer saved to %s\n", filename);
}

BPE* load_bpe(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read metadata
    uint32_t vocab_size, num_merges, target_vocab_size;
    int training_step;
    fread(&vocab_size, sizeof(uint32_t), 1, file);
    fread(&num_merges, sizeof(uint32_t), 1, file);
    fread(&target_vocab_size, sizeof(uint32_t), 1, file);
    fread(&training_step, sizeof(int), 1, file);
    
    // Initialize BPE structure
    BPE* bpe = (BPE*)malloc(sizeof(BPE));
    bpe->vocab_size = vocab_size;
    bpe->num_merges = num_merges;
    bpe->target_vocab_size = target_vocab_size;
    bpe->training_step = training_step;
    
    bpe->merges = (Merge *)malloc(MAX_VOCAB_SIZE * sizeof(Merge));
    bpe->vocab = (char **)malloc(MAX_VOCAB_SIZE * sizeof(char*));
    bpe->vocab_lens = (uint32_t *)malloc(MAX_VOCAB_SIZE * sizeof(uint32_t));
    
    // Read merge rules
    fread(bpe->merges, sizeof(Merge), bpe->num_merges, file);
    
    // Read vocabulary
    for (uint32_t i = 0; i < bpe->vocab_size; i++) {
        fread(&bpe->vocab_lens[i], sizeof(uint32_t), 1, file);
        bpe->vocab[i] = (char *)malloc(bpe->vocab_lens[i] + 1);
        fread(bpe->vocab[i], 1, bpe->vocab_lens[i], file);
        bpe->vocab[i][bpe->vocab_lens[i]] = '\0';
    }
    
    fclose(file);
    printf("Tokenizer loaded from %s\n", filename);
    printf("  Vocab size: %u | Merge rules: %u | Training step: %d\n", 
           bpe->vocab_size, bpe->num_merges, bpe->training_step);
    
    return bpe;
}

void print_vocab_bpe(BPE* bpe, uint32_t max_entries) {
    printf("\n=== Vocabulary Sample ===\n");
    uint32_t start = INITIAL_VOCAB_SIZE;
    uint32_t limit = start + max_entries;
    if (limit > bpe->vocab_size) limit = bpe->vocab_size;
    
    for (uint32_t i = start; i < limit; i++) {
        printf("Token %5u (len %2u): \"", i, bpe->vocab_lens[i]);
        for (uint32_t j = 0; j < bpe->vocab_lens[i]; j++) {
            unsigned char c = bpe->vocab[i][j];
            if (c >= 32 && c < 127) {
                printf("%c", c);
            } else {
                printf("\\x%02x", c);
            }
        }
        printf("\"\n");
    }
    
    if (bpe->vocab_size > limit) {
        printf("... (%u more entries)\n", bpe->vocab_size - limit);
    }
}

void print_stats_bpe(BPE* bpe, const char* corpus, size_t corpus_size) {
    printf("\n=== Tokenization Statistics ===\n");
    
    uint32_t num_tokens;
    uint32_t* tokens = encode_bpe(bpe, corpus, corpus_size, &num_tokens);
    
    printf("Original size:      %zu bytes\n", corpus_size);
    printf("Tokenized size:     %u tokens\n", num_tokens);
    printf("Compression ratio:  %.2f%%\n", 100.0 * (1.0 - (double)num_tokens / corpus_size));
    printf("Bytes per token:    %.2f\n", (double)corpus_size / num_tokens);
    printf("Vocabulary size:    %u\n", bpe->vocab_size);
    
    free(tokens);
}