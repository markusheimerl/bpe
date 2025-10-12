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

static HashTable* hash_create() {
    HashTable* ht = malloc(sizeof(HashTable));
    ht->size = HASH_SIZE;
    ht->buckets = calloc(HASH_SIZE, sizeof(HashNode*));
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
    
    HashNode* new_node = malloc(sizeof(HashNode));
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

BPETokenizer* bpe_create() {
    BPETokenizer* tokenizer = malloc(sizeof(BPETokenizer));
    tokenizer->vocab_size = INITIAL_VOCAB_SIZE;
    tokenizer->num_merges = 0;
    tokenizer->merges = malloc(MAX_VOCAB_SIZE * sizeof(Merge));
    tokenizer->vocab = malloc(MAX_VOCAB_SIZE * sizeof(char*));
    tokenizer->vocab_lens = malloc(MAX_VOCAB_SIZE * sizeof(uint32_t));
    
    for (uint32_t i = 0; i < INITIAL_VOCAB_SIZE; i++) {
        tokenizer->vocab[i] = malloc(2);
        tokenizer->vocab[i][0] = (char)i;
        tokenizer->vocab[i][1] = '\0';
        tokenizer->vocab_lens[i] = 1;
    }
    
    return tokenizer;
}

void bpe_free(BPETokenizer* tokenizer) {
    if (!tokenizer) return;
    
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->vocab_lens);
    free(tokenizer->merges);
    free(tokenizer);
}

void bpe_train(BPETokenizer* tokenizer, const char* corpus, size_t corpus_size, uint32_t num_merges) {
    printf("\n=== Training BPE Tokenizer ===\n");
    printf("Corpus size: %zu bytes\n", corpus_size);
    printf("Target merges: %u\n", num_merges);
    printf("Initial vocab size: %u\n\n", tokenizer->vocab_size);
    
    uint32_t* tokens = malloc(corpus_size * sizeof(uint32_t));
    uint32_t num_tokens = corpus_size;
    
    for (size_t i = 0; i < corpus_size; i++) {
        tokens[i] = (unsigned char)corpus[i];
    }
    
    for (uint32_t merge_idx = 0; merge_idx < num_merges; merge_idx++) {
        HashTable* ht = hash_create();
        count_pairs(tokens, num_tokens, ht);
        
        uint32_t max_count;
        uint64_t best_pair = find_most_frequent_pair(ht, &max_count);
        hash_free(ht);
        
        if (max_count == 0) {
            printf("No more pairs to merge at iteration %u\n", merge_idx);
            break;
        }
        
        uint32_t token1 = (uint32_t)(best_pair >> 32);
        uint32_t token2 = (uint32_t)(best_pair & 0xFFFFFFFF);
        uint32_t new_token = tokenizer->vocab_size;
        
        tokenizer->merges[tokenizer->num_merges].token1 = token1;
        tokenizer->merges[tokenizer->num_merges].token2 = token2;
        tokenizer->num_merges++;
        
        uint32_t new_len = tokenizer->vocab_lens[token1] + tokenizer->vocab_lens[token2];
        tokenizer->vocab[new_token] = malloc(new_len + 1);
        memcpy(tokenizer->vocab[new_token], tokenizer->vocab[token1], tokenizer->vocab_lens[token1]);
        memcpy(tokenizer->vocab[new_token] + tokenizer->vocab_lens[token1], 
               tokenizer->vocab[token2], tokenizer->vocab_lens[token2]);
        tokenizer->vocab[new_token][new_len] = '\0';
        tokenizer->vocab_lens[new_token] = new_len;
        tokenizer->vocab_size++;
        
        num_tokens = merge_pair(tokens, num_tokens, token1, token2, new_token);
        
        if ((merge_idx + 1) % 100 == 0 || merge_idx < 10 || merge_idx == num_merges - 1) {
            printf("Merge %4u: (%5u, %5u) -> %5u | count: %6u | tokens: %6u\n", 
                   merge_idx + 1, token1, token2, new_token, max_count, num_tokens);
        }
    }
    
    free(tokens);
    printf("\nTraining complete!\n");
    printf("Final vocabulary size: %u\n", tokenizer->vocab_size);
}

uint32_t* bpe_encode(BPETokenizer* tokenizer, const char* text, size_t text_len, uint32_t* num_tokens) {
    uint32_t* tokens = malloc(text_len * sizeof(uint32_t));
    *num_tokens = text_len;
    
    for (size_t i = 0; i < text_len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    
    for (uint32_t i = 0; i < tokenizer->num_merges; i++) {
        uint32_t token1 = tokenizer->merges[i].token1;
        uint32_t token2 = tokenizer->merges[i].token2;
        uint32_t new_token = INITIAL_VOCAB_SIZE + i;
        
        *num_tokens = merge_pair(tokens, *num_tokens, token1, token2, new_token);
    }
    
    return tokens;
}

char* bpe_decode(BPETokenizer* tokenizer, const uint32_t* tokens, uint32_t num_tokens) {
    size_t total_len = 0;
    for (uint32_t i = 0; i < num_tokens; i++) {
        if (tokens[i] < tokenizer->vocab_size) {
            total_len += tokenizer->vocab_lens[tokens[i]];
        }
    }
    
    char* text = malloc(total_len + 1);
    size_t pos = 0;
    
    for (uint32_t i = 0; i < num_tokens; i++) {
        if (tokens[i] < tokenizer->vocab_size) {
            memcpy(text + pos, tokenizer->vocab[tokens[i]], tokenizer->vocab_lens[tokens[i]]);
            pos += tokenizer->vocab_lens[tokens[i]];
        }
    }
    text[pos] = '\0';
    
    return text;
}

int bpe_save(BPETokenizer* tokenizer, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file for writing: %s\n", filename);
        return -1;
    }
    
    fwrite(&tokenizer->vocab_size, sizeof(uint32_t), 1, f);
    fwrite(&tokenizer->num_merges, sizeof(uint32_t), 1, f);
    fwrite(tokenizer->merges, sizeof(Merge), tokenizer->num_merges, f);
    
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        fwrite(&tokenizer->vocab_lens[i], sizeof(uint32_t), 1, f);
        fwrite(tokenizer->vocab[i], 1, tokenizer->vocab_lens[i], f);
    }
    
    fclose(f);
    printf("\n✓ Tokenizer saved to: %s\n", filename);
    return 0;
}

BPETokenizer* bpe_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Could not open file for reading: %s\n", filename);
        return NULL;
    }
    
    BPETokenizer* tokenizer = malloc(sizeof(BPETokenizer));
    tokenizer->merges = malloc(MAX_VOCAB_SIZE * sizeof(Merge));
    tokenizer->vocab = malloc(MAX_VOCAB_SIZE * sizeof(char*));
    tokenizer->vocab_lens = malloc(MAX_VOCAB_SIZE * sizeof(uint32_t));
    
    fread(&tokenizer->vocab_size, sizeof(uint32_t), 1, f);
    fread(&tokenizer->num_merges, sizeof(uint32_t), 1, f);
    fread(tokenizer->merges, sizeof(Merge), tokenizer->num_merges, f);
    
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        fread(&tokenizer->vocab_lens[i], sizeof(uint32_t), 1, f);
        tokenizer->vocab[i] = malloc(tokenizer->vocab_lens[i] + 1);
        fread(tokenizer->vocab[i], 1, tokenizer->vocab_lens[i], f);
        tokenizer->vocab[i][tokenizer->vocab_lens[i]] = '\0';
    }
    
    fclose(f);
    printf("✓ Tokenizer loaded from: %s\n", filename);
    printf("  Vocab size: %u | Merges: %u\n", tokenizer->vocab_size, tokenizer->num_merges);
    return tokenizer;
}

void bpe_print_vocab(BPETokenizer* tokenizer, uint32_t max_entries) {
    printf("\n=== Vocabulary ===\n");
    uint32_t limit = max_entries < tokenizer->vocab_size ? max_entries : tokenizer->vocab_size;
    
    for (uint32_t i = INITIAL_VOCAB_SIZE; i < limit && i < tokenizer->vocab_size; i++) {
        printf("Token %5u (len %2u): \"", i, tokenizer->vocab_lens[i]);
        for (uint32_t j = 0; j < tokenizer->vocab_lens[i]; j++) {
            unsigned char c = tokenizer->vocab[i][j];
            if (c >= 32 && c < 127) {
                printf("%c", c);
            } else {
                printf("\\x%02x", c);
            }
        }
        printf("\"\n");
    }
    if (tokenizer->vocab_size > limit) {
        printf("... (%u more entries)\n", tokenizer->vocab_size - limit);
    }
}