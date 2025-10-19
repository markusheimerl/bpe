#include "bpe.h"

#define INVALID_TOKEN UINT32_MAX

// CUDA kernel to count all adjacent token pairs
__global__ void count_pairs_kernel(const uint32_t* tokens, uint32_t* pair_counts, 
                                   size_t num_tokens, uint32_t max_vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < num_tokens - 1; i += stride) {
        uint32_t t1 = tokens[i];
        uint32_t t2 = tokens[i + 1];

        if (t1 != INVALID_TOKEN && t2 != INVALID_TOKEN) {
            atomicAdd(&pair_counts[t1 * max_vocab_size + t2], 1);
        }
    }
}

// CUDA kernel to replace a specific token pair with a new token
__global__ void replace_pair_kernel(uint32_t* tokens, size_t num_tokens, 
                                    uint32_t t1, uint32_t t2, uint32_t new_token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < num_tokens - 1; i += stride) {
        if (tokens[i] == t1 && tokens[i + 1] == t2) {
            tokens[i] = new_token;
            tokens[i + 1] = INVALID_TOKEN;
        }
    }
}

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
    
    // Initialize token sequence as bytes on host
    uint32_t* h_tokens = (uint32_t*)malloc(corpus_size * sizeof(uint32_t));
    for (size_t i = 0; i < corpus_size; i++) {
        h_tokens[i] = (unsigned char)corpus[i];
    }
    
    // Allocate GPU memory
    uint32_t* d_tokens;
    uint32_t* d_pair_counts;
    uint32_t num_merges = max_vocab_size - 256;
    size_t counts_size = (size_t)max_vocab_size * max_vocab_size;
    
    CHECK_CUDA(cudaMalloc(&d_tokens, corpus_size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_pair_counts, counts_size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_tokens, h_tokens, corpus_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    uint32_t* h_pair_counts = (uint32_t*)malloc(counts_size * sizeof(uint32_t));
    
    int block_size = 256;
    int num_blocks = (corpus_size + block_size - 1) / block_size;
    
    // Do num_merges iterations
    for (uint32_t merge_iter = 0; merge_iter < num_merges; merge_iter++) {
        // Reset pair counts
        CHECK_CUDA(cudaMemset(d_pair_counts, 0, counts_size * sizeof(uint32_t)));
        
        // Count pairs on GPU
        count_pairs_kernel<<<num_blocks, block_size>>>(d_tokens, d_pair_counts, corpus_size, max_vocab_size);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy counts to host
        CHECK_CUDA(cudaMemcpy(h_pair_counts, d_pair_counts, counts_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        
        // Find most frequent pair
        uint32_t best_t1 = 0, best_t2 = 0, best_count = 0;
        for (uint32_t t1 = 0; t1 < bpe->vocab_size; t1++) {
            for (uint32_t t2 = 0; t2 < bpe->vocab_size; t2++) {
                uint32_t count = h_pair_counts[t1 * max_vocab_size + t2];
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
        
        // Replace all occurrences on GPU
        replace_pair_kernel<<<num_blocks, block_size>>>(d_tokens, corpus_size, best_t1, best_t2, new_token);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        printf("Merge %u: (%u, %u) -> %u | count: %u\n", merge_iter + 1, best_t1, best_t2, new_token, best_count);
    }
    
    // Cleanup
    free(h_tokens);
    free(h_pair_counts);
    CHECK_CUDA(cudaFree(d_tokens));
    CHECK_CUDA(cudaFree(d_pair_counts));
    
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