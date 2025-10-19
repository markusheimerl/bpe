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
// Fixed: claim position i first, then i+1, with proper rollback
__global__ void replace_pair_kernel(uint32_t* tokens, size_t num_tokens, 
                                    uint32_t t1, uint32_t t2, uint32_t new_token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < num_tokens - 1; i += stride) {
        if (tokens[i] == t1 && tokens[i + 1] == t2) {
            // First, atomically claim position i by replacing t1 with new_token
            uint32_t old_i = atomicCAS(&tokens[i], t1, new_token);
            if (old_i == t1) {
                // Successfully claimed position i, now try to invalidate position i+1
                uint32_t old_i1 = atomicCAS(&tokens[i + 1], t2, INVALID_TOKEN);
                if (old_i1 != t2) {
                    // Failed to claim i+1 (another thread got it first), rollback
                    atomicCAS(&tokens[i], new_token, t1);
                }
            }
        }
    }
}

BPE* init_bpe() {
    BPE* bpe = (BPE*)malloc(sizeof(BPE));
    bpe->vocab_size = 256;
    bpe->vocab = (char**)malloc(256 * sizeof(char*));
    bpe->merge_t1 = NULL;
    bpe->merge_t2 = NULL;
    bpe->num_merges = 0;
    
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
    if (bpe->merge_t1) free(bpe->merge_t1);
    if (bpe->merge_t2) free(bpe->merge_t2);
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
    
    // Allocate merge rule arrays
    bpe->merge_t1 = (uint32_t*)malloc(num_merges * sizeof(uint32_t));
    bpe->merge_t2 = (uint32_t*)malloc(num_merges * sizeof(uint32_t));
    
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
        
        // Record this merge rule
        bpe->merge_t1[merge_iter] = best_t1;
        bpe->merge_t2[merge_iter] = best_t2;
        bpe->num_merges = merge_iter + 1;
        
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
    
    // Allocate GPU memory
    uint32_t* d_tokens;
    CHECK_CUDA(cudaMalloc(&d_tokens, text_len * sizeof(uint32_t)));
    
    // Initialize token sequence as bytes on host and copy to device
    uint32_t* h_tokens = (uint32_t*)malloc(text_len * sizeof(uint32_t));
    for (size_t i = 0; i < text_len; i++) h_tokens[i] = (unsigned char)text[i];
    CHECK_CUDA(cudaMemcpy(d_tokens, h_tokens, text_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int num_blocks = (text_len + block_size - 1) / block_size;
    
    // Apply all learned merges in the order they were learned
    for (uint32_t merge_idx = 0; merge_idx < bpe->num_merges; merge_idx++) {
        uint32_t t1 = bpe->merge_t1[merge_idx];
        uint32_t t2 = bpe->merge_t2[merge_idx];
        uint32_t new_token = 256 + merge_idx;
        
        replace_pair_kernel<<<num_blocks, block_size>>>(d_tokens, text_len, t1, t2, new_token);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Copy the token array back to host
    CHECK_CUDA(cudaMemcpy(h_tokens, d_tokens, text_len * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Compact the array by removing INVALID_TOKENs
    uint32_t* final_tokens = (uint32_t*)malloc(text_len * sizeof(uint32_t));
    uint32_t count = 0;
    for (size_t i = 0; i < text_len; i++) {
        if (h_tokens[i] != INVALID_TOKEN) final_tokens[count++] = h_tokens[i];
    }
    *num_tokens = count;
    
    // Cleanup
    free(h_tokens);
    CHECK_CUDA(cudaFree(d_tokens));
    
    return final_tokens;
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
    char* ptr = text;
    for (uint32_t i = 0; i < num_tokens; i++) {
        size_t len = strlen(bpe->vocab[tokens[i]]);
        memcpy(ptr, bpe->vocab[tokens[i]], len);
        ptr += len;
    }
    *ptr = '\0';
    
    return text;
}