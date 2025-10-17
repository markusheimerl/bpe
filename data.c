#include "data.h"

#define MAX_CORPUS_SIZE (1024ULL * 1024ULL * 2ULL)

// Load the text corpus from a file
char* load_corpus(const char* filename, size_t* corpus_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open corpus file: %s\n", filename);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size == 0) {
        printf("Error: Corpus file is empty: %s\n", filename);
        fclose(file);
        return NULL;
    }

    // Cap the corpus size
    *corpus_size = file_size;
    if (*corpus_size > MAX_CORPUS_SIZE) {
        printf("Warning: Corpus file is %zu bytes, capping at %llu bytes\n", 
               file_size, (unsigned long long)MAX_CORPUS_SIZE);
        *corpus_size = MAX_CORPUS_SIZE;
    }

    char* corpus = (char*)malloc((*corpus_size + 1) * sizeof(char));
    if (!corpus) {
        printf("Error: Could not allocate memory for corpus (%zu bytes)\n", *corpus_size);
        fclose(file);
        return NULL;
    }

    size_t read_size = fread(corpus, 1, *corpus_size, file);
    corpus[read_size] = '\0';
    *corpus_size = read_size;

    fclose(file);
    printf("Loaded corpus: %zu characters", *corpus_size);
    if (file_size > MAX_CORPUS_SIZE) {
        printf(" (truncated from %zu)", file_size);
    }
    printf("\n");
    
    return corpus;
}