/**
 * GGUF weight file loader with mmap support.
 * Compatible with llama.cpp ecosystem GGUF format v3.
 */

#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ── Internal helpers ─────────────────────────────────────────────────── */

static uint32_t read_u32(const void *ptr) {
    uint32_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

static int64_t read_i64(const void *ptr) {
    int64_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

static float read_f32(const void *ptr) {
    float val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

static double read_f64(const void *ptr) {
    double val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

/* ── Open GGUF file ───────────────────────────────────────────────────── */

gguf_file_t *gguf_open(const char *path, bool use_mmap) {
    gguf_file_t *f = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    if (!f) return NULL;
    
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        free(f);
        fprintf(stderr, "[gguf] Failed to open file: %s\n", path);
        return NULL;
    }
    
    f->file_ptr = (void *)hFile;
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        free(f);
        fprintf(stderr, "[gguf] Failed to get file size\n");
        return NULL;
    }
    f->file_size = (size_t)fileSize.QuadPart;
    
    if (use_mmap && f->file_size > 0) {
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMap) {
            f->mmap_ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
            CloseHandle(hMap);
            if (f->mmap_ptr) {
                f->use_mmap = true;
            }
        }
    }
    
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        free(f);
        fprintf(stderr, "[gguf] Failed to open file: %s\n", path);
        return NULL;
    }
    
    f->file_ptr = (void *)(intptr_t)fd;
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        free(f);
        fprintf(stderr, "[gguf] Failed to stat file\n");
        return NULL;
    }
    f->file_size = (size_t)st.st_size;
    
    if (use_mmap && f->file_size > 0) {
        void *ptr = mmap(NULL, f->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr != MAP_FAILED) {
            f->mmap_ptr = ptr;
            f->use_mmap = true;
        }
    }
#endif
    
    /* Read and validate header */
    const uint8_t *base = (const uint8_t *)f->mmap_ptr ? f->mmap_ptr : NULL;
    
    if (!base) {
#ifdef _WIN32
        DWORD bytesRead;
        uint8_t header[16];
        if (!ReadFile(hFile, header, 16, &bytesRead, NULL)) {
            CloseHandle(hFile);
            free(f);
            return NULL;
        }
        base = header;
#else
        uint8_t header[16];
        if (read(fd, header, 16) != 16) {
            close(fd);
            free(f);
            return NULL;
        }
        base = header;
#endif
    }
    
    f->magic = read_u32(base);
    if (f->magic != GGUF_MAGIC) {
        fprintf(stderr, "[gguf] Invalid magic: expected 0x%08X, got 0x%08X\n", 
                GGUF_MAGIC, f->magic);
#ifdef _WIN32
        CloseHandle((HANDLE)f->file_ptr);
#else
        close((int)(intptr_t)f->file_ptr);
#endif
        if (f->use_mmap && f->mmap_ptr) munmap(f->mmap_ptr, f->file_size);
        free(f);
        return NULL;
    }
    
    f->version = read_u32(base + 4);
    if (f->version != GGUF_VERSION_3) {
        fprintf(stderr, "[gguf] Unsupported version: %u\n", f->version);
#ifdef _WIN32
        CloseHandle((HANDLE)f->file_ptr);
#else
        close((int)(intptr_t)f->file_ptr);
#endif
        if (f->use_mmap && f->mmap_ptr) munmap(f->mmap_ptr, f->file_size);
        free(f);
        return NULL;
    }
    
    /* Read tensor and KV counts */
    const uint8_t *ptr = base + 8;
    f->n_tensors = read_u64(ptr); ptr += 8;
    f->n_kv = read_u64(ptr); ptr += 8;
    
    /* Allocate metadata arrays */
    f->kv_keys = (char **)calloc(f->n_kv, sizeof(char *));
    f->kv_values = (void **)calloc(f->n_kv, sizeof(void *));
    f->kv_types = (gguf_val_type_t *)calloc(f->n_kv, sizeof(gguf_val_type_t));
    if (!f->kv_keys || !f->kv_values || !f->kv_types) {
        fprintf(stderr, "[gguf] Out of memory allocating metadata arrays\n");
#ifdef _WIN32
        CloseHandle((HANDLE)f->file_ptr);
#else
        close((int)(intptr_t)f->file_ptr);
#endif
        if (f->use_mmap && f->mmap_ptr) munmap(f->mmap_ptr, f->file_size);
        free(f->kv_keys);
        free(f->kv_values);
        free(f->kv_types);
        free(f);
        return NULL;
    }
    
    /* Read KV pairs */
    for (uint64_t i = 0; i < f->n_kv; i++) {
        uint32_t key_len = read_u32(ptr); ptr += 4;
        
        f->kv_keys[i] = (char *)malloc(key_len + 1);
        if (!f->kv_keys[i]) continue;
        
        memcpy(f->kv_keys[i], ptr, key_len);
        f->kv_keys[i][key_len] = '\0';
        ptr += key_len;
        
        f->kv_types[i] = (gguf_val_type_t)read_u32(ptr); ptr += 4;
        
        /* Read value based on type */
        switch (f->kv_types[i]) {
            case GGUF_VAL_TYPE_UINT8:
                f->kv_values[i] = malloc(sizeof(uint8_t));
                memcpy(f->kv_values[i], ptr, sizeof(uint8_t));
                ptr += sizeof(uint8_t);
                break;
            case GGUF_VAL_TYPE_INT8:
                f->kv_values[i] = malloc(sizeof(int8_t));
                memcpy(f->kv_values[i], ptr, sizeof(int8_t));
                ptr += sizeof(int8_t);
                break;
            case GGUF_VAL_TYPE_UINT16:
                f->kv_values[i] = malloc(sizeof(uint16_t));
                memcpy(f->kv_values[i], ptr, sizeof(uint16_t));
                ptr += sizeof(uint16_t);
                break;
            case GGUF_VAL_TYPE_INT16:
                f->kv_values[i] = malloc(sizeof(int16_t));
                memcpy(f->kv_values[i], ptr, sizeof(int16_t));
                ptr += sizeof(int16_t);
                break;
            case GGUF_VAL_TYPE_UINT32:
                f->kv_values[i] = malloc(sizeof(uint32_t));
                memcpy(f->kv_values[i], ptr, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                break;
            case GGUF_VAL_TYPE_INT32:
                f->kv_values[i] = malloc(sizeof(int32_t));
                memcpy(f->kv_values[i], ptr, sizeof(int32_t));
                ptr += sizeof(int32_t);
                break;
            case GGUF_VAL_TYPE_FLOAT32:
                f->kv_values[i] = malloc(sizeof(float));
                memcpy(f->kv_values[i], ptr, sizeof(float));
                ptr += sizeof(float);
                break;
            case GGUF_VAL_TYPE_BOOL:
                f->kv_values[i] = malloc(sizeof(bool));
                memcpy(f->kv_values[i], ptr, sizeof(bool));
                ptr += sizeof(bool);
                break;
            case GGUF_VAL_TYPE_STRING: {
                uint32_t str_len = read_u32(ptr); ptr += 4;
                f->kv_values[i] = malloc(str_len + 1);
                memcpy(f->kv_values[i], ptr, str_len);
                ((char *)f->kv_values[i])[str_len] = '\0';
                ptr += str_len;
                break;
            }
            case GGUF_VAL_TYPE_UINT64:
                f->kv_values[i] = malloc(sizeof(uint64_t));
                memcpy(f->kv_values[i], ptr, sizeof(uint64_t));
                ptr += sizeof(uint64_t);
                break;
            case GGUF_VAL_TYPE_INT64:
                f->kv_values[i] = malloc(sizeof(int64_t));
                memcpy(f->kv_values[i], ptr, sizeof(int64_t));
                ptr += sizeof(int64_t);
                break;
            case GGUF_VAL_TYPE_FLOAT64:
                f->kv_values[i] = malloc(sizeof(double));
                memcpy(f->kv_values[i], ptr, sizeof(double));
                ptr += sizeof(double);
                break;
            default:
                fprintf(stderr, "[gguf] Unknown KV value type: %u\n", f->kv_types[i]);
                break;
        }
    }
    
    /* Read tensor info array */
    f->tensors = (struct gguf_tensor_info *)calloc(f->n_tensors, sizeof(struct gguf_tensor_info));
    if (!f->tensors) {
        fprintf(stderr, "[gguf] Out of memory allocating tensor info\n");
#ifdef _WIN32
        CloseHandle((HANDLE)f->file_ptr);
#else
        close((int)(intptr_t)f->file_ptr);
#endif
        if (f->use_mmap && f->mmap_ptr) munmap(f->mmap_ptr, f->file_size);
        free(f->kv_keys);
        free(f->kv_values);
        free(f->kv_types);
        free(f);
        return NULL;
    }
    
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        uint32_t name_len = read_u32(ptr); ptr += 4;
        
        if (name_len >= sizeof(f->tensors[i].name)) name_len = sizeof(f->tensors[i].name) - 1;
        memcpy(f->tensors[i].name, ptr, name_len);
        f->tensors[i].name[name_len] = '\0';
        ptr += name_len;
        
        f->tensors[i].n_dims = read_u32(ptr); ptr += 4;
        for (uint32_t d = 0; d < f->tensors[i].n_dims && d < 4; d++) {
            f->tensors[i].dims[d] = read_u64(ptr); ptr += 8;
        }
        
        f->tensors[i].type = (gguf_type_t)read_u32(ptr); ptr += 4;
        f->tensors[i].offset = read_u64(ptr); ptr += 8;
        f->tensors[i].size_bytes = read_u64(ptr); ptr += 8;
    }
    
    return f;
}

void gguf_close(gguf_file_t *f) {
    if (!f) return;
    
#ifdef _WIN32
    if (f->use_mmap && f->mmap_ptr) UnmapViewOfFile(f->mmap_ptr);
    CloseHandle((HANDLE)f->file_ptr);
#else
    if (f->use_mmap && f->mmap_ptr) munmap(f->mmap_ptr, f->file_size);
    close((int)(intptr_t)f->file_ptr);
#endif
    
    /* Free KV metadata */
    for (uint64_t i = 0; i < f->n_kv; i++) {
        free(f->kv_keys[i]);
        free(f->kv_values[i]);
    }
    free(f->kv_keys);
    free(f->kv_values);
    free(f->kv_types);
    
    /* Free tensor info */
    free(f->tensors);
    
    free(f);
}

/* ── Metadata accessors ───────────────────────────────────────────────── */

uint32_t gguf_get_u32(const gguf_file_t *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (strcmp(f->kv_keys[i], key) == 0 && f->kv_types[i] == GGUF_VAL_TYPE_UINT32) {
            uint32_t val;
            memcpy(&val, f->kv_values[i], sizeof(val));
            return val;
        }
    }
    return 0;
}

int64_t gguf_get_i64(const gguf_file_t *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (strcmp(f->kv_keys[i], key) == 0 && f->kv_types[i] == GGUF_VAL_TYPE_INT64) {
            int64_t val;
            memcpy(&val, f->kv_values[i], sizeof(val));
            return val;
        }
    }
    return 0;
}

float gguf_get_f32(const gguf_file_t *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (strcmp(f->kv_keys[i], key) == 0 && f->kv_types[i] == GGUF_VAL_TYPE_FLOAT32) {
            float val;
            memcpy(&val, f->kv_values[i], sizeof(val));
            return val;
        }
    }
    return 0.0f;
}

bool gguf_get_bool(const gguf_file_t *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (strcmp(f->kv_keys[i], key) == 0 && f->kv_types[i] == GGUF_VAL_TYPE_BOOL) {
            bool val;
            memcpy(&val, f->kv_values[i], sizeof(val));
            return val;
        }
    }
    return false;
}

const char *gguf_get_string(const gguf_file_t *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (strcmp(f->kv_keys[i], key) == 0 && f->kv_types[i] == GGUF_VAL_TYPE_STRING) {
            return (const char *)f->kv_values[i];
        }
    }
    return NULL;
}

/* ── Tensor access ────────────────────────────────────────────────────── */

bool gguf_tensor_info(const gguf_file_t *f, const char *name,
                      uint64_t *offset_out, uint64_t *size_bytes_out) {
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            if (offset_out) *offset_out = f->tensors[i].offset;
            if (size_bytes_out) *size_bytes_out = f->tensors[i].size_bytes;
            return true;
        }
    }
    return false;
}

uint64_t gguf_read_tensor(const gguf_file_t *f, const char *name, void *buf, size_t buf_size) {
    uint64_t offset = 0, size = 0;
    if (!gguf_tensor_info(f, name, &offset, &size)) return 0;
    
    if (size > buf_size) return 0;
    
#ifdef _WIN32
    HANDLE hFile = (HANDLE)f->file_ptr;
    SetFilePointer(hFile, (DWORD)offset, NULL, FILE_BEGIN);
    DWORD bytesRead;
    ReadFile(hFile, buf, (DWORD)size, &bytesRead, NULL);
    return bytesRead;
#else
    int fd = (int)(intptr_t)f->file_ptr;
    lseek(fd, offset, SEEK_SET);
    return read(fd, buf, size);
#endif
}

/* ── Weight loading convenience ───────────────────────────────────────── */

size_t gguf_load_all_weights(gguf_file_t *f, float *out_buffer) {
    size_t total_elements = 0;
    
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const struct gguf_tensor_info *t = &f->tensors[i];
        
        /* Calculate element count */
        uint64_t elem_count = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            elem_count *= t->dims[d];
        }
        
        /* For now, assume F32 tensors. Real implementation would handle quantization */
        if (t->type == GGUF_TYPE_F32) {
            uint64_t bytes_to_read = elem_count * sizeof(float);
            
#ifdef _WIN32
            HANDLE hFile = (HANDLE)f->file_ptr;
            SetFilePointer(hFile, (DWORD)t->offset, NULL, FILE_BEGIN);
            DWORD bytesRead;
            ReadFile(hFile, out_buffer + total_elements, (DWORD)bytes_to_read, &bytesRead, NULL);
#else
            int fd = (int)(intptr_t)f->file_ptr;
            lseek(fd, t->offset, SEEK_SET);
            read(fd, out_buffer + total_elements, bytes_to_read);
#endif
            
            total_elements += elem_count;
        } else {
            /* Skip non-F32 tensors for now (would need dequantization) */
            fprintf(stderr, "[gguf] Skipping tensor %s (type %u not supported)\n", 
                    t->name, t->type);
        }
    }
    
    return total_elements;
}

uint64_t gguf_total_params(const gguf_file_t *f) {
    uint64_t total = 0;
    
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const struct gguf_tensor_info *t = &f->tensors[i];
        
        uint64_t elem_count = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            elem_count *= t->dims[d];
        }
        
        /* Adjust for quantization block size */
        switch (t->type) {
            case GGUF_TYPE_Q4_0:
            case GGUF_TYPE_Q4_1:
                elem_count /= 32; /* Q4 blocks of 32 elements */
                break;
            case GGUF_TYPE_Q5_0:
            case GGUF_TYPE_Q5_1:
                elem_count /= 32;
                break;
            case GGUF_TYPE_Q8_0:
            case GGUF_TYPE_Q8_1:
                elem_count /= 32;
                break;
            default:
                break;
        }
        
        total += elem_count;
    }
    
    return total;
}
